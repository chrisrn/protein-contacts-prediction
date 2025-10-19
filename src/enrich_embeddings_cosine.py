import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# CONFIG: number of similar sequences to use
NUM_SIMILAR = 3
# ========================================


def load_embeddings(emb_dir, struct_dir):
    """
    Load ESM2 embeddings and corresponding structural feature tensors.

    This function reads all `.pt` files from the given embedding directory and, when available,
    loads structural feature tensors (also `.pt` files) from the specified structural feature directory.

    Parameters
    ----------
    emb_dir : str
        Path to the directory containing ESM2 embeddings (PyTorch `.pt` files).
    struct_dir : str
        Path to the directory containing structural feature files corresponding to the embeddings.

    Returns
    -------
    tuple of dict
        A tuple `(embeddings, struct_features)` where:
        - `embeddings`: dict mapping protein names to tensors of shape `[L, emb_dim]`
        - `struct_features`: dict mapping protein names to tensors of shape `[L, feat_dim]`
          or `None` if no structural features exist for that protein.
    """
    files = [f for f in os.listdir(emb_dir) if f.endswith(".pt")]
    embeddings = {}
    struct_features = {}

    for f in files:
        name = f.replace(".pt", "")
        emb_path = os.path.join(emb_dir, f)
        emb = torch.load(emb_path, weights_only=True)  # [L, emb_dim]
        embeddings[name] = emb

        # check for structural features
        struct_path = os.path.join(struct_dir, f)
        if os.path.exists(struct_path):
            struct_features[name] = torch.load(struct_path, weights_only=True)  # [L, feat_dim]
        else:
            struct_features[name] = None

    return embeddings, struct_features


def find_similar_proteins(embeddings, target_name, top_k=NUM_SIMILAR):
    """
    Identify the most similar proteins to a target based on cosine similarity of mean embeddings.

    Each protein embedding is averaged along its sequence dimension before computing pairwise cosine similarity.
    The function returns the top-k most similar proteins (excluding the target itself).

    Parameters
    ----------
    embeddings : dict of str -> torch.Tensor
        Dictionary mapping protein names to ESM2 embeddings of shape `[L, emb_dim]`.
    target_name : str
        Name of the target protein for which to find similar proteins.
    top_k : int, optional
        Number of most similar proteins to return (default is NUM_SIMILAR).

    Returns
    -------
    list of str
        List of names of the top-k most similar proteins.
    """
    target_emb = embeddings[target_name].mean(dim=0, keepdim=True)  # [1, emb_dim]
    all_names = list(embeddings.keys())
    all_embs = torch.stack([embeddings[n].mean(dim=0) for n in all_names])  # [N, emb_dim]

    sims = cosine_similarity(target_emb.numpy(), all_embs.numpy())[0]  # [N]
    sorted_idx = np.argsort(-sims)  # descending
    similar_names = []
    for idx in sorted_idx:
        if all_names[idx] != target_name:
            similar_names.append(all_names[idx])
        if len(similar_names) >= top_k:
            break
    return similar_names


def merge_structural_features(target_name, struct_features, similar_names):
    """
    Merge structural features from a target protein and its similar proteins.

    Structural feature tensors are concatenated along the feature dimension (`axis=1`).
    Features from similar proteins are truncated or zero-padded to match the target's length.

    Parameters
    ----------
    target_name : str
        Name of the target protein.
    struct_features : dict of str -> torch.Tensor or None
        Dictionary mapping protein names to their structural feature tensors or None.
    similar_names : list of str
        Names of proteins deemed similar to the target.

    Returns
    -------
    torch.Tensor or None
        A merged structural feature tensor of shape `[L, total_feat_dim]`, or `None` if no features are available.
    """
    L = struct_features[target_name].shape[0] if struct_features[target_name] is not None else None
    feat_dim = 0
    merged_feats = []

    # Collect features of target first
    if struct_features[target_name] is not None:
        merged_feats.append(struct_features[target_name])
        feat_dim += struct_features[target_name].shape[1]

    # Add features from similar proteins (truncated or padded to L)
    for sim_name in similar_names:
        f = struct_features[sim_name]
        if f is not None:
            if L is not None:
                if f.shape[0] > L:
                    f = f[:L]
                elif f.shape[0] < L:
                    pad = np.zeros((L - f.shape[0], f.shape[1]), dtype=np.float32)
                    f = np.vstack([f, pad])
            merged_feats.append(f)
            feat_dim += f.shape[1]

    if merged_feats:
        merged_feats = np.concatenate(merged_feats, axis=1)
        return torch.from_numpy(merged_feats).float()  # [L, feat_dim]
    else:
        return None  # no features


def enrich_embeddings(embeddings, struct_features, output_dir):
    """
    Enrich ESM2 embeddings with structural information from similar proteins.

    For each protein, this function finds its most similar sequences, merges their structural
    features, and concatenates them with the target embedding. The enriched embeddings are saved
    as `.pt` files in the specified output directory.

    Parameters
    ----------
    embeddings : dict of str -> torch.Tensor
        Dictionary mapping protein names to their ESM2 embeddings `[L, emb_dim]`.
    struct_features : dict of str -> torch.Tensor or None
        Dictionary mapping protein names to structural feature tensors `[L, feat_dim]` or `None`.
    output_dir : str
        Directory path where enriched embeddings will be saved.

    Notes
    -----
    - Only proteins with at least one structural feature (own or from similar proteins) are enriched.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name in tqdm(embeddings.keys()):
        output_path = os.path.join(output_dir, name + ".pt")
        if not os.path.exists(output_path):
            emb = embeddings[name]  # [L, emb_dim]

            # Find similar proteins
            similar_names = find_similar_proteins(embeddings, name)

            # Merge structural features
            struct_feats = merge_structural_features(name, struct_features, similar_names)
            L = min(emb.size(0), struct_feats.size(0))
            emb = emb[:L]
            struct_feats = struct_feats[:L]

            if struct_feats is not None:
                new_emb = torch.cat([emb, struct_feats], dim=-1)
            else:
                new_emb = emb

            torch.save(new_emb, output_path)


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich ESM2 embeddings with structural features from similar proteins.")
    parser.add_argument("--emb_dir", type=str, required=True, help="Input directory with embeddings (.pt).")
    parser.add_argument("--struct_dir", type=str, required=True, help="Input directory with structural features (.pt).")
    parser.add_argument("--output_dir", type=str, default="embeddings/enriched_embeddings_cosine",
                        help="Directory to save enriched embeddings.")
    args = parser.parse_args()

    for split in ["test", "train"]:
        split_emb_dir = os.path.join(args.emb_dir, split)
        split_struct_dir = os.path.join(args.struct_dir, split)
        print(f"Loading {split} embeddings...")
        embeddings, struct_features = load_embeddings(split_emb_dir, split_struct_dir)
        print(f"Loaded {len(embeddings)} embeddings")

        print(f"Enriching {split} embeddings...")
        split_out_dir = os.path.join(args.output_dir, split)
        enrich_embeddings(embeddings, struct_features, split_out_dir)
        print("Done! Enriched embeddings saved to:", split_out_dir)
