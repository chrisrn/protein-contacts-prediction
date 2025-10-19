import os
import torch
import argparse
from tqdm import tqdm
from Bio import pairwise2
from utils import extract_sequence_from_pdb


def compute_seq_identity(seq1, seq2):
    """
    Compute normalized sequence identity between two amino acid sequences.

    The function performs a global pairwise alignment using identity scoring
    (globalxx mode from Biopython) and normalizes the alignment score by the
    length of the longer sequence.

    Parameters
    ----------
    seq1 : str
        The first amino acid sequence.
    seq2 : str
        The second amino acid sequence.

    Returns
    -------
    float
        Normalized sequence identity in the range [0, 1].
    """
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
    max_len = max(len(seq1), len(seq2))
    return alignments / max_len if max_len > 0 else 0.0


def enrich_embeddings_with_structural_similarity(emb_dir, pdb_dir, struct_dir, out_dir, max_residues, top_k=3):
    """
    Enrich ESM2 residue embeddings by integrating structural context from sequence-similar proteins.

    This function enhances residue-level embeddings by incorporating averaged structural
    features from the most sequence-similar proteins. The enrichment process involves:

    1. Extracting amino acid sequences from all PDB files.
    2. Computing pairwise sequence similarities to find the top-K similar proteins.
    3. Averaging the structural feature tensors of those top-K similar proteins.
    4. Concatenating the original ESM2 embeddings, the target protein's own structural
       features, and the averaged neighbor features to produce enriched per-residue embeddings.

    Parameters
    ----------
    emb_dir : str
        Directory containing base ESM2 embeddings (as `.pt` tensors, shape [L, D_emb]).
    pdb_dir : str
        Directory containing PDB files corresponding to the embeddings.
    struct_dir : str
        Directory containing structural feature tensors (as `.pt` files, shape [L, D_struct]).
    out_dir : str
        Directory where enriched embeddings will be saved.
    max_residues : int
        Maximum number of residues to process per protein (used to truncate sequences).
    top_k : int, optional
        Number of top sequence-similar proteins used for structural enrichment. Default is 3.

    Notes
    -----
    - If no suitable similar proteins are found, the enrichment falls back to concatenating
      only the base embeddings and the target’s own structural features.
    - The enrichment is performed independently for each dataset split (e.g., train/test).

    Returns
    -------
    None
        Saves enriched embeddings as `.pt` files in the specified output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    sequences = {}

    print("Extracting sequences...")
    for pdb_file in tqdm(pdb_files):
        seq = extract_sequence_from_pdb(os.path.join(pdb_dir, pdb_file))
        sequences[pdb_file] = seq[:max_residues]

    print("Enriching embeddings using sequence similarity...")
    for pdb_file in tqdm(pdb_files):
        base_name = pdb_file.replace(".pdb", "")
        out_path = os.path.join(out_dir, f"{base_name}.pt")
        if not os.path.exists(out_path):
            emb_path = os.path.join(emb_dir, f"{base_name}.pt")
            struct_path = os.path.join(struct_dir, f"{base_name}.pt")

            if not (os.path.exists(emb_path) and os.path.exists(struct_path)):
                continue

            emb = torch.load(emb_path, weights_only=True)
            struct_feats = torch.load(struct_path, weights_only=True)

            L = min(emb.shape[0], struct_feats.shape[0])
            emb = emb[:L]
            struct_feats = struct_feats[:L]

            seq = sequences[pdb_file]
            similarities = []

            # compute similarity to all other proteins (simple brute-force for small dataset)
            for other_file, other_seq in sequences.items():
                if other_file == pdb_file:
                    continue
                sim = compute_seq_identity(seq, other_seq)
                similarities.append((other_file, sim))

            # take top-K similar
            top_k_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

            # load and average their structural embeddings
            neighbor_structs = []
            for other_file, sim in top_k_similar:
                path = os.path.join(struct_dir, f"{other_file.replace('.pdb', '')}.pt")
                if os.path.exists(path):
                    other_feats = torch.load(path, weights_only=True)
                    # Align all neighbor tensors to same residue length (min length)
                    min_len = min(L, other_feats.shape[0])
                    neighbor_structs.append(other_feats[:min_len])

            if neighbor_structs:
                # Find the minimum residue length across all tensors
                min_len = min([n.shape[0] for n in neighbor_structs] + [L, struct_feats.shape[0]])
                # Truncate all tensors to match
                emb = emb[:min_len]
                struct_feats = struct_feats[:min_len]
                neighbor_structs = [n[:min_len] for n in neighbor_structs]
                neighbor_avg = torch.stack(neighbor_structs).mean(dim=0)
                enriched_emb = torch.cat([emb, struct_feats, neighbor_avg], dim=-1)
            else:
                # Fall back if no neighbors found
                min_len = min(L, struct_feats.shape[0])
                emb = emb[:min_len]
                struct_feats = struct_feats[:min_len]
                enriched_emb = torch.cat([emb, struct_feats], dim=-1)

            torch.save(enriched_emb, out_path)

    print(f"Enriched embeddings saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich ESM2 embeddings using sequence similarity and structural data")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory with ESM2 embeddings (.pt)")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory with corresponding PDB files")
    parser.add_argument("--struct_dir", type=str, required=True, help="Directory with structural feature tensors")
    parser.add_argument("--output_dir", type=str, default="embeddings/enriched_embeddings_pairwise",
                        help="Output directory for enriched embeddings")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K similar sequences to use for enrichment")
    parser.add_argument("--max_residues", type=int, default=50, help="Max residues to process per protein.")
    args = parser.parse_args()

    for split in ["test", "train"]:
        split_emb_dir = os.path.join(args.emb_dir, split)
        split_pdb_dir = os.path.join(args.pdb_dir, split)
        split_struct_dir = os.path.join(args.struct_dir, split)
        split_out_dir = os.path.join(args.output_dir, split)
        enrich_embeddings_with_structural_similarity(split_emb_dir, split_pdb_dir, split_struct_dir,
                                                     split_out_dir, args.max_residues, args.top_k)
