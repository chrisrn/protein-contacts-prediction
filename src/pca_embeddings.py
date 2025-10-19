import os
import torch
import joblib
from tqdm import tqdm
from sklearn.decomposition import PCA
import argparse


def fit_pca(emb_dir, n_components=64, max_files=5000, save_path="pca_model.joblib"):
    """
    Fit a PCA model using a subset of embedding tensors from a training directory.

    This function loads a limited number of `.pt` embedding files (from the `train`
    subdirectory of `emb_dir`), concatenates them, and fits a PCA model with a specified
    number of components. The trained PCA model is saved to disk using joblib.

    Parameters
    ----------
    emb_dir : str
        Root directory containing `train` and `test` subfolders with embeddings.
    n_components : int, optional
        Number of PCA components to retain (default is 64).
    max_files : int, optional
        Maximum number of embedding files to use for fitting (default is 5000).
    save_path : str, optional
        Path to save the fitted PCA model file (default is "pca_model.joblib").

    Returns
    -------
    sklearn.decomposition.PCA
        The fitted PCA model.

    Raises
    ------
    ValueError
        If no valid embeddings are found in the specified directory.
    """
    train_emb_dir = os.path.join(emb_dir, "train")
    files = [os.path.join(train_emb_dir, f) for f in os.listdir(train_emb_dir) if f.endswith(".pt")]
    files = files[:max_files]
    all_embs = []

    print(f"Loading up to {len(files)} embeddings to fit PCA...")
    for f in tqdm(files):
        emb = torch.load(f, map_location="cpu", weights_only=True)
        if emb.ndim == 2:
            all_embs.append(emb)
    if not all_embs:
        raise ValueError("No valid embeddings found for PCA fitting!")

    X = torch.cat(all_embs, dim=0).numpy()
    print(f"Fitting PCA with {n_components} components on {X.shape[0]} samples...")
    pca = PCA(n_components=n_components)
    pca.fit(X)

    joblib.dump(pca, save_path)
    print(f"PCA model saved at: {save_path}")
    return pca


def apply_pca(emb_dir, out_dir, pca_model_path, n_components=64):
    """
    Apply a trained PCA model to reduce the dimensionality of all embeddings in a dataset.

    This function loads a previously fitted PCA model (or trains one if missing) and applies it
    to all `.pt` embedding tensors in both `train` and `test` directories. The reduced embeddings
    are saved in the specified output directory.

    Parameters
    ----------
    emb_dir : str
        Root directory containing `train` and `test` subdirectories with original embeddings.
    out_dir : str
        Output directory where PCA-reduced embeddings will be saved.
    pca_model_path : str
        Path to the saved PCA model file (.joblib).
    n_components : int, optional
        Number of PCA components (used only if fitting a new model). Default is 64.

    Notes
    -----
    - Each embedding file is assumed to be a 2D tensor of shape (L, D).
    - The PCA-reduced embeddings are saved as `.pt` tensors of shape (L, n_components).
    - If the PCA model file does not exist, it will be trained using data from `emb_dir/train`.
    """
    if os.path.exists(pca_model_path):
        print(f"Loading existing PCA model from {pca_model_path}")
        pca = joblib.load(pca_model_path)
    else:
        print("PCA model not found — fitting a new one from data...")
        pca = fit_pca(emb_dir, n_components=n_components, save_path=pca_model_path)

    for split in ["train", "test"]:
        emb_dir_split = os.path.join(emb_dir, split)
        files = [f for f in os.listdir(emb_dir_split) if f.endswith(".pt")]
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        for f in tqdm(files, desc="Applying PCA"):
            path_in = os.path.join(emb_dir_split, f)
            path_out = f"{out_dir}/{split}/{f}"
            if not os.path.exists(path_out):
                emb = torch.load(path_in, map_location="cpu", weights_only=False)
                emb_np = emb.numpy()
                emb_pca = pca.transform(emb_np)
                emb_reduced = torch.tensor(emb_pca, dtype=torch.float32)
                torch.save(emb_reduced, path_out)

    print(f"Saved reduced embeddings to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply PCA to ESM2 embeddings")
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory containing esm2 embeddings")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save PCA-reduced embeddings")
    parser.add_argument("--pca_model", type=str, default="pca/pca_esm2_128.joblib",
                        help="Path to save/load PCA model")
    parser.add_argument("--n_components", type=int, default=64,
                        help="Number of PCA components")
    parser.add_argument("--max_files", type=int, default=5000,
                        help="Max number of embeddings used for PCA fitting if no model exists")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.pca_model), exist_ok=True)

    if not os.path.exists(args.pca_model):
        fit_pca(args.emb_dir, n_components=args.n_components,
                max_files=args.max_files, save_path=args.pca_model)

    apply_pca(args.emb_dir, args.out_dir, args.pca_model, args.n_components)
