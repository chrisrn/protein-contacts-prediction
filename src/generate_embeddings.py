import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, is_aa
from esm import pretrained
from utils import extract_sequence_from_pdb

MAX_SEQ_LEN = 10000
PARSER = PDBParser(QUIET=True)


def extract_structural_features(pdb_path, max_neighbors=8.0):
    """
    Extract lightweight structural features for each residue in a protein.

    For each residue (based on CA atoms), this function computes:
    - The Cartesian coordinates (x, y, z)
    - The number of neighboring residues within a specified distance threshold

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    max_neighbors : float, optional
        Distance threshold (in Ångströms) to count neighboring residues.
        Default is 8.0 Å.

    Returns
    -------
    torch.Tensor or None
        A tensor of shape (L, 4) where each row contains:
        [x, y, z, neighbor_count].
        Returns None if no CA atoms are found in the structure.
    """
    structure = PARSER.get_structure("protein", pdb_path)
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":
            coords.append(atom.get_coord())
    coords = np.array(coords, dtype=np.float32)
    if len(coords) == 0:
        return None

    # Neighbor counts
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    neighbor_count = (dist_matrix < max_neighbors).astype(np.float32).sum(axis=1) - 1  # exclude self

    # Combine features: [x, y, z, neighbor_count]
    features = np.concatenate([coords, neighbor_count[:, None]], axis=1)
    return torch.from_numpy(features)


def generate_embeddings_for_dir(model, alphabet, input_dir, output_dir, struct_out_dir):
    """
    Generate and save ESM2 embeddings and structural features for all PDB files in a directory.

    For each PDB file:
    - Extracts the amino acid sequence.
    - Generates ESM2 residue embeddings (if not already saved).
    - Computes structural features (CA coordinates and neighbor counts).
    - Saves both outputs as .pt files for later use.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained ESM2 model used to compute embeddings.
    alphabet : esm.data.Alphabet
        ESM2 model alphabet used for batch conversion.
    input_dir : str
        Directory containing input PDB files.
    output_dir : str
        Directory to save ESM2 embedding tensors (.pt).
    struct_out_dir : str
        Directory to save structural feature tensors (.pt).

    Notes
    -----
    - Skips sequences longer than `MAX_SEQ_LEN`.
    - Skips proteins where no valid residues or CA atoms are found.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(struct_out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
    print(f"Found {len(pdb_files)} PDB files in {input_dir}")

    batch_converter = alphabet.get_batch_converter()
    model.eval()

    skipped = 0
    for pdb_file in tqdm(pdb_files):
        pdb_path = os.path.join(input_dir, pdb_file)
        output_path = os.path.join(output_dir, pdb_file.replace(".pdb", ".pt"))
        struct_output_path = os.path.join(struct_out_dir, pdb_file.replace(".pdb", ".pt"))

        try:
            seq = extract_sequence_from_pdb(pdb_path)
            if len(seq) == 0:
                continue
            if len(seq) > MAX_SEQ_LEN:
                print(f"Skipping {pdb_file} (sequence too long: {len(seq)})")
                skipped += 1
                continue

            # === ESM2 embeddings ===
            if not os.path.exists(output_path):
                batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[6], return_contacts=False)
                token_representations = results["representations"][6][0, 1: len(seq) + 1]
                torch.save(token_representations.cpu(), output_path)

            # === Structural features ===
            if not os.path.exists(struct_output_path):
                struct_feats = extract_structural_features(pdb_path)
                if struct_feats is not None:
                    torch.save(struct_feats.cpu(), struct_output_path)

        except Exception as e:
            print(f"Failed on {pdb_file}: {e}")

    print(f"{skipped} skipped protein sequences")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to train-test folders")
    parser.add_argument("--out_dir", type=str, default="embeddings/cache",
                        help="Directory to save ESM2 features")
    parser.add_argument("--struct_out_dir", type=str, default="embeddings/structural_features",
                        help="Directory to save structural features")
    args = parser.parse_args()

    print("Loading ESM2 model...")
    model, alphabet = pretrained.esm2_t6_8M_UR50D()
    model = model.to("cpu")

    for split in ["test", "train"]:
        input_dir = os.path.join(args.data_dir, split)
        output_dir = os.path.join(args.out_dir, split)
        struct_out_dir = os.path.join(args.struct_out_dir, split)
        generate_embeddings_for_dir(model, alphabet, input_dir, output_dir, struct_out_dir)
