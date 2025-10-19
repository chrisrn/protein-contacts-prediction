import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, is_aa


RESIDUE_MAP = {
    "ALA": "A", "ARG": "R", "ASP": "D", "CYS": "C", "CYX": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "ASN": "N",
    "PHE": "F", "PRO": "P", "SEC": "U", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V"
}

PARSER = PDBParser(QUIET=True)


def extract_sequence_from_pdb(pdb_path):
    """
    Extract the amino acid sequence from a PDB file.

    Uses residue name mapping defined in `RESIDUE_MAP` to convert
    three-letter residue codes to one-letter amino acid codes.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.

    Returns
    -------
    str
        The extracted amino acid sequence as a string of one-letter codes.
        Residues not found in `RESIDUE_MAP` are skipped.
    """
    structure = PARSER.get_structure("protein", pdb_path)
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                resname = residue.get_resname().strip()
                if resname in RESIDUE_MAP:
                    seq.append(RESIDUE_MAP[resname])
    return "".join(seq)


def get_contact_map(pdb_path, L, max_dist=8.0):
    """
    Compute a binary contact map for a protein from its PDB file.

    The contact map indicates whether pairs of residues (based on their
    alpha carbon atom distances) are within a specified distance threshold.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    L : int
        Maximum number of residues to include (to match embedding length).
    max_dist : float, optional
        Distance threshold in Ångströms to define a contact (default is 8.0).

    Returns
    -------
    torch.Tensor
        A tensor of shape (L, L) representing the binary contact map,
        where 1 indicates contact and 0 indicates no contact.
    """
    structure = PARSER.get_structure("protein", pdb_path)
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                # Avoid these resnames because ESM2 is not trained on them
                if resname not in RESIDUE_MAP:
                    continue
                if "CA" not in residue:
                    continue
                coords.append(residue["CA"].get_coord())

    coords = np.array(coords)

    # Ensure we only consider first L residues (match embedding)
    if len(coords) > L:
        coords = coords[:L]

    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    contact_map = (dist_matrix < max_dist).astype(np.float32)
    return torch.tensor(contact_map)


# ======================
# DATASET
# ======================
class ProteinDataset(Dataset):
    """
    Custom PyTorch dataset for protein embeddings and their contact maps.

    This dataset loads ESM2 embeddings (.pt files) and computes corresponding
    contact maps from PDB files for supervised training.

    Parameters
    ----------
    pdb_dir : str
        Directory containing PDB files.
    emb_dir : str
        Directory containing embedding files (.pt).
    max_residues : int, optional
        Maximum number of residues to include per protein (default is 1000).
    max_dist : float, optional
        Distance threshold for defining contacts (default is 8.0 Å).
    """
    def __init__(self, pdb_dir, emb_dir, max_residues=1000, max_dist=8.0):
        self.pdb_dir = pdb_dir
        self.emb_dir = emb_dir
        self.files = [f.replace(".pt", "") for f in os.listdir(emb_dir) if f.endswith(".pt")]
        self.max_residues = max_residues
        self.max_dist = max_dist

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pdb_id = self.files[idx]
        pdb_path = os.path.join(self.pdb_dir, pdb_id + ".pdb")
        emb_path = os.path.join(self.emb_dir, pdb_id + ".pt")

        emb = torch.load(emb_path, weights_only=True)
        L = min(emb.shape[0], self.max_residues)
        emb = emb[:L, :]

        contact_map = get_contact_map(pdb_path, L, self.max_dist)
        return emb, contact_map


# ======================
# MODEL
# ======================
class ContactPredictor(nn.Module):
    """
    Simple feedforward model for predicting protein contact maps
    from residue embeddings.

    The model takes pairwise concatenated residue embeddings and predicts
    whether each residue pair is in contact.

    Parameters
    ----------
    emb_dim : int, optional
        Dimensionality of the input embeddings (default is 64).
    hidden_dim : int, optional
        Dimensionality of the hidden layer (default is 16).
    """
    def __init__(self, emb_dim=64, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        L = embeddings.size(0)
        pairs = torch.cat([
            embeddings.unsqueeze(1).repeat(1, L, 1),
            embeddings.unsqueeze(0).repeat(L, 1, 1)
        ], dim=-1)
        x = torch.relu(self.fc1(pairs))
        x = self.fc2(x).squeeze(-1)
        # x = torch.sigmoid(self.fc2(x)).squeeze(-1)
        return x


def export_metrics_plots(train_metrics, val_metrics, export_dir="runs/plots"):
    """
    Generate and save line plots of training and validation metrics per epoch.

    Parameters
    ----------
    train_metrics : list of dict
        List of metric dictionaries per epoch for the training phase.
    val_metrics : list of dict
        List of metric dictionaries per epoch for the validation phase.
    export_dir : str, optional
        Directory path to save PNG plots (default is 'runs/plots').
    """
    os.makedirs(export_dir, exist_ok=True)
    metric_names = train_metrics[0].keys()

    for metric in metric_names:
        train_values = [m[metric] for m in train_metrics]
        val_values = [m[metric] for m in val_metrics]
        epochs = range(1, len(train_values) + 1)

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_values, marker="o", label="Train")
        plt.plot(epochs, val_values, marker="s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} over epochs")
        plt.legend()

        path = os.path.join(export_dir, f"{metric.lower()}_train_val.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


def add_tensorboard_scalars(writer, metrics, split, epoch):
    """
    Log scalar metrics to TensorBoard for a given epoch and data split.

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard writer instance.
    metrics : dict
        Dictionary containing metric names and their values.
    split : str
        Data split name (e.g., 'Train' or 'Validation').
    epoch : int
        Current epoch number.
    """
    writer.add_scalar(f"Loss/{split}", metrics["Loss"], epoch)
    writer.add_scalar(f"Accuracy/{split}", metrics["Accuracy"], epoch)
    writer.add_scalar(f"Precision/{split}", metrics["Precision"], epoch)
    writer.add_scalar(f"Recall/{split}", metrics["Recall"], epoch)
    writer.add_scalar(f"F1/{split}", metrics["F1"], epoch)
