import os
import argparse
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import ContactPredictor, ProteinDataset
from utils import add_tensorboard_scalars, export_metrics_plots


def epoch_metrics(model, optimizer, dataloader, epoch,
                  total_epochs, device, criterion,
                  steps_per_log, mode="train"):
    """
    Run one full epoch of training or evaluation and compute performance metrics.

    This function handles both training and validation phases for a contact prediction model.
    It iterates over all protein samples, performs forward passes, computes the loss,
    updates weights (if in training mode), and accumulates classification metrics such as
    accuracy, precision, recall, and F1 score.

    Parameters
    ----------
    model : torch.nn.Module
        The contact prediction model being trained or evaluated.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters during training.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of embeddings and contact maps.
    epoch : int
        Current epoch index (0-based).
    total_epochs : int
        Total number of training epochs.
    device : str
        Device used for computation ("cuda" or "cpu").
    criterion : torch.nn.Module
        Loss function used for optimization (e.g., BCEWithLogitsLoss).
    steps_per_log : int
        Frequency (in steps) at which to log intermediate losses.
    mode : str, optional
        Either `"train"` or `"eval"`. Default is `"train"`.

    Returns
    -------
    dict
        Dictionary containing averaged metrics for the epoch:
        {
            "Loss": float,
            "Accuracy": float,
            "Precision": float,
            "Recall": float,
            "F1": float
        }
    """
    if mode == "train":
        model.train()
        desc = f"Epoch train {epoch+1}/{total_epochs}"
    else:
        model.eval()
        desc = f"Epoch eval {epoch+1}/{total_epochs}"

    total_loss = 0.0
    tp = fp = tn = fn = 0
    step = 0
    for emb, contact_map in tqdm(dataloader, desc=desc):
        emb, contact_map = emb[0].to(device), contact_map[0].to(device)
        pred = model(emb)
        loss = criterion(pred, contact_map.float())
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

        step += 1
        if step % steps_per_log == 0:
            avg_loss = total_loss / step
            print(f"Loss: {avg_loss:.4f}")

        preds_bin = (torch.sigmoid(pred) > 0.5)
        targets_bin = contact_map.bool()
        tp += ((preds_bin == 1) & (targets_bin == 1)).sum().item()
        tn += ((preds_bin == 0) & (targets_bin == 0)).sum().item()
        fp += ((preds_bin == 1) & (targets_bin == 0)).sum().item()
        fn += ((preds_bin == 0) & (targets_bin == 1)).sum().item()

    # === Compute metrics ===
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    final_loss = total_loss / len(dataloader)

    return {"Loss": final_loss, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def train_model(args):
    """
    Train a contact prediction model using ESM2 embeddings and PDB-derived contact maps.

    This function orchestrates the full training pipeline, including:
    - Loading train/test datasets
    - Initializing the contact predictor model
    - Training over multiple epochs with loss/metric tracking
    - Logging to TensorBoard and saving results to CSV
    - Exporting metric plots and final model weights

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing paths, hyperparameters, and training settings.

    Notes
    -----
    - Each protein sample is processed independently (`batch_size=1`) due to variable sequence lengths.
    - Training and validation metrics are logged per epoch to both CSV and TensorBoard.
    - The model is saved to `<results_dir>/contact_predictor.pth` after training.

    Returns
    -------
    None
        The function saves model weights and logs, but does not return values.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load dataset ===
    train_dataset = ProteinDataset(os.path.join(args.pdb_dir, "train"),
                                   os.path.join(args.emb_dir, "train"),
                                   args.max_residues)
    test_dataset = ProteinDataset(os.path.join(args.pdb_dir, "test"),
                                  os.path.join(args.emb_dir, "test"),
                                  args.max_residues)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === Model setup ===
    sample_emb = torch.load(os.path.join(args.emb_dir, "train", train_dataset.files[0] + ".pt"), weights_only=True)
    emb_dim = sample_emb.shape[1]
    model = ContactPredictor(emb_dim=emb_dim, hidden_dim=args.hidden_dim).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{num_trainable_params} trainable parameters")

    # === Loss, optimizer, logger ===
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sum_writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "tensorboard"))

    all_train_metrics, all_val_metrics = [], []
    with open(os.path.join(args.results_dir, "test_results.csv"), mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1"])
        for epoch in range(args.epochs):
            # === Run epoch on train-test sets ===
            train_metrics = epoch_metrics(model, optimizer, train_loader, epoch,
                                          args.epochs, device, criterion,
                                          args.steps_per_log)
            val_metrics = epoch_metrics(model, optimizer, val_loader, epoch,
                                        args.epochs, device, criterion,
                                        args.steps_per_log, mode="eval")
            all_train_metrics.append(train_metrics)
            all_val_metrics.append(val_metrics)

            # === Log to TensorBoard ===
            add_tensorboard_scalars(sum_writer, train_metrics, "train", epoch)
            add_tensorboard_scalars(sum_writer, val_metrics, "val", epoch)

            # === Write metrics to CSV ===
            csv_writer.writerow([
                epoch + 1,
                f"{val_metrics['Loss']:.4f}",
                f"{val_metrics['Accuracy']:.4f}",
                f"{val_metrics['Precision']:.4f}",
                f"{val_metrics['Recall']:.4f}",
                f"{val_metrics['F1']:.4f}"
                ])

            print(f"\nEpoch {epoch+1}/{args.epochs}")
            for ((metric, train_value), (metric, val_value)) in zip(train_metrics.items(), val_metrics.items()):
                print(f"Train {metric}: {train_value:.4f} | Val {metric}: {val_value:.4f}")

    sum_writer.close()
    export_metrics_plots(all_train_metrics, all_val_metrics, os.path.join(args.results_dir, "plots"))
    model_path = os.path.join(args.results_dir, "contact_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train contact predictor on ESM2 embeddings.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Path to train-test PDB files.")
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to train-test embeddings (.pt).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension size.")
    parser.add_argument("--steps_per_log", type=int, default=1000, help="Per num steps to print loss.")
    parser.add_argument("--max_residues", type=int, default=50, help="Max residues to process per protein.")
    parser.add_argument("--results_dir", type=str, default="results/runs", help="Results directory.")
    args = parser.parse_args()

    train_model(args)
