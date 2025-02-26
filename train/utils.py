import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from itertools import islice
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import os

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, edge_index, labels, _ in train_loader:
        X, edge_index, labels = X.to(device), edge_index.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(X, edge_index[0])  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Convert probabilities to binary predictions
        predicted = (outputs >= 0.5).long()
        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, edge_index, labels, _ in val_loader:
            X, edge_index, labels = X.to(device), edge_index.to(device), labels.to(device)
            labels = labels.float()  # Ensure labels are float for BCELoss

            outputs = model(X, edge_index[0])  # Forward pass
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = (outputs >= 0.5).long()
            all_preds.append(predicted.cpu())
            all_labels.append(labels.long().cpu())

            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=1)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, f1, recall

def train(dataset, 
          model_class, 
          model_params,
          num_epochs=10, 
          optimizer_class=None, 
          optimizer_params=None, 
          expt_name="test",
          max_folds=None, 
          criterion=None):
    """
    Trains a model using cross-validation on a grouped dataset.
    
    Args:
        dataset: PyTorch dataset with `subjects` attribute for grouped cross-validation.
        model_class: Class of the model to be trained.
        model_params: Dictionary of model initialization parameters.
        num_epochs: Number of training epochs.
        optimizer_class: Optimizer class (default: Adam).
        optimizer_params: Dictionary of optimizer parameters.
        expt_name: Name of the experiment for TensorBoard logging.
        max_folds: Maximum number of folds for cross-validation.
        criterion: Loss function (default: BCELoss).
    """
    
    subjects = np.array(dataset.subjects)  
    num_subjects = len(set(subjects))  
    print(f"Total subjects: {num_subjects}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if criterion is None:
        criterion = nn.BCELoss()

    if optimizer_class is None:
        optimizer_class = optim.Adam  # Default to Adam

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}  # Default learning rate

    log_dir = os.path.join("/media/thatblueboy/Seagate/LOP/logs", expt_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Log Hyperparameters
    writer.add_hparams(
        {
            "num_epochs": num_epochs,
            "learning_rate": optimizer_params.get("lr", 1e-3),
            "num_subjects": num_subjects,
            "max_folds": max_folds if max_folds is not None else num_subjects,
            **{f"model_param_{k}": v for k, v in model_params.items()},
            **{f"opt_param_{k}": v for k, v in optimizer_params.items()}
        },
        {}
    )

    gkf = GroupKFold(n_splits=num_subjects)
    fold_iterator = gkf.split(range(len(subjects)), groups=subjects)
    fold_iterator = islice(enumerate(fold_iterator), max_folds) if max_folds is not None else enumerate(fold_iterator)

    # Store per-fold metrics
    f1_scores, accuracies, recalls = [], [], []

    for fold_idx, (train_idx, val_idx) in fold_iterator:
        print(f"Fold {fold_idx+1}/{num_subjects}: Training on {len(train_idx)} samples, Testing on {len(val_idx)} samples")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=24, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False)

        model = model_class(**model_params) if isinstance(model_params, dict) else model_class(model_params)
        model.to(device)

        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        for epoch in range(num_epochs):  
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_recall = evaluate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, "
                  f"F1={val_f1:.4f}, Recall={val_recall:.4f}")

            writer.add_scalar(f"Fold{fold_idx+1}/Train Loss", train_loss, epoch)
            writer.add_scalar(f"Fold{fold_idx+1}/Train Accuracy", train_acc, epoch)
            writer.add_scalar(f"Fold{fold_idx+1}/Val Loss", val_loss, epoch)
            writer.add_scalar(f"Fold{fold_idx+1}/Val Accuracy", val_acc, epoch)
            writer.add_scalar(f"Fold{fold_idx+1}/F1 Score", val_f1, epoch)
            writer.add_scalar(f"Fold{fold_idx+1}/Recall", val_recall, epoch)

        # Store fold results
        f1_scores.append(val_f1)
        accuracies.append(val_acc)
        recalls.append(val_recall)

    # Compute average metrics
    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(accuracies)
    avg_recall = np.mean(recalls)

    # Create Markdown Table
    table = f"""
    | Fold | Accuracy | F1 Score | Recall |
    |------|---------|---------|--------|
    """
    for i, (acc, f1, recall) in enumerate(zip(accuracies, f1_scores, recalls)):
        table += f"| {i+1} | {acc:.4f} | {f1:.4f} | {recall:.4f} |\n"
    
    # Add average metrics
    table += f"| **Average** | **{avg_acc:.4f}** | **{avg_f1:.4f}** | **{avg_recall:.4f}** |\n"

    # Log per-fold metrics to TensorBoard
    for i, (acc, f1, recall) in enumerate(zip(accuracies, f1_scores, recalls)):
        writer.add_scalars("Per-Fold Metrics", {
            "Accuracy": acc,
            "F1 Score": f1,
            "Recall": recall
        }, i + 1)

    # Log final average metrics
    writer.add_scalars("Average Metrics", {
        "Accuracy": avg_acc,
        "F1 Score": avg_f1,
        "Recall": avg_recall
    }, num_subjects + 1)
    
        # Log final table as text in TensorBoard
    writer.add_text("Cross-Validation Results", table)

    print("\nFinal Average Metrics:")
    print(table)

    writer.close()
    print("Training completed!")