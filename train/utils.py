import os
from itertools import islice

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.stats import mode
from sklearn.model_selection import GroupKFold
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, edge_index, labels, _ in train_loader:
        X, edge_index, labels = X.to(device), edge_index.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(X, edge_index[0])  # Forward pass, edge indexed to remove batch
        
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Convert probabilities to binary predictions
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).long()
        else:
            predicted = (outputs >= 0.5).long()
        # predicted = (outputs >= 0.5).long()

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
            
            # If using BCEWithLogitsLoss, apply sigmoid for thresholding
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).long()
            else:
                predicted = (outputs >= 0.5).long()

            all_preds.append(predicted.cpu())
            all_labels.append(labels.long().cpu())

            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, f1, recall, precision

def LOSO_CV(dataset, 
          model_class, 
          model_params,
          num_epochs=10, 
          optimizer_class=None, 
          optimizer_params=None, 
          expt_name="test",
          max_folds=None, 
          criterion=None,
          device =None,
          saved_model_path=None,
          balanced_sampling=False,
          save_path=None):
    """
    Leave One Subject Out(LOSO) cross validation

    Args:
        dataset: PyTorch dataset with `subjects` attribute
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
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if criterion is None:
        criterion = nn.BCELoss()

    if optimizer_class is None:
        optimizer_class = optim.Adam  # Default to Adam

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}  # Default learning rate

    if save_path is None:
        log_dir = os.path.join("/media/thatblueboy/Seagate/LOP/logs", expt_name) #TODO: relative path
    else:
        log_dir = os.path.join(save_path, expt_name)

    writer = SummaryWriter(log_dir=log_dir)

    # Log Hyperparameters
    for k, v in {
    "num_epochs": num_epochs,
    "learning_rate": optimizer_params.get("lr", 1e-3),
    "num_subjects": num_subjects,
    "max_folds": max_folds if max_folds is not None else num_subjects,
    **{f"model_param_{k}": v for k, v in model_params.items()},
    **{f"opt_param_{k}": v for k, v in optimizer_params.items()}
    }.items():writer.add_scalar(f"hparams/{k}", v, 0)  # Log hyperparams under "hparams/"

    gkf = GroupKFold(n_splits=num_subjects)
    fold_iterator = gkf.split(range(len(subjects)), groups=subjects)
    fold_iterator = islice(enumerate(fold_iterator), max_folds) if max_folds is not None else enumerate(fold_iterator)

    # Store per-fold metrics
    f1_scores, accuracies, recalls, precisions = [], [], [], []

    for fold_idx, (train_idx, val_idx) in fold_iterator:
        print(f"Fold {fold_idx+1}/{num_subjects}: Training on {len(train_idx)} samples, Testing on {len(val_idx)} samples")

        if balanced_sampling:
            train_labels = torch.tensor([dataset[i][2].item() for i in train_idx], dtype=torch.long)  
            class_counts = torch.bincount(train_labels)
            class_counts = class_counts.float() + 1e-6  
            class_weights = 1.0 / class_counts
            weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=24, sampler=sampler)
        else:
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=24, shuffle=True)

        val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False)

        train_labels = torch.tensor([dataset[i][2] for i in train_idx])  
        val_labels = torch.tensor([dataset[i][2] for i in val_idx])

        num_train_ones = (train_labels == 1).sum().item()
        num_train_zeros = (train_labels == 0).sum().item()
        num_val_ones = (val_labels == 1).sum().item()
        num_val_zeros = (val_labels == 0).sum().item()

        print(f"Train set: {num_train_ones} ones, {num_train_zeros} zeros")
        print(f"Val set: {num_val_ones} ones, {num_val_zeros} zeros")

        writer.add_scalar(f"Commons/Train Ones", num_train_ones, fold_idx+1)
        writer.add_scalar(f"Commons/Train Zeros", num_train_zeros, fold_idx+1)
        writer.add_scalar(f"Commons/Val Ones", num_val_ones, fold_idx+1)
        writer.add_scalar(f"Commons/Val Zeros", num_val_zeros, fold_idx+1)
        
        model = model_class(**model_params) if isinstance(model_params, dict) else model_class(model_params)
        if saved_model_path is not None:
            model.load_state_dict(torch.load(saved_model_path))

        model.to(device)

        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        for epoch in range(num_epochs):  
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_recall, val_precision = evaluate(model, val_loader, criterion, device)

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
            writer.add_scalar(f"Fold{fold_idx+1}/Precision", val_precision, epoch)

        # Store fold results
        writer.add_scalar(f"Commons/F1", val_f1, fold_idx+1)
        writer.add_scalar(f"Commons/Accuracy", val_acc, fold_idx+1)
        writer.add_scalar(f"Commons/Recall", val_recall, fold_idx+1)
        writer.add_scalar(f"Commons/Precision", val_precision, fold_idx+1)

        f1_scores.append(val_f1)
        accuracies.append(val_acc)
        recalls.append(val_recall)
        precisions.append(val_precision)

    # Compute average metrics
    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(accuracies)
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    # Create Markdown Table
    table = f"""
    | Fold | Accuracy | F1 Score | Recall |
    |------|---------|---------|--------|
    """
    for i, (acc, f1, recall) in enumerate(zip(accuracies, f1_scores, recalls)):
        table += f"| {i+1} | {acc:.4f} | {f1:.4f} | {recall:.4f} |\n"
    
    # Add average metrics
    table += f"| **Average** | **{avg_acc:.4f}** | **{avg_f1:.4f}** | **{avg_recall:.4f}** |\n"

    writer.add_scalar(f"Average Metrics/Recall", avg_recall, 0)
    writer.add_scalar(f"Average Metrics/Accuracy", avg_acc, 0)
    writer.add_scalar(f"Average Metrics/F1", avg_f1, 0)
    writer.add_scalar(f"Average Metrics/Prevision", avg_precision, 0)
    writer.add_text("Cross-Validation Results", table)

    print("\nFinal Average Metrics:")
    print(table)

    writer.close()
    print("Training completed!")

def KFold(dataset, 
          model_class, 
          model_params,
          num_epochs=10, 
          optimizer_class=None, 
          optimizer_params=None, 
          expt_name="test",
          folds=None,
          max_folds=None, 
          criterion=None,
          device =None,
          batch_size=24,
          balanced_sampling=False,
          saved_model_path=None,
          save_path=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if criterion is None:
        criterion = nn.BCELoss()

    if optimizer_class is None:
        optimizer_class = optim.Adam  # Default to Adam

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}  # Default learning rate

    if save_path is None:
        log_dir = os.path.join("/media/thatblueboy/Seagate/LOP/logs", expt_name) #TODO: relative path
    else:
        log_dir = os.path.join(save_path, expt_name)

    writer = SummaryWriter(log_dir=log_dir)

    # Log Hyperparameters
    for k, v in {
    "num_epochs": num_epochs,
    "learning_rate": optimizer_params.get("lr", 1e-3),
    "num_folds": len(folds),
    "max_folds": max_folds if max_folds is not None else len(folds),
    **{f"model_param_{k}": v for k, v in model_params.items()},
    **{f"opt_param_{k}": v for k, v in optimizer_params.items()}
    }.items():writer.add_scalar(f"hparams/{k}", v, 0)  # Log hyperparams under "hparams/"
    
    subjects = np.array(dataset.subjects)  

    f1_scores, accuracies, recalls, precisions = [], [], [], []

    for fold_idx, val_subjects in enumerate(folds):
        train_idx = np.where(~(np.isin(subjects, val_subjects)))[0]
        val_idx = np.where(np.isin(subjects, val_subjects))[0]

        # Create DataLoaders
        if balanced_sampling:
            train_labels = torch.tensor([dataset[i][2].item() for i in train_idx], dtype=torch.long)  
            class_counts = torch.bincount(train_labels)
            class_counts = class_counts.float() + 1e-6  
            class_weights = 1.0 / class_counts
            weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)

        # train_loader = DataLoader(Subset(dataset, train_idx), batch_size=24, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)


        train_labels = torch.tensor([dataset[i][2] for i in train_idx])  
        val_labels = torch.tensor([dataset[i][2] for i in val_idx])

        num_train_ones = (train_labels == 1).sum().item()
        num_train_zeros = (train_labels == 0).sum().item()
        num_val_ones = (val_labels == 1).sum().item()
        num_val_zeros = (val_labels == 0).sum().item()

        print(f"Train set: {num_train_ones} ones, {num_train_zeros} zeros")
        print(f"Val set: {num_val_ones} ones, {num_val_zeros} zeros")

        writer.add_scalar(f"Commons/Train Ones", num_train_ones, fold_idx+1)
        writer.add_scalar(f"Commons/Train Zeros", num_train_zeros, fold_idx+1)
        writer.add_scalar(f"Commons/Val Ones", num_val_ones, fold_idx+1)
        writer.add_scalar(f"Commons/Val Zeros", num_val_zeros, fold_idx+1)
        
        model = model_class(**model_params) if isinstance(model_params, dict) else model_class(model_params)
        if saved_model_path is not None:
            model.load_state_dict(torch.load(saved_model_path))

        model.to(device)

        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        for epoch in range(num_epochs):  
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_recall, val_precision = evaluate(model, val_loader, criterion, device)

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
            writer.add_scalar(f"Fold{fold_idx+1}/Precision", val_precision, epoch)

        # Store fold results
        writer.add_scalar(f"Commons/F1", val_f1, fold_idx+1)
        writer.add_scalar(f"Commons/Accuracy", val_acc, fold_idx+1)
        writer.add_scalar(f"Commons/Recall", val_recall, fold_idx+1)
        writer.add_scalar(f"Commons/Precision", val_precision, fold_idx+1)

        f1_scores.append(val_f1)
        accuracies.append(val_acc)
        recalls.append(val_recall)
        precisions.append(val_precision)

    # Compute average metrics
    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(accuracies)
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    # Create Markdown Table
    table = f"""
    | Fold | Accuracy | F1 Score | Recall |
    |------|---------|---------|--------|
    """
    for i, (acc, f1, recall) in enumerate(zip(accuracies, f1_scores, recalls)):
        table += f"| {i+1} | {acc:.4f} | {f1:.4f} | {recall:.4f} |\n"
    
    # Add average metrics
    table += f"| **Average** | **{avg_acc:.4f}** | **{avg_f1:.4f}** | **{avg_recall:.4f}** |\n"

    writer.add_scalar(f"Average Metrics/Recall", avg_recall, 0)
    writer.add_scalar(f"Average Metrics/Accuracy", avg_acc, 0)
    writer.add_scalar(f"Average Metrics/F1", avg_f1, 0)
    writer.add_scalar(f"Average Metrics/Prevision", avg_precision, 0)
    writer.add_text("Cross-Validation Results", table)

    print("\nFinal Average Metrics:")
    print(table)

    writer.close()
    print("Training completed!")

def train_one_fold(dataset, num_epochs, train_subjects, val_subjects, model_class, model_parameters):
    """Trains and evaluates on one fold with specified train/val subjects and plots results."""
    
    # Get indices for train/val subjects
    subjects = np.array(dataset.subjects)  
    train_idx = np.where(np.isin(subjects, train_subjects))[0]
    val_idx = np.where(np.isin(subjects, val_subjects))[0]

    # Create DataLoaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=24, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=24, shuffle=False)

    # Initialize model, optimizer, loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    print(f"Training on subjects: {train_subjects}")
    print(f"Validating on subjects: {val_subjects}")

    # Lists to store metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []

    # Training loop
    for epoch in range(num_epochs):  
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, "
              f"F1={val_f1:.4f}")
    
    # Plot results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
    ax1.plot(range(1, num_epochs+1), val_losses, 'r-', label='Val Loss')
    ax2.plot(range(1, num_epochs+1), train_accs, 'g-', label='Train Acc')
    ax2.plot(range(1, num_epochs+1), val_accs, 'm-', label='Val Acc')
    ax2.plot(range(1, num_epochs+1), val_f1s, 'c-', label='Val F1')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy / F1')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Training Progress')
    plt.show()
    
    return model

# def create_4_subject_independent_folds(subjects, labels):
#     """
#     Automatically generate 4 subject-independent folds with balanced class distribution.

#     Args:
#         subjects: List or array of subject IDs.
#         labels: List or array of corresponding labels (0 or 1).
    
#     Returns:
#         List of 4 folds, where each fold is a list of subject IDs.
#     """
#     unique_subjects = np.unique(subjects)
#     subject_stats = {}

#     for subject in unique_subjects:
#         mask = (subjects == subject)
#         num_zeros = np.sum(labels[mask] == 0)
#         num_ones = np.sum(labels[mask] == 1)
#         total_samples = num_zeros + num_ones
#         subject_stats[subject] = {'0s': num_zeros, '1s': num_ones, 'total': total_samples}

#     subjects_sorted = sorted(subject_stats.items(), key=lambda x: x[1]['total'], reverse=True)

#     num_folds = 4
#     folds = [[] for _ in range(num_folds)]
#     fold_counts = [{'0s': 0, '1s': 0, 'total': 0} for _ in range(num_folds)]

#     for subject, stats in subjects_sorted:
#         best_fold = min(range(num_folds), key=lambda f: fold_counts[f]['total'])
#         folds[best_fold].append(subject)
#         fold_counts[best_fold]['0s'] += stats['0s']
#         fold_counts[best_fold]['1s'] += stats['1s']
#         fold_counts[best_fold]['total'] += stats['total']

#     for i, fold in enumerate(folds):
#         print(f"Fold {i+1}: Subjects {fold}, Samples: {fold_counts[i]['total']}, Label Counts: {fold_counts[i]}")
        
#     return folds

def create_4_subject_independent_folds(subjects, labels, n_splits=4, random_state=42):
    """
    Create subject-independent folds with balanced class distribution using StratifiedGroupKFold.

    Args:
        subjects (array-like): Subject IDs.
        labels (array-like): Corresponding labels (0 or 1).
        n_splits (int): Number of folds (default=4).
    
    Returns:
        List of folds, where each fold is a list of subject IDs.
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    unique_subjects = np.unique(subjects)
    subject_to_label = {}
    
    for subj in unique_subjects:
        mask = (subjects == subj)
        subject_labels = labels[mask]
        if len(subject_labels) > 0:
            subject_to_label[subj] = mode(subject_labels, keepdims=True).mode[0]  # Ensure correct mode extraction
    
    subjects_arr = np.array(list(subject_to_label.keys()))
    labels_arr = np.array(list(subject_to_label.values()))  # Ensure categorical labels
    
    folds = [[] for _ in range(n_splits)]
    # fold_counts = [{'0s': 0, '1s': 0, 'total': 0} for _ in range(n_splits)]
    
    for fold_idx, (_, test_idx) in enumerate(skf.split(subjects_arr, labels_arr, groups=subjects_arr)):
        selected_subjects = subjects_arr[test_idx].tolist()
        print(selected_subjects)
        folds[fold_idx] = selected_subjects
        
    #     for subj in selected_subjects:
    #         mask = (subjects == subj)
    #         num_zeros = np.sum(labels[mask] == 0)
    #         num_ones = np.sum(labels[mask] == 1)
    #         fold_counts[fold_idx]['0s'] += num_zeros
    #         fold_counts[fold_idx]['1s'] += num_ones
    #         fold_counts[fold_idx]['total'] += num_zeros + num_ones
    
    # # Print fold distributions
    # for i, fold in enumerate(folds):
    #     print(f"Fold {i+1}: {len(fold)} subjects, {fold_counts[i]['total']} samples, {fold_counts[i]['0s']} class 0, {fold_counts[i]['1s']} class 1")
    
    return folds

def train_1(dataset, model_class, model_params, train_subjects, test_subjects, num_epochs=10, 
             optimizer_class=None, optimizer_params=None, criterion=None, device=None, 
             batch_size=24, balanced_sampling=False, log_dir="runs/train", save_every_n_epochs=5): 
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if criterion is None:
        criterion = torch.nn.BCELoss()
    
    if optimizer_class is None:
        optimizer_class = torch.optim.Adam
    
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}
    
    writer = SummaryWriter(log_dir)
    model_save_dir = os.path.join(log_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Filter dataset by subjects
    train_indices = [i for i, subj in enumerate(dataset.subjects) if subj in train_subjects]
    test_indices = [i for i, subj in enumerate(dataset.subjects) if subj in test_subjects]
    
    if balanced_sampling:
            train_labels = torch.tensor([dataset[i][2].item() for i in train_indices], dtype=torch.long)  
            class_counts = torch.bincount(train_labels)
            class_counts = class_counts.float() + 1e-6  
            class_weights = 1.0 / class_counts
            weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)
            train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, sampler=sampler)
    else:
            train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    
    model = model_class(**model_params).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):  
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_recall, val_precision = evaluate(model, test_loader, criterion, device)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, F1={val_f1:.4f}, Recall={val_recall:.4f}")
        
        # Save model every 'n' epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            model_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1}.")

    writer.close()
    print("Training completed!")

def train_2(train_dataset, test_dataset, train_subjects, test_subjects, model_class, model_params, 
          num_epochs=10, optimizer_class=None, optimizer_params=None, criterion=None, device=None, 
          batch_size=24, balanced_sampling=False, log_dir="runs/pretrain", 
          save_every_n_epochs=5, model_path=None):  # Accept both datasets and subjects
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Set loss function
    if criterion is None:
        criterion = torch.nn.BCELoss()
    
    # Set optimizer
    if optimizer_class is None:
        optimizer_class = torch.optim.Adam
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}
    
    # Setup for TensorBoard logging
    writer = SummaryWriter(log_dir)
    model_save_dir = os.path.join(log_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Filter train dataset by train subjects
    train_indices = [i for i, subj in enumerate(train_dataset.subjects) if subj in train_subjects]
    
    # Filter test dataset by test subjects
    test_indices = [i for i, subj in enumerate(test_dataset.subjects) if subj in test_subjects]
    
    # Prepare DataLoader for train and test datasets
    if balanced_sampling:
        # Get labels from filtered train dataset
        train_labels = torch.tensor([train_dataset[i][2].item() for i in train_indices], dtype=torch.long)
        
        # Compute class weights for balancing
        class_counts = torch.bincount(train_labels)
        class_counts = class_counts.float() + 1e-6  # Add small constant to avoid division by zero
        class_weights = 1.0 / class_counts
        weights = class_weights[train_labels]
        
        # Create a weighted random sampler
        sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)
        
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model_class(**model_params).to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Load model from checkpoint if provided
    start_epoch = 0
    

    # Initialize optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    print("Starting Training...")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):  
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on the validation set (test set in your case)
        val_loss, val_acc, val_f1, val_recall, val_precision = evaluate(model, test_loader, criterion, device)
        
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, F1={val_f1:.4f}, Recall={val_recall:.4f}")
        
        # Save model every 'n' epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            model_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print(f"Model saved at epoch {epoch+1}.")

    writer.close()
    print("Training completed!")

def evaluate_subjects(dataset, subjects, model_class, model_params, 
                      model_path, criterion=None, device=None, batch_size=24):
    """
    Evaluates a model on a specific subset of subjects in a dataset.

    Args:
        dataset: PyTorch dataset with a .subjects attribute.
        subjects: List of subject identifiers to evaluate on.
        model_class: Class of the model to instantiate.
        model_params: Dictionary of parameters to pass to the model constructor.
        model_path: Path to the saved model checkpoint (.pth).
        criterion: Loss function (default: BCELoss).
        device: PyTorch device (default: auto-detect CUDA if available).
        batch_size: Batch size for evaluation.

    Returns:
        Tuple of (loss, accuracy, F1, recall, precision).
    """


    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set loss function
    if criterion is None:
        criterion = torch.nn.BCELoss()

    # Filter dataset by given subjects
    subject_indices = [i for i, subj in enumerate(dataset.subjects) if subj in subjects]
    eval_loader = DataLoader(Subset(dataset, subject_indices), batch_size=batch_size, shuffle=False)

    # Initialize and load model
    model = model_class(**model_params).to(device)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Evaluate the model
    print(f"Evaluating on {len(subject_indices)} samples from specified subjects...")
    val_loss, val_acc, val_f1, val_recall, val_precision = evaluate(model, eval_loader, criterion, device)

    print(f"Evaluation Results — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
          f"F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}")

    return val_loss, val_acc, val_f1, val_recall, val_precision

class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon  # Avoids division by zero

    def forward(self, y_pred, y_true):
        """
        Computes soft F1 loss.
        :param y_pred: Probabilities (already passed through sigmoid)
        :param y_true: Ground truth labels (binary values 0 or 1)
        :return: F1 loss (to be minimized)
        """
        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)
        
        soft_f1 = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)
        return 1 - soft_f1.mean()  # Maximize F1 by minimizing 1 - F1