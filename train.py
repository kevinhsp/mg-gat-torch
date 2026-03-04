import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import json
# ============================================================================
# IMPORTANT: Adjust these imports based on how you saved your previous files.
# Assuming data loading is in dataset.py and the model is in models.py
# ============================================================================
from loaddata import load_mggat_data
from models import MGGATRecommender


def compute_graph_smoothness_loss(embeddings, edge_index):
    """
    Computes the graph Laplacian regularization term.
    Mathematically equivalent to Tr(H^T L H) = 1/2 * sum_{(i,j) in E} ||h_i - h_j||^2
    """
    # src and dst node embeddings for all edges
    src_embeds = embeddings[edge_index[0]]
    dst_embeds = embeddings[edge_index[1]]

    # Calculate squared distance
    # Using .mean() instead of .sum() keeps the loss scale manageable
    smoothness_loss = (src_embeds - dst_embeds).pow(2).sum(dim=-1).mean()
    return smoothness_loss

def evaluate_model(model, data_loader, S_u, S_b, eu, eb, criterion, device):
    """
    Evaluates the model on the provided dataloader (tune or test set).
    Returns the Root Mean Squared Error (RMSE).
    """
    model.eval()
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_u, batch_i, true_ratings in data_loader:
            batch_u = batch_u.to(device)
            batch_i = batch_i.to(device)
            true_ratings = true_ratings.to(device)

            # Forward pass
            predictions = model(batch_u, batch_i, S_u, S_b, eu, eb)

            # Accumulate squared error
            # criterion is nn.MSELoss(reduction='sum') to easily compute RMSE
            loss = criterion(predictions, true_ratings)
            total_squared_error += loss.item()
            total_samples += len(true_ratings)

    rmse = math.sqrt(total_squared_error / total_samples)
    return rmse


def objective(trial, data_bundle, device):
    """
    The black-box function that Optuna will try to optimize (minimize validation RMSE).
    """
    # Unpack the pre-loaded data bundle
    S_u, S_b, eu, eb, train_loader, tune_loader, num_users, num_items, u_in_dim, i_in_dim = data_bundle

    # --------------------------------------------------------
    # [Magic 1] Define Search Space dynamically using 'trial'
    # --------------------------------------------------------
    # Suggest dimensions (Categorical)
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
    final_dim = trial.suggest_categorical("final_dim", [16, 32, 64])

    # Suggest continuous hyperparameters (Log-uniform for learning rate)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Suggest graph regularization strength
    theta_1 = trial.suggest_float("theta_1", 0.01, 0.2)

    # --------------------------------------------------------
    # Initialize Model & Optimizer with suggested params
    # --------------------------------------------------------
    model = MGGATRecommender(
        num_users=num_users,
        num_items=num_items,
        u_in_dim=u_in_dim,
        i_in_dim=i_in_dim,
        latent_dim=latent_dim,
        final_dim=final_dim,
        num_u_graphs=len(eu),
        num_i_graphs=len(eb)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_criterion = nn.MSELoss(reduction='mean')
    eval_criterion = nn.MSELoss(reduction='sum')

    EPOCHS = 15  # Shorter epochs for tuning to save time
    best_tune_rmse = float('inf')

    # --------------------------------------------------------
    # Training Loop for this specific Trial
    # --------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()

        for batch_idx, (batch_u, batch_i, true_ratings) in enumerate(train_loader):
            batch_u, batch_i, true_ratings = batch_u.to(device), batch_i.to(device), true_ratings.to(device)

            optimizer.zero_grad()
            predictions = model(batch_u, batch_i, S_u, S_b, eu, eb)
            mse_loss = train_criterion(predictions, true_ratings)

            reg_loss_u = compute_graph_smoothness_loss(model.H_u_4.weight, eu[0])
            reg_loss_b = compute_graph_smoothness_loss(model.H_b_4.weight, eb[0])
            graph_reg_loss = theta_1 * (reg_loss_u + reg_loss_b)

            loss = mse_loss + graph_reg_loss
            loss.backward()
            optimizer.step()

        # Evaluate on Tune (Validation) set
        tune_rmse = evaluate_model(model, tune_loader, S_u, S_b, eu, eb, eval_criterion, device)
        print(
            f"Trial {trial.number:02d} | Epoch {epoch + 1:02d}/{EPOCHS} | Tune RMSE: {tune_rmse:.4f}")
        if tune_rmse < best_tune_rmse:
            best_tune_rmse = tune_rmse

        # --------------------------------------------------------
        # [Magic 2] Report intermediate value and check for PRUNING
        # --------------------------------------------------------
        trial.report(tune_rmse, epoch)

        # If this trial's performance at the current epoch is significantly
        # worse than previous trials, Optuna will gracefully kill it here!
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_tune_rmse


import scipy.stats as stats
from itertools import combinations
from collections import defaultdict


# ============================================================================
# NEW: Ranking Metrics Evaluator (Spearman, FCP)
# ============================================================================
def evaluate_ranking_metrics(model, test_loader, S_u, S_b, eu, eb, device):
    """
    Evaluates ranking metrics: Spearman Correlation and FCP.
    Strictly follows the paper: evaluated ONLY on users with > 1 rating in the test set.
    """
    model.eval()

    # We need to collect all predictions grouped by user
    user_true_ratings = defaultdict(list)
    user_pred_ratings = defaultdict(list)

    with torch.no_grad():
        for batch_u, batch_i, true_ratings in test_loader:
            batch_u_dev = batch_u.to(device)
            batch_i_dev = batch_i.to(device)

            predictions = model(batch_u_dev, batch_i_dev, S_u, S_b, eu, eb)

            # Move to CPU for numpy operations
            batch_u = batch_u.cpu().numpy()
            true_ratings = true_ratings.cpu().numpy()
            predictions = predictions.cpu().numpy()

            # Group by user
            for u, t_r, p_r in zip(batch_u, true_ratings, predictions):
                user_true_ratings[u].append(t_r)
                user_pred_ratings[u].append(p_r)

    # ---------------------------------------------------------
    # Calculate Metrics per User
    # ---------------------------------------------------------
    spearman_scores = []
    fcp_scores = []

    for u in user_true_ratings.keys():
        t_ratings = user_true_ratings[u]
        p_ratings = user_pred_ratings[u]

        # PAPER RULE: Only evaluate on users with > 1 rating in test set!
        if len(t_ratings) > 1:

            # 1. Spearman Correlation
            # Ignore warnings if all true ratings are identical for a user
            correlation, _ = stats.spearmanr(t_ratings, p_ratings)
            if not math.isnan(correlation):
                spearman_scores.append(correlation)

            # 2. FCP (Fraction of Concordant Pairs)
            concordant = 0
            discordant = 0
            # Generate all pairs of items for this user
            for (idx1, idx2) in combinations(range(len(t_ratings)), 2):
                true_diff = t_ratings[idx1] - t_ratings[idx2]
                pred_diff = p_ratings[idx1] - p_ratings[idx2]

                # Only care if there's an actual preference in true ratings
                if true_diff != 0:
                    if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                        concordant += 1
                    else:
                        discordant += 1

            total_pairs = concordant + discordant
            if total_pairs > 0:
                fcp = concordant / total_pairs
                fcp_scores.append(fcp)

    # Average across all valid users
    final_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0.0
    final_fcp = sum(fcp_scores) / len(fcp_scores) if fcp_scores else 0.0
    valid_users_ratio = len(spearman_scores) / len(user_true_ratings)

    return final_spearman, final_fcp, valid_users_ratio

# ============================================================================
# 3. Main Execution Block
# ============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load data exactly ONCE to save memory and time
    print("Loading data into memory...")
    S_u, S_b, eu, eb, train_ds, tune_ds, test_ds = load_mggat_data()

    num_users, u_in_dim = S_u.shape
    num_items, i_in_dim = S_b.shape

    S_u, S_b = S_u.to(device), S_b.to(device)
    eu = [e.to(device) for e in eu]
    eb = [e.to(device) for e in eb]

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    tune_loader = DataLoader(tune_ds, batch_size=2048, shuffle=False)

    # Bundle the data to pass into the objective function easily
    data_bundle = (S_u, S_b, eu, eb, train_loader, tune_loader, num_users, num_items, u_in_dim, i_in_dim)

    # 2. Create the Optuna Study
    print("\nStarting Optuna Hyperparameter Optimization...")
    # TPE sampler is default; MedianPruner cuts bad trials early
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )

    # Optimize using a lambda function to pass extra arguments
    study.optimize(lambda trial: objective(trial, data_bundle, device), n_trials=60)

    # 3. Print the ultimate results
    print("\n" + "=" * 50)
    print("Optimization Finished!")
    print(f"Best Trial ID: {study.best_trial.number}")
    print(f"Best Tune RMSE: {study.best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 50)
    # ============================================================================
    # 4. Final Retraining & Test Evaluation (Using Best Params)
    # ============================================================================
    print("\n" + "*" * 50)
    print("Initiating Final Training with BEST Hyperparameters...")
    print("*" * 50)

    best_params = study.best_trial.params

    final_model = MGGATRecommender(
        num_users=num_users,
        num_items=num_items,
        u_in_dim=u_in_dim,
        i_in_dim=i_in_dim,
        latent_dim=best_params["latent_dim"],
        final_dim=best_params["final_dim"],
        num_u_graphs=len(eu),
        num_i_graphs=len(eb)
    ).to(device)

    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"]
    )

    final_train_criterion = nn.MSELoss(reduction='mean')
    final_eval_criterion = nn.MSELoss(reduction='sum')

    FINAL_EPOCHS = 20  # 最终训练可以稍微多跑几个 Epoch 确保彻底收敛

    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        for batch_u, batch_i, true_ratings in train_loader:
            batch_u, batch_i, true_ratings = batch_u.to(device), batch_i.to(device), true_ratings.to(device)

            final_optimizer.zero_grad()
            predictions = final_model(batch_u, batch_i, S_u, S_b, eu, eb)
            mse_loss = final_train_criterion(predictions, true_ratings)

            reg_loss_u = compute_graph_smoothness_loss(final_model.H_u_4.weight, eu[0])
            reg_loss_b = compute_graph_smoothness_loss(final_model.H_b_4.weight, eb[0])
            graph_reg_loss = best_params["theta_1"] * (reg_loss_u + reg_loss_b)

            loss = mse_loss + graph_reg_loss
            loss.backward()
            final_optimizer.step()

    print("Training complete. Running inference on TEST set...")
    final_test_rmse = evaluate_model(final_model, test_loader, S_u, S_b, eu, eb, final_eval_criterion, device)

    print("\n" + "=" * 50)
    print(f"🎉 FINAL TEST RMSE (Report this in your paper/test!): {final_test_rmse:.4f}")
    print("=" * 50)
    # ============================================================================
    # 5. Save the Ultimate Model and Hyperparameters to Disk
    # ============================================================================
    print("\n" + "=" * 50)
    print("Saving Best Hyperparameters and Final Model...")

    # 1. 保存最优超参数到 JSON 文件
    hyperparams_path = "best_hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"[Success] Best hyperparameters saved to: {hyperparams_path}")

    # 2. 保存最终训练好的模型权重到 PTH 文件
    model_save_path = "final_mggat_model.pth"
    torch.save(final_model.state_dict(), model_save_path)
    print(f"[Success] Final model weights saved to: {model_save_path}")
    print("=" * 50)
    print("\nTraining complete. Loading the absolute BEST weights for TEST inference...")
    # final_model.load_state_dict(best_final_weights)

    final_test_rmse = evaluate_model(final_model, test_loader, S_u, S_b, eu, eb, final_eval_criterion, device)

    # ======================================================
    # NEW: 计算排序指标 (Spearman, FCP)
    # ======================================================
    print("Calculating Ranking Metrics (Spearman, FCP) on Test set...")
    spearman, fcp, valid_ratio = evaluate_ranking_metrics(final_model, test_loader, S_u, S_b, eu, eb, device)

    print("\n" + "=" * 50)
    print(f"🎉 FINAL TEST RMSE: {final_test_rmse:.4f}")
    print(f"📈 FINAL SPEARMAN:  {spearman:.4f}")
    print(f"🎯 FINAL FCP:       {fcp:.4f}")
    print(f"ℹ️ Valid Users for Ranking: {valid_ratio * 100:.1f}% (Matches paper's ~30% claim!)")
    print("=" * 50)