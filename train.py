import os
import time
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse.linalg import svds
from torch.utils.data import DataLoader
import optuna
import json
import scipy.stats as stats
from itertools import combinations
from collections import defaultdict

from loaddata import load_mggat_data
from models import MGGATRecommender
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def compute_graph_regularization(embeddings, edge_indices_list):
    smoothness_loss = 0.0
    total_graphs = len(edge_indices_list)

    for edge_index in edge_indices_list:
        src_embeds = embeddings[edge_index[0]]
        dst_embeds = embeddings[edge_index[1]]

        smoothness_loss += (src_embeds - dst_embeds).pow(2).mean()

    if total_graphs > 0:
        smoothness_loss = smoothness_loss / total_graphs

    return smoothness_loss


def evaluate_model(model, data_loader, S_u, S_b, eu, eb, criterion, device):
    model.eval()
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_u, batch_i, true_ratings in data_loader:
            batch_u, batch_i, true_ratings = batch_u.to(device), batch_i.to(device), true_ratings.to(device)
            predictions = model(batch_u, batch_i, S_u, S_b, eu, eb)
            loss = criterion(predictions, true_ratings)
            total_squared_error += loss.item()
            total_samples += len(true_ratings)

    return math.sqrt(total_squared_error / total_samples)

def generate_implicit_features(train_df, num_users, num_items, latent_dim=20):
    """
    Generate implicit features via SVD on the binarized training rating matrix.
    Matches the methodology described in the paper.
    """
    print(f"Generating SVD implicit features (dim={latent_dim})...")

    # 1. Binarize continuous ratings: any interaction > 0 becomes 1
    row = train_df['user_id'].values
    col = train_df['item_id'].values
    data = np.ones(len(train_df))  # Binarized feedback

    # Create sparse rating matrix
    X_bina = sp.coo_matrix((data, (row, col)), shape=(num_users, num_items)).asfptype()

    # 2. Apply SVD
    k = min(latent_dim, min(num_users, num_items) - 1)
    u, s, vt = svds(X_bina, k=k)

    # 3. Compute implicit embeddings: U * \Sigma^{1/2} and V^T * \Sigma^{1/2}
    s_sqrt = np.diag(np.sqrt(s))
    S_u_imp = np.dot(u, s_sqrt)
    S_b_imp = np.dot(vt.T, s_sqrt)

    return torch.tensor(S_u_imp, dtype=torch.float32), torch.tensor(S_b_imp, dtype=torch.float32)

def evaluate_ranking_metrics(model, test_loader, S_u, S_b, eu, eb, device):
    model.eval()
    user_true_ratings = defaultdict(list)
    user_pred_ratings = defaultdict(list)

    with torch.no_grad():
        for batch_u, batch_i, true_ratings in test_loader:
            batch_u_dev, batch_i_dev = batch_u.to(device), batch_i.to(device)
            predictions = model(batch_u_dev, batch_i_dev, S_u, S_b, eu, eb)

            batch_u = batch_u.cpu().numpy()
            true_ratings = true_ratings.cpu().numpy()
            predictions = predictions.cpu().numpy()

            for u, t_r, p_r in zip(batch_u, true_ratings, predictions):
                user_true_ratings[u].append(t_r)
                user_pred_ratings[u].append(p_r)

    spearman_scores = []
    fcp_scores = []
    bpr_log_sum = 0.0
    bpr_pairs = 0

    def logsigmoid(x):
        if x >= 0:
            return -math.log1p(math.exp(-x))
        else:
            return x - math.log1p(math.exp(x))

    for u, t_ratings in user_true_ratings.items():
        p_ratings = user_pred_ratings[u]
        if len(t_ratings) > 1:
            correlation, _ = stats.spearmanr(t_ratings, p_ratings)
            if not math.isnan(correlation):
                spearman_scores.append(correlation)

            concordant = 0
            discordant = 0

            for (idx1, idx2) in combinations(range(len(t_ratings)), 2):
                true_diff = t_ratings[idx1] - t_ratings[idx2]
                pred_diff = p_ratings[idx1] - p_ratings[idx2]

                if true_diff != 0:
                    if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                        concordant += 1
                    else:
                        discordant += 1

                    if true_diff > 0:
                        bpr_log_sum += logsigmoid(pred_diff)
                        bpr_pairs += 1
                    else:
                        bpr_log_sum += logsigmoid(-pred_diff)
                        bpr_pairs += 1

            total_pairs = concordant + discordant
            if total_pairs > 0:
                fcp_scores.append(concordant / total_pairs)

    final_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0.0
    final_fcp = sum(fcp_scores) / len(fcp_scores) if fcp_scores else 0.0

    final_bpr = math.exp(bpr_log_sum / bpr_pairs) if bpr_pairs > 0 else 0.0

    valid_users_ratio = len(spearman_scores) / len(user_true_ratings)

    return final_spearman, final_fcp, final_bpr, valid_users_ratio

BEST_RESULTS_PATH = "best_results.json"
BEST_MODEL_PATH   = "best_model.pth"
BEST_PARAMS_PATH  = "best_hyperparameters.json"

def run_optuna_search(data_bundle, device, n_trials):
    S_u_exp, S_b_exp, eu, eb, train_loader, tune_loader, num_users, num_items, u_in_dim, i_in_dim, train_df = data_bundle
    activations = [
        'elu', 'hard_sigmoid', 'linear', 'relu',
        'selu', 'sigmoid', 'softplus', 'softsign', 'tanh'
    ]
    if os.path.exists(BEST_RESULTS_PATH):
        with open(BEST_RESULTS_PATH, "r") as f:
            best_results = json.load(f)
        best_test_rmse = best_results["test_rmse"]
        print(f"load history best Test RMSE: {best_test_rmse:.4f} (Trial {best_results.get('trial_number', '?')})")
    else:
        best_results = {}
        best_test_rmse = float('inf')
    state = {"best_test_rmse": best_test_rmse, "best_results": best_results}
    def objective(trial):
        latent_dim = trial.suggest_int("latent_dim", 8, 128, log=True)
        final_dim = trial.suggest_int("final_dim", 8, 128, log=True)
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
        theta_1 = trial.suggest_float("theta_1", 1e-4, 0.5,log=True)
        activation_in = trial.suggest_categorical("activation_in", activations)
        activation_out = trial.suggest_categorical("activation_out", activations)

        implicit_dim = trial.suggest_int("implicit_dim", 1, 20,log = True)

        S_u_imp, S_b_imp = generate_implicit_features(train_df, num_users, num_items, latent_dim=implicit_dim)

        S_u = torch.cat([S_u_exp.to(device), S_u_imp.to(device)], dim=1)
        S_b = torch.cat([S_b_exp.to(device), S_b_imp.to(device)], dim=1)

        u_in_dim = S_u.shape[1]
        i_in_dim = S_b.shape[1]
        model = MGGATRecommender(
            num_users=num_users, num_items=num_items, u_in_dim=u_in_dim, i_in_dim=i_in_dim,
            latent_dim=latent_dim, final_dim=final_dim, num_u_graphs=len(eu), num_i_graphs=len(eb),
            actv_in_name=activation_in,
            actv_out_name=activation_out
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_criterion = nn.MSELoss(reduction='mean')
        eval_criterion = nn.MSELoss(reduction='sum')


        OPTUNA_EPOCHS = 100
        EARLY_STOP_PATIENCE = 15
        best_tune_rmse = float('inf')
        patience_counter = 0

        for epoch in range(OPTUNA_EPOCHS):
            model.train()
            for batch_u, batch_i, true_ratings in train_loader:
                batch_u, batch_i, true_ratings = batch_u.to(device), batch_i.to(device), true_ratings.to(device)

                optimizer.zero_grad()
                predictions = model(batch_u, batch_i, S_u, S_b, eu, eb)
                mse_loss = train_criterion(predictions, true_ratings)

                reg_loss_u = compute_graph_regularization(model.H_u_4.weight, eu)
                reg_loss_b = compute_graph_regularization(model.H_b_4.weight, eb)
                graph_reg_loss = theta_1 * (reg_loss_u + reg_loss_b)

                loss = mse_loss + graph_reg_loss
                loss.backward()
                optimizer.step()

            tune_rmse = evaluate_model(model, tune_loader, S_u, S_b, eu, eb, eval_criterion, device)

            if tune_rmse < best_tune_rmse:
                best_tune_rmse = tune_rmse
                best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

            trial.report(tune_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        temp_path = f"temp_trial_{trial.number}.pth"
        best_tune = study.best_trial.value
        threshold = best_tune * 1.003
        if best_tune_rmse <= threshold:
            torch.save({
                "best_state_dict": best_state_dict,
                "S_u_imp": S_u_imp.cpu(),
                "S_b_imp": S_b_imp.cpu()
            }, temp_path)

            trial.set_user_attr("temp_path", temp_path)
        return best_tune_rmse

    def eval_on_new_best(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        best_tune = study.best_trial.value
        threshold = best_tune * 1.003
        if trial.value > threshold:
            return
        temp_path = trial.user_attrs.get("temp_path")
        checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)
        S_u_imp = checkpoint["S_u_imp"]
        S_b_imp = checkpoint["S_b_imp"]
        best_state_dict = checkpoint["best_state_dict"]

        S_u = torch.cat([S_u_exp.to(device), S_u_imp.to(device)], dim=1)
        S_b = torch.cat([S_b_exp.to(device), S_b_imp.to(device)], dim=1)
        u_in_dim, i_in_dim = S_u.shape[1], S_b.shape[1]

        # print(f"\n=== Trial {trial.number} is new Possible Best Tune RMSE ({trial.value:.4f}) Start training ===")
        # model, test_rmse, spearman, fcp, valid_ratio,S_u_imp, S_b_imp = train_and_eval(
        #     trial.params, data_bundle, test_loader, device
        # )

        model = MGGATRecommender(
            num_users=num_users, num_items=num_items, u_in_dim=u_in_dim, i_in_dim=i_in_dim,
            latent_dim=trial.params["latent_dim"], final_dim=trial.params["final_dim"],
            num_u_graphs=len(eu), num_i_graphs=len(eb),
            actv_in_name=trial.params["activation_in"],
            actv_out_name=trial.params["activation_out"]
        ).to(device)

        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

        eval_criterion = nn.MSELoss(reduction='sum')
        test_rmse = evaluate_model(model, test_loader, S_u, S_b, eu, eb, eval_criterion, device)
        spearman, fcp, bpr, valid_ratio = evaluate_ranking_metrics(model, test_loader, S_u, S_b, eu, eb, device)

        print(f"  Test RMSE: {test_rmse:.4f} | Spearman: {spearman:.4f} | FCP: {fcp:.4f}")

        if test_rmse < state["best_test_rmse"]:
            state["best_test_rmse"] = test_rmse
            state["best_results"] = {
                "test_rmse": test_rmse,
                "spearman": spearman,
                "fcp": fcp,
                "bpr": bpr,
                "valid_ratio": valid_ratio,
                "tune_rmse": trial.value,
                "trial_number": trial.number,
                "params": trial.params,
            }
            with open(BEST_RESULTS_PATH, "w") as f:
                json.dump(state["best_results"], f, indent=4)
            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(trial.params, f, indent=4)

            torch.save({
                "model_state_dict": model.state_dict(),
                "params": trial.params,
                "S_u_imp": S_u_imp,
                "S_b_imp": S_b_imp,
                "num_i_graphs": len(eb),
                "num_u_graphs": len(eu),
            }, BEST_MODEL_PATH)

            print(f"  new Best Test RMSE: {test_rmse:.4f} ")
        else:
            print(f"  Test RMSE {test_rmse:.4f} isn't better than {state['best_test_rmse']:.4f}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    study = optuna.create_study(
        study_name="mggat_pa_4grphs",
        storage="sqlite:///optuna4g.db",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.PercentilePruner(
            percentile=75,
            n_startup_trials=10,
            n_warmup_steps=20
        ))
    study.optimize(objective, n_trials=n_trials, callbacks=[eval_on_new_best])

    print("\n=== Optuna Search Finished ===")
    print(f"Best Tune RMSE: {study.best_trial.value:.4f}")
    print("Best Hyperparameters:", study.best_trial.params)
    return state["best_results"]


# def train_and_eval(best_params, data_bundle, test_loader, device):
#
#     S_u_exp, S_b_exp, eu, eb, train_loader, tune_loader, num_users, num_items, u_in_dim, i_in_dim, train_df = data_bundle
#     implicit_dim = best_params.get("implicit_dim")
#     S_u_imp, S_b_imp = generate_implicit_features(train_df, num_users, num_items, latent_dim=implicit_dim)
#
#     S_u = torch.cat([S_u_exp.to(device), S_u_imp.to(device)], dim=1)
#     S_b = torch.cat([S_b_exp.to(device), S_b_imp.to(device)], dim=1)
#
#     u_in_dim = S_u.shape[1]
#     i_in_dim = S_b.shape[1]
#
#     model = MGGATRecommender(
#         num_users=num_users, num_items=num_items, u_in_dim=u_in_dim, i_in_dim=i_in_dim,
#         latent_dim=best_params["latent_dim"], final_dim=best_params["final_dim"],
#         num_u_graphs=len(eu), num_i_graphs=len(eb), actv_in_name=best_params.get("activation_in"),
#         actv_out_name=best_params.get("activation_out")
#     ).to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
#     train_criterion = nn.MSELoss(reduction='mean')
#     eval_criterion = nn.MSELoss(reduction='sum')
#
#     FINAL_EPOCHS = 100
#     EARLY_STOP_PATIENCE = 15
#     best_tune_rmse = float('inf')
#     best_state_dict = None
#     patience_counter = 0
#
#     for epoch in range(FINAL_EPOCHS):
#         model.train()
#         for batch_u, batch_i, true_ratings in train_loader:
#             batch_u, batch_i, true_ratings = batch_u.to(device), batch_i.to(device), true_ratings.to(device)
#
#             optimizer.zero_grad()
#             predictions = model(batch_u, batch_i, S_u, S_b, eu, eb)
#             mse_loss = train_criterion(predictions, true_ratings)
#
#             reg_loss_u = compute_graph_regularization(model.H_u_4.weight, eu)
#             reg_loss_b = compute_graph_regularization(model.H_b_4.weight, eb)
#             graph_reg_loss = best_params["theta_1"] * (reg_loss_u + reg_loss_b)
#
#             loss = mse_loss + graph_reg_loss
#             loss.backward()
#             optimizer.step()
#
#         tune_rmse = evaluate_model(model, tune_loader, S_u, S_b, eu, eb, eval_criterion, device)
#
#         if tune_rmse < best_tune_rmse:
#             best_tune_rmse = tune_rmse
#             best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
#             patience_counter = 0
#             print(f"Epoch [{epoch + 1}] ✓ New best: {tune_rmse:.4f}")
#         else:
#             patience_counter += 1
#             print(f"Epoch [{epoch + 1}] Tune RMSE: {tune_rmse:.4f} "
#                   f"(best: {best_tune_rmse:.4f}, patience: {patience_counter}/{EARLY_STOP_PATIENCE})")
#
#             if patience_counter >= EARLY_STOP_PATIENCE:
#                 print(f"Early stopping triggered at epoch {epoch + 1}")
#                 break
#
#     print("\n=== Evaluating on TEST Set ===")
#     model.load_state_dict(best_state_dict)
#     test_rmse = evaluate_model(model, test_loader, S_u, S_b, eu, eb, eval_criterion, device)
#     spearman, fcp, valid_ratio = evaluate_ranking_metrics(model, test_loader, S_u, S_b, eu, eb, device)
#     return model, test_rmse, spearman, fcp, valid_ratio,S_u_imp, S_b_imp



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    S_u, S_b, eu, eb, train_ds, tune_ds, test_ds, train_df = load_mggat_data()
    num_users, u_in_dim = S_u.shape
    num_items, i_in_dim = S_b.shape

    S_u, S_b = S_u.to(device), S_b.to(device)
    eu = [e.to(device) for e in eu]
    eb = [e.to(device) for e in eb]

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    tune_loader = DataLoader(tune_ds, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    data_bundle = (S_u, S_b, eu, eb, train_loader, tune_loader, num_users, num_items, u_in_dim, i_in_dim, train_df)

    # best_results = run_optuna_search(data_bundle, device, n_trials=400)
    # print("\n=== Final Results ===")
    # print(f"Test RMSE:  {best_results['test_rmse']:.4f}")
    # print(f"Spearman:   {best_results['spearman']:.4f}")
    # print(f"FCP:        {best_results['fcp']:.4f}")
    # print(f"BPR:        {best_results.get('bpr', 0.0):.4f}")
    # print(f"Trial:      {best_results['trial_number']}")
    # print(f"\n[Success] Best models saved in {BEST_MODEL_PATH}")

    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n=== Loading Best Model from {BEST_MODEL_PATH} ===")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
        best_params = checkpoint["params"]

        # 取出保存好的 SVD 隐式特征并与显式特征拼接
        S_u_imp = checkpoint["S_u_imp"].to(device)
        S_b_imp = checkpoint["S_b_imp"].to(device)
        S_u = torch.cat([S_u.to(device), S_u_imp], dim=1)
        S_b = torch.cat([S_b.to(device), S_b_imp], dim=1)

        num_users, u_in_dim = S_u.shape
        num_items, i_in_dim = S_b.shape

        model = MGGATRecommender(
            num_users=num_users, num_items=num_items, u_in_dim=u_in_dim, i_in_dim=i_in_dim,
            latent_dim=best_params["latent_dim"], final_dim=best_params["final_dim"],
            num_u_graphs=len(eu), num_i_graphs=len(eb),
            actv_in_name=best_params["activation_in"],
            actv_out_name=best_params["activation_out"]
        ).to(device)


        model.load_state_dict(checkpoint["model_state_dict"])


        print("Evaluating on test set...")
        eval_criterion = nn.MSELoss(reduction='sum')
        test_rmse = evaluate_model(model, test_loader, S_u, S_b, eu, eb, eval_criterion, device)
        spearman, fcp, bpr, valid_ratio = evaluate_ranking_metrics(model, test_loader, S_u, S_b, eu, eb, device)

        print("\n=== Final Test Results (Loaded Model) ===")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Spearman:   {spearman:.4f}")
        print(f"FCP:        {fcp:.4f}")
        print(f"BPR:        {bpr:.4f}")
        print("=========================================")
    else:
        print(f"Error: {BEST_MODEL_PATH} not found. Please run Optuna search first.")

