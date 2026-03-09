"""
analyze.py
==========
MG-GAT Interpretability Analysis for Report Findings.
Three core analyses:
  1. NIG Sparsity  — does attention truly filter noisy neighbors?
  2. Feature Relevance (FR) — which auxiliary features drive neighbor importance?
  3. Transparency Validation — are NIG weights consistent with feature similarity?

Assumes a trained MGGATRecommender model is available.
Run after training: python analyze.py
"""

import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from train import evaluate_model

matplotlib.use('Agg')  # Non-interactive backend for server environments
from scipy.stats import spearmanr
from torch_geometric.utils import softmax
from torch.utils.data import DataLoader

from loaddata import load_mggat_data
from models import MGGATRecommender

# ─────────────────────────────────────────────
# 0.  HELPERS
# ─────────────────────────────────────────────

def load_trained_model(model_path, hyperparams_path, device):
    with open(hyperparams_path, "r") as f:
        params = json.load(f)

    S_u_exp, S_b_exp, eu, eb, train_ds, tune_ds, test_ds, train_df = load_mggat_data()

    checkpoint = torch.load(model_path, map_location=device)
    S_u_imp = checkpoint["S_u_imp"].to(device)
    S_b_imp = checkpoint["S_b_imp"].to(device)

    S_u = torch.cat([S_u_exp.to(device), S_u_imp], dim=1)
    S_b = torch.cat([S_b_exp.to(device), S_b_imp], dim=1)

    eu = [e.to(device) for e in eu]
    eb = [e.to(device) for e in eb]

    num_users = S_u.shape[0]
    num_items = S_b.shape[0]

    num_i_graphs = checkpoint.get("num_i_graphs")
    num_u_graphs = checkpoint.get("num_u_graphs")
    print("num_i_graphs", num_i_graphs)
    print("num_u_graphs", num_u_graphs)
    model = MGGATRecommender(
        num_users=num_users, num_items=num_items,
        u_in_dim=S_u.shape[1], i_in_dim=S_b.shape[1],
        latent_dim=params["latent_dim"], final_dim=params["final_dim"],
        num_u_graphs=num_u_graphs, num_i_graphs=num_i_graphs,
        actv_in_name=params.get("activation_in"),
        actv_out_name=params.get("activation_out")
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Params: {params}")

    return model, S_u, S_b, eu, eb, train_ds, tune_ds, test_ds, params


def compute_gini(weights: np.ndarray) -> float:
    """
    Gini coefficient of an attention weight distribution.
    0 = perfectly uniform (no filtering),  1 = all weight on one neighbor.
    Formula: G = 1 - sum_i (2*(n-i+1)/(n*(n+1))) * w_i  for sorted ascending w.
    """
    w = np.sort(np.abs(weights))
    n = len(w)
    if n < 2 or w.sum() == 0:
        return 0.0
    cumsum = np.cumsum(w)
    return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n


# ─────────────────────────────────────────────
# 1.  NIG SPARSITY ANALYSIS
# ─────────────────────────────────────────────

def analyze_nig_sparsity(model, S_u, S_b, eu, eb, device,
                          num_samples=2000, output_dir="figures"):
    """
    Analysis 1: Does NIG truly filter noisy neighbors?

    Method:
      - Recompute attention weights α for a sample of nodes
      - Measure Gini coefficient per node (0=uniform, 1=concentrated)
      - Compare top-1 and top-3 weight share vs. uniform baseline
      - Plot distributions

    Maps to model params:
      user_gat.a_self, user_gat.a_nb, user_gat.W_1  (Eq. 3)
      item_gat.a_self, item_gat.a_nb, item_gat.W_1
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    results = {}

    for side, gat_layer, edge_index, S, label in [
        ('User', model.user_gat, eu[0], S_u, 'User NIG'),
        ('Business', model.item_gat, eb[0], S_b, 'Business NIG'),
    ]:
        with torch.no_grad():
            # H^(1) = W^(1) * S   (Eq. 2)
            H1 = gat_layer.W_1(S)                          # [N, out_dim]

            # Pre-compute dot products with attention vectors
            # alpha_score_i = a_self · H1_i  (focal node contribution)
            # alpha_score_j = a_nb  · H1_j   (neighbor contribution)
            score_self = (H1 * gat_layer.a_self).sum(dim=-1)   # [N]
            score_nb   = (H1 * gat_layer.a_nb  ).sum(dim=-1)   # [N]

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()

        score_self_np = score_self.cpu().numpy()
        score_nb_np   = score_nb.cpu().numpy()

        # Sample unique destination nodes
        unique_dst = np.unique(dst)
        if len(unique_dst) > num_samples:
            unique_dst = np.random.choice(unique_dst, num_samples, replace=False)

        gini_scores, top1_shares, top3_shares, n_neighbors_list = [], [], [], []

        for node_i in unique_dst:
            mask = (dst == node_i)
            neighbors = src[mask]
            k = len(neighbors)
            if k < 2:
                continue

            # Attention scores for all neighbors of node_i  (Eq. 3 numerator)
            raw = score_self_np[node_i] + score_nb_np[neighbors]   # [k]
            raw = np.where(raw > 0, raw, 0.2 * raw)                # LeakyReLU(0.2)

            # Softmax over neighbors
            raw_exp = np.exp(raw - raw.max())
            alpha = raw_exp / raw_exp.sum()                         # [k]

            gini_scores.append(compute_gini(alpha))
            top1_shares.append(np.sort(alpha)[::-1][0])
            top3_shares.append(np.sort(alpha)[::-1][:min(3, k)].sum())
            n_neighbors_list.append(k)

        gini_arr = np.array(gini_scores)
        top1_arr = np.array(top1_shares)

        # Uniform baseline: if k neighbors, each gets 1/k
        uniform_top1 = np.mean(1.0 / np.array(n_neighbors_list))

        print(f"\n{'─'*50}")
        print(f"[NIG Sparsity] {side} side  (n_nodes = {len(gini_arr)})")
        print(f"  Gini coefficient:  mean={gini_arr.mean():.4f}  median={np.median(gini_arr):.4f}")
        print(f"  Top-1 weight share: {top1_arr.mean():.4f}  vs. uniform={uniform_top1:.4f}")
        print(f"  Top-3 weight share: {np.mean(top3_shares):.4f}")
        print(f"  NIG selectivity ratio (top-1 / uniform): {top1_arr.mean()/uniform_top1:.2f}x")

        results[side] = {
            'gini': gini_arr,
            'top1': top1_arr,
            'top3': np.array(top3_shares),
            'uniform_top1': uniform_top1,
        }

        # ── Plot ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(gini_arr, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        axes[0].axvline(gini_arr.mean(), color='red', linestyle='--',
                        label=f'Mean={gini_arr.mean():.3f}')
        axes[0].set_xlabel('Gini Coefficient', fontsize=12)
        axes[0].set_ylabel('Node Count', fontsize=12)
        axes[0].set_title(f'{label}: Attention Weight Sparsity\n'
                          f'(Higher = More Selective Filtering)', fontsize=12)
        axes[0].legend()

        axes[1].hist(top1_arr, bins=60, color='darkorange', edgecolor='white', alpha=0.85)
        axes[1].axvline(uniform_top1, color='navy', linestyle='--',
                        label=f'Uniform baseline={uniform_top1:.3f}')
        axes[1].axvline(top1_arr.mean(), color='red', linestyle='--',
                        label=f'NIG mean={top1_arr.mean():.3f}')
        axes[1].set_xlabel('Top-1 Neighbor Weight Share', fontsize=12)
        axes[1].set_ylabel('Node Count', fontsize=12)
        axes[1].set_title(f'{label}: Top-1 Neighbor Dominance', fontsize=12)
        axes[1].legend()

        plt.tight_layout()
        path = os.path.join(output_dir, f'nig_sparsity_{side.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved → {path}")

    return results


# ─────────────────────────────────────────────
# 2.  FEATURE RELEVANCE (FR) EXTRACTION
# ─────────────────────────────────────────────

def extract_and_plot_fr(model, feature_names_user, feature_names_biz,
                         top_k=20, output_dir="figures"):
    """
    Analysis 2: Which auxiliary features drive neighbor importance?

    FR definition (paper Def. 2):
      FR_focal_user     = a_{u,self}^T @ W^(1)_u   → shape [s_u]
      FR_neighbor_user  = a_{u,nb}^T  @ W^(1)_u    → shape [s_u]
      FR_focal_biz      = a_{b,self}^T @ W^(1)_b   → shape [s_b]
      FR_neighbor_biz   = (Σ_g ω_g) * a_{b,nb}^T @ W^(1)_b

    Maps to model params:
      user_gat.a_self, user_gat.a_nb, user_gat.W_1.weight
      item_gat.a_self, item_gat.a_nb, item_gat.W_1.weight, item_gat.omega
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_fr = {}

    with torch.no_grad():

        # ── User FR ──
        W1_u = model.user_gat.W_1.weight.cpu().numpy()     # [out_dim, s_u]
        au_self = model.user_gat.a_self.cpu().numpy()       # [1, out_dim]
        au_nb   = model.user_gat.a_nb.cpu().numpy()         # [1, out_dim]

        FR_focal_u    = (au_self @ W1_u).squeeze()          # [s_u]
        FR_neighbor_u = (au_nb   @ W1_u).squeeze()          # [s_u]
        FR_user_signed   = FR_focal_u + FR_neighbor_u       # combined, signed
        FR_user_combined = np.abs(FR_focal_u) + np.abs(FR_neighbor_u)  # for ranking

        # ── Business FR ──
        W1_b  = model.item_gat.W_1.weight.cpu().numpy()    # [out_dim, s_b]
        ab_self = model.item_gat.a_self.cpu().numpy()       # [1, out_dim]
        ab_nb   = model.item_gat.a_nb.cpu().numpy()         # [1, out_dim]
        omega   = model.item_gat.omega.cpu().numpy()        # [num_graphs]

        FR_focal_b    = (ab_self @ W1_b).squeeze()                    # [s_b]
        FR_neighbor_b = omega.sum() * (ab_nb @ W1_b).squeeze()        # [s_b]
        FR_biz_signed   = FR_focal_b + FR_neighbor_b
        FR_biz_combined = np.abs(FR_focal_b) + np.abs(FR_neighbor_b)

        all_fr['user_signed']    = FR_user_signed
        all_fr['user_combined']  = FR_user_combined
        all_fr['biz_signed']     = FR_biz_signed
        all_fr['biz_combined']   = FR_biz_combined

    # ── Print top features ──
    for side, fr_vals, feat_names in [
        ('User',     FR_user_combined,  feature_names_user),
        ('Business', FR_biz_combined,   feature_names_biz),
    ]:
        idx = np.argsort(fr_vals)[::-1][:top_k]
        print(f"\n{'─'*50}")
        print(f"[FR] Top-{top_k} {side} Features driving NIG:")
        for rank, i in enumerate(idx):
            direction = '(+)' if (FR_user_signed if side=='User' else FR_biz_signed)[i] > 0 else '(-)'
            print(f"  #{rank+1:2d} {direction}  {feat_names[i]:<45s}  FR={fr_vals[i]:.4f}")

    # ── Plot: combined magnitude bar charts ──
    for side, fr_vals, fr_signed, feat_names in [
        ('User',     FR_user_combined, FR_user_signed,  feature_names_user),
        ('Business', FR_biz_combined,  FR_biz_signed,   feature_names_biz),
    ]:
        idx = np.argsort(fr_vals)[::-1][:top_k]
        vals   = fr_vals[idx]
        signed = fr_signed[idx]
        names  = [feat_names[i] for i in idx]
        colors = ['steelblue' if s > 0 else 'tomato' for s in signed]

        # Normalize to [0,1]
        vals_norm = vals / vals.max()

        fig, ax = plt.subplots(figsize=(8, top_k * 0.4 + 1))
        bars = ax.barh(range(top_k), vals_norm[::-1], color=colors[::-1], edgecolor='white')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel('Normalized Feature Relevance', fontsize=11)
        ax.set_title(f'Feature Relevance (FR) — {side} Side\n'
                     f'Blue=positive influence, Red=negative influence', fontsize=11)

        # Add legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='steelblue', label='Positive FR'),
                            Patch(color='tomato',    label='Negative FR')],
                  loc='lower right', fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, f'fr_{side.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  FR figure saved → {path}")

    return all_fr


def fr_stability_analysis(model_path, hyperparams_path, S_u, S_b, eu, eb,
                           feature_names_biz, device, n_seeds=5, top_k=20):
    """
    Runs training from multiple random seeds and checks FR ranking stability.
    High stability → FR is a reliable explanation; Low stability → caveat needed.
    (Requires re-training, so only run if compute allows.)
    """
    print("\n[FR Stability] Checking ranking stability across seeds...")
    all_rankings = []

    with open(hyperparams_path, 'r') as f:
        params = json.load(f)

    S_u_orig, S_b_orig, eu_orig, eb_orig, train_ds, tune_ds, test_ds = load_mggat_data()
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    for seed in [42, 123, 456, 789, 1024][:n_seeds]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        num_users, u_in_dim = S_u_orig.shape
        num_items, i_in_dim = S_b_orig.shape

        m = MGGATRecommender(
            num_users=num_users, num_items=num_items,
            u_in_dim=u_in_dim, i_in_dim=i_in_dim,
            latent_dim=params['latent_dim'], final_dim=params['final_dim'],
            num_u_graphs=len(eu_orig), num_i_graphs=len(eb_orig)
        ).to(device)

        opt = torch.optim.Adam(m.parameters(), lr=params['lr'],
                                weight_decay=params['weight_decay'])
        crit = nn.MSELoss()

        Su_dev  = S_u_orig.to(device)
        Sb_dev  = S_b_orig.to(device)
        eu_dev  = [e.to(device) for e in eu_orig]
        eb_dev  = [e.to(device) for e in eb_orig]

        for _ in range(10):  # Quick 10-epoch training
            m.train()
            for bu, bi, br in train_loader:
                bu, bi, br = bu.to(device), bi.to(device), br.to(device)
                opt.zero_grad()
                loss = crit(m(bu, bi, Su_dev, Sb_dev, eu_dev, eb_dev), br)
                loss.backward()
                opt.step()

        m.eval()
        with torch.no_grad():
            W1_b  = m.item_gat.W_1.weight.cpu().numpy()
            ab_self = m.item_gat.a_self.cpu().numpy()
            ab_nb   = m.item_gat.a_nb.cpu().numpy()
            fr_b = np.abs((ab_self @ W1_b).squeeze()) + np.abs((ab_nb @ W1_b).squeeze())
            ranking = np.argsort(fr_b)[::-1][:top_k]
            all_rankings.append(ranking)
        print(f"  Seed {seed} done.")

    # Spearman correlation between all pairs of rankings
    from itertools import combinations as combs
    corrs = []
    for (r1, r2) in combs(all_rankings, 2):
        c, _ = spearmanr(r1, r2)
        corrs.append(c)

    print(f"\n  FR Ranking Stability (Spearman): "
          f"mean={np.mean(corrs):.4f}  std={np.std(corrs):.4f}")
    print(f"  Interpretation: >0.8 = stable explanations, "
          f"<0.5 = unstable (caveat in report)")
    return corrs


# ─────────────────────────────────────────────
# 3.  TRANSPARENCY VALIDATION
#     Are NIG weights consistent with feature similarity?
# ─────────────────────────────────────────────

def validate_nig_transparency(model, S_b, eb, device, sample_nodes=1000):
    """
    Analysis 3: IS Transparency Claim Validation.

    Mathematical insight:
      Attention score for neighbor l → focal node j:
        raw[l] = a_self·H1[j] + a_nb·H1[l]
      In the stable softmax (raw - max), the constant a_self·H1[j] cancels out.
      Therefore: alpha[l] is determined SOLELY by a_nb·H1[l] — an absolute
      property of the neighbor, independent of the focal node j.

    Correct transparency test:
      NIG claims to surface neighbors that matter because of their auxiliary
      features (via FR). So the right question is:
        "Does alpha[l] ranking align with FR-weighted feature similarity
         between j and l?"
      i.e., do neighbors that are most similar to j on the FR-important features
      receive the highest attention weight?

    Method:
      1. Compute FR weights (|a_nb·W1| per feature dim)
      2. For each (j, neighbors): compute FR-weighted cosine similarity
      3. Spearman r between alpha ranking and FR-sim ranking
    """
    model.eval()
    correlations = []
    p_values = []

    edge_index = eb[0]
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    with torch.no_grad():
        ab_self = model.item_gat.a_self   # [1, out_dim]
        ab_nb   = model.item_gat.a_nb     # [1, out_dim]
        W1_b    = model.item_gat.W_1.weight  # [out_dim, s_b]

        H1_b = model.item_gat.W_1(S_b)      # [N, out_dim]
        score_self = (H1_b * ab_self).sum(-1).cpu().numpy()  # [N]
        score_nb   = (H1_b * ab_nb  ).sum(-1).cpu().numpy()  # [N]

        # FR weights: importance of each raw feature dim for neighbor attention
        # FR[d] = |a_nb · W1[:,d]|  — how much raw feature d contributes to alpha
        FR_weights = np.abs((ab_nb.cpu().numpy() @ W1_b.cpu().numpy()).squeeze())  # [s_b]
        # Normalize to sum=1 for use as similarity weights
        FR_weights = FR_weights / (FR_weights.sum() + 1e-8)

    S_b_np = S_b.cpu().numpy()  # [N, s_b] raw features

    unique_dst = np.unique(dst)
    if len(unique_dst) > sample_nodes:
        unique_dst = np.random.choice(unique_dst, sample_nodes, replace=False)

    for node_j in unique_dst:
        mask = (dst == node_j)
        neighbors = src[mask]
        k = len(neighbors)
        if k < 3:
            continue

        # NIG alpha: driven by a_nb·H1[l], focal term cancels in softmax
        raw = score_self[node_j] + score_nb[neighbors]
        raw = np.where(raw > 0, raw, 0.2 * raw)  # LeakyReLU
        raw_exp = np.exp(raw - raw.max())
        alpha = raw_exp / raw_exp.sum()           # [k]

        # FR-weighted feature similarity between focal j and each neighbor l
        # weight each feature dim by its FR importance before computing cosine sim
        feat_j = S_b_np[node_j] * FR_weights          # [s_b] weighted
        feat_j_norm = np.linalg.norm(feat_j) + 1e-8
        fr_sims = []
        for nb in neighbors:
            feat_nb = S_b_np[nb] * FR_weights          # [s_b] weighted
            sim = np.dot(feat_j, feat_nb) / (feat_j_norm * np.linalg.norm(feat_nb) + 1e-8)
            fr_sims.append(sim)
        fr_sims = np.array(fr_sims)

        if np.std(alpha) < 1e-8 or np.std(fr_sims) < 1e-8:
            continue

        corr, pval = spearmanr(alpha, fr_sims)
        if not np.isnan(corr):
            correlations.append(corr)
            p_values.append(pval)

    corr_arr = np.array(correlations)
    pval_arr = np.array(p_values)

    print(f"\n{'─'*50}")
    print(f"[Transparency Validation] NIG alpha vs. FR-weighted Feature Similarity")
    print(f"  Nodes analyzed: {len(corr_arr)}")
    print(f"  Mean Spearman r = {corr_arr.mean():.4f}  (std={corr_arr.std():.4f})")
    print(f"  Median Spearman r = {np.median(corr_arr):.4f}")
    print(f"  % nodes with positive r = {(corr_arr > 0).mean()*100:.1f}%")
    print(f"  % nodes with r > 0.3    = {(corr_arr > 0.3).mean()*100:.1f}%")
    print(f"  % significant (p<0.05)  = {(pval_arr < 0.05).mean()*100:.1f}%")

    if corr_arr.mean() > 0.2:
        interp = "NIG preferentially attends to FR-similar neighbors → supports IS transparency claim"
    elif corr_arr.mean() > 0:
        interp = "Weak positive alignment: NIG partially consistent with FR-driven similarity"
    else:
        interp = "NIG attention not aligned with FR-feature similarity — transparency claim limited"
    print(f"  Interpretation: {interp}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corr_arr, bins=60, color='seagreen', edgecolor='white', alpha=0.85)
    ax.axvline(corr_arr.mean(), color='red', linestyle='--',
               label=f'Mean r = {corr_arr.mean():.3f}')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Spearman r  (NIG alpha vs. FR-weighted feature similarity)', fontsize=10)
    ax.set_ylabel('Node Count', fontsize=11)
    ax.set_title('IS Transparency Validation:\nDo NIG Weights Align with FR-important Features?', fontsize=11)
    ax.legend()
    plt.tight_layout()
    path = 'figures/transparency_validation.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved → {path}")

    return corr_arr


# ─────────────────────────────────────────────
# 4.  CASE STUDY: Single recommendation explained
# ─────────────────────────────────────────────

def explain_one_recommendation(model, user_id, item_id,
                                 S_u, S_b, eu, eb,
                                 feature_names_user, feature_names_biz,
                                 device, top_k_fr=5, top_k_nb=3):
    """
    Generates a structured NIG + FR explanation for one (user, item) pair.
    This is the qualitative case study for the report.
    Directly mirrors the Appendix G examples in the paper.
    """
    model.eval()

    with torch.no_grad():
        # ── Predicted rating ──
        pred = model(
            torch.tensor([user_id]).to(device),
            torch.tensor([item_id]).to(device),
            S_u, S_b, eu, eb
        ).item()

        # ── Business NIG: find top neighbors of this item ──
        H1_b = model.item_gat.W_1(S_b)
        score_self_b = (H1_b * model.item_gat.a_self).sum(-1)
        score_nb_b   = (H1_b * model.item_gat.a_nb  ).sum(-1)

        edge_b = eb[0]
        src_b = edge_b[0].cpu().numpy()
        dst_b = edge_b[1].cpu().numpy()

        mask = (dst_b == item_id)
        neighbors_b = src_b[mask]

        if len(neighbors_b) > 0:
            raw_b = score_self_b[item_id].item() + score_nb_b[neighbors_b].cpu().numpy()
            raw_b = np.where(raw_b > 0, raw_b, 0.2 * raw_b)
            exp_b = np.exp(raw_b - raw_b.max())
            alpha_b = exp_b / exp_b.sum()
            top_nb_idx = np.argsort(alpha_b)[::-1][:top_k_nb]
            top_nb_ids  = neighbors_b[top_nb_idx]
            top_nb_weights = alpha_b[top_nb_idx]
        else:
            top_nb_ids, top_nb_weights = [], []

        # ── FR: top features driving this recommendation ──
        W1_b    = model.item_gat.W_1.weight.cpu().numpy()
        ab_self = model.item_gat.a_self.cpu().numpy()
        ab_nb   = model.item_gat.a_nb.cpu().numpy()
        omega   = model.item_gat.omega.cpu().numpy()

        FR_focal_b    = (ab_self @ W1_b).squeeze()
        FR_neighbor_b = omega.sum() * (ab_nb @ W1_b).squeeze()
        FR_combined   = np.abs(FR_focal_b) + np.abs(FR_neighbor_b)
        top_fr_idx    = np.argsort(FR_combined)[::-1][:top_k_fr]

    # ── Print structured explanation ──
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION EXPLANATION")
    print(f"  User ID:        {user_id}")
    print(f"  Business ID:    {item_id}")
    print(f"  Predicted Rating: {pred:.2f} / 5.0")

    print(f"\n[NIG] Top-{top_k_nb} Most Similar Businesses (Neighbor Importance):")
    for rank, (nb_id, w) in enumerate(zip(top_nb_ids, top_nb_weights)):
        print(f"  #{rank+1}  Business {nb_id}  |  α = {w:.4f}")

    print(f"\n[FR] Top-{top_k_fr} Features Driving Neighbor Selection:")
    for rank, i in enumerate(top_fr_idx):
        direction = 'positive' if (FR_focal_b + FR_neighbor_b)[i] > 0 else 'negative'
        print(f"  #{rank+1}  {feature_names_biz[i]:<40s}  FR={FR_combined[i]:.4f}  ({direction})")



# ─────────────────────────────────────────────
# 5.  ABLATION: NIG-OFF vs NIG-ON
# ─────────────────────────────────────────────

def ablation_nig_vs_uniform(model, test_loader, S_u, S_b, eu, eb, device):
    """
    Temporarily zeroes out a_self and a_nb to make attention uniform,
    then measures RMSE degradation. Quantifies NIG's contribution to accuracy.

    This replicates the 'NIG Removed' row of Table 2 in the paper using
    your trained model (instead of retraining from scratch).
    Note: results will differ from full ablation retraining — discuss in report.
    """
    from math import sqrt

    def rmse_with_attention(model, zero_attention):
        # Temporarily zero attention vectors → uniform softmax
        orig_self = model.user_gat.a_self.data.clone()
        orig_nb   = model.user_gat.a_nb.data.clone()
        orig_self_b = model.item_gat.a_self.data.clone()
        orig_nb_b   = model.item_gat.a_nb.data.clone()

        if zero_attention:
            model.user_gat.a_self.data.zero_()
            model.user_gat.a_nb.data.zero_()
            model.item_gat.a_self.data.zero_()
            model.item_gat.a_nb.data.zero_()

        model.eval()
        total_sq, total_n = 0.0, 0
        with torch.no_grad():
            for bu, bi, br in test_loader:
                bu, bi, br = bu.to(device), bi.to(device), br.to(device)
                preds = model(bu, bi, S_u, S_b, eu, eb)
                total_sq += ((preds - br) ** 2).sum().item()
                total_n  += len(br)

        # Restore
        model.user_gat.a_self.data = orig_self
        model.user_gat.a_nb.data   = orig_nb
        model.item_gat.a_self.data = orig_self_b
        model.item_gat.a_nb.data   = orig_nb_b

        return sqrt(total_sq / total_n)

    rmse_nig     = rmse_with_attention(model, zero_attention=False)
    rmse_uniform = rmse_with_attention(model, zero_attention=True)
    improvement  = (rmse_uniform - rmse_nig) / rmse_uniform * 100

    print(f"\n{'─'*50}")
    print(f"[Ablation: NIG vs Uniform Attention]")
    print(f"  RMSE with NIG (trained):   {rmse_nig:.4f}")
    print(f"  RMSE with Uniform (zeroed): {rmse_uniform:.4f}")
    print(f"  NIG improvement: {improvement:+.2f}%")
    print(f"  Note: This uses weight-zeroing, not full retraining.")
    print(f"        True ablation RMSE from paper (PA): 1.303 (NIG removed)")

    return rmse_nig, rmse_uniform


def ablation_component_contribution(model, test_loader, S_u, S_b, eu, eb, device):
    from math import sqrt

    def rmse_with_zeroing(zero_gat_u=False, zero_gat_b=False,
                          zero_svd_u=False, zero_svd_b=False):
        orig_hu4 = model.H_u_4.weight.data.clone()
        orig_hb4 = model.H_b_4.weight.data.clone()

        if zero_svd_u: model.H_u_4.weight.data.zero_()
        if zero_svd_b: model.H_b_4.weight.data.zero_()

        hooks = []
        if zero_gat_u:
            hooks.append(model.user_gat.register_forward_hook(
                lambda m, inp, out: torch.zeros_like(out)
            ))
        if zero_gat_b:
            hooks.append(model.item_gat.register_forward_hook(
                lambda m, inp, out: torch.zeros_like(out)
            ))

        eval_criterion = nn.MSELoss(reduction='sum')
        rmse = evaluate_model(model, test_loader, S_u, S_b, eu, eb, eval_criterion, device)

        for h in hooks: h.remove()
        model.H_u_4.weight.data = orig_hu4
        model.H_b_4.weight.data = orig_hb4

        return rmse

    results = {
        "Full model": rmse_with_zeroing(),
        "Zero user SVD (H_u_4)": rmse_with_zeroing(zero_svd_u=True),
        "Zero item SVD (H_b_4)": rmse_with_zeroing(zero_svd_b=True),
        "Zero user GAT": rmse_with_zeroing(zero_gat_u=True),
        "Zero item GAT": rmse_with_zeroing(zero_gat_b=True),
        "Zero all SVD": rmse_with_zeroing(zero_svd_u=True, zero_svd_b=True),
        "Zero all GAT": rmse_with_zeroing(zero_gat_u=True, zero_gat_b=True),
    }

    print(f"\n{'─' * 55}")
    print(f"[Ablation: Component Contribution]")
    baseline = results["Full model"]
    for name, rmse in results.items():
        delta = rmse - baseline
        print(f"  {name:<30s}  RMSE={rmse:.4f}  Δ={delta:+.4f}")

    return results


