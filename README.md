# MG-GAT: PyTorch Reproduction & Interpretability Analysis

PyTorch reimplementation of **MG-GAT** (Multi-Graph Graph Attention Network) for explainable recommendation, evaluated on the Yelp Pennsylvania (PA) dataset.

---

## Project Structure

```
├── train.py                  # Optuna hyperparameter search + model training
├── models.py                 # MGGATRecommender (PyTorch)
├── loaddata.py               # Data loading + 4-graph business side construction
├── analyze.py                # Model loading, FR/NIG analysis, ablation utilities
├── mggat_analysis.ipynb      # Full interpretability analysis + visualizations
├── best_model.pth            # Best model checkpoint (trial 178)
├── best_results.json         # Best evaluation metrics
├── best_hyperparameters.json # Best hyperparameter configuration
├── optuna4g.db               # Optuna SQLite study (study_name: mggat_pa_4graphs)
├── environment.yml           # Conda environment
└── dataset/PA/               # Yelp PA dataset
    ├── data.csv
    ├── user_features.csv
    ├── item_features.csv
    ├── user_graph.npz
    ├── item_graph.npz          # Category KNN graph
    ├── item_graph_geo.npz      # Geographic KNN graph
    ├── item_graph_covisit.npz  # Co-visitation graph
    └── item_graph_segment.npz  # LLM segment graph
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate mggat_pytorch
```

Requires CUDA-compatible GPU. Tested with PyTorch 2.10.0 + CUDA 12.8.

---

## Training

```bash
python train.py
```

Runs Optuna TPE search (SQLite-persisted). Results are saved to `best_model.pth`, `best_results.json`, and `best_hyperparameters.json` whenever a new best tune RMSE is found. To resume a previous study, the existing `optuna4g.db` is loaded automatically.

---

## Analysis

All interpretability analysis is in `mggat_analysis.ipynb`, including:

- Multi-graph omega weight analysis
- NIG attention sparsity (Gini coefficient, Top-1 share)
- Feature Relevance (FR) rankings for user and business sides
- Transparency validation (NIG vs FR-weighted feature similarity)
- Component ablation (SVD vs GAT contribution)
- NIG vs uniform attention ablation

---

## Best Results

| Metric | Reproduced | Paper (MG-GAT) |
|---|---|---|
| Test RMSE | 1.3241 | 1.249 |
| Tune RMSE | 1.2995 | — |
| Spearman | 0.3125 | 0.405 |
| FCP | 0.6615 | 0.602 |
| BPR | 0.5403 | 0.520 |

Best hyperparameters: `latent_dim=61`, `final_dim=81`, `lr=4.8e-4`, `activation_in=relu`, `activation_out=tanh`, `implicit_dim=4` (trial 178, 300 total trials).
