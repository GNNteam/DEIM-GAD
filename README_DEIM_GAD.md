# DEIM-GAD: Diffusion-Enhanced Multi-Perspective Inconsistency Modeling for Graph Anomaly Detection

This repository provides an implementation of a graph anomaly detection framework based on dual-view reconstruction. The model constructs an original feature view and a heat-kernel-diffused feature view, and jointly models structural, attribute, and degree-level reconstruction errors for node anomaly scoring.

## 1. Requirements

python==3.8.20
torch==2.0.0
dgl==2.1.0+cu118
torchdata==0.6.0
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu118
torch-sparse==0.6.18+pt20cu118
numpy==1.23.5
scipy==1.10.1
scikit-learn==1.0.2
networkx==2.8.8
tqdm==4.67.1


## 2. Hyperparameter Settings

The following hyperparameters are used for the main experiments.

| Dataset  | T | λ | α | w_s | w_a | Dim |
|---|---:|---:|---:|---:|---:|---:|
| Citeseer | 4 | 0.8 | 0.3 | 0.0 | 0.4 | 32 |
| Cora     | 4 | 0.7 | 0.3 | 0.2 | 0.2 | 128 |
| Pubmed   | 8 | 0.3 | 0.4 | 0.0 | 0.2 | 64 |
| Books    | 4 | 0.7 | 0.3 | 0.6 | 0.2 | 128 |
| Enron    | 6 | 0.4 | 0.5 | 0.6 | 0.0 | 64 |
| Reddit   | 4 | 0.7 | 0.4 | 0.6 | 0.4 | 128 |






