# TVRec: Flow-Guided Triangle Velocities Synergy for Multimodal Recommendation

This is the PyTorch implementation of **TVRec**, a multimodal recommendation framework built upon [**DiffMM**](https://github.com/HKUDS/DiffMM) (ACM MM 2024) and inspired by **Triangle Velocities Synergy (TVS)** from [**Optical Flow Matching (OFM)**](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Optical_Flow_Matching_Reframing_Optical_Flow_as_Continuous_Transport_Dynamics_CVPR_2026_paper.html) (CVPR 2026).

TVRec adapts the idea of TVS to user–item interaction reconstruction. Instead of directly predicting clean interaction vectors, the denoising module learns velocities along a main trajectory, an anchor-directed auxiliary trajectory, and a stationary auxiliary trajectory. The predicted velocity is converted into interaction scores to rebuild modality-specific user–item graphs. The framework retains DiffMM's multimodal graph aggregation and cross-modal contrastive learning. TVS is attributed to OFM; TVRec denotes the recommendation-specific adaptation implemented in this repository.

<img src="./figures/model.png" alt="DiffMM backbone architecture" width="100%" />

## Environment

The implementation uses the following dependencies:

- Python
- PyTorch with CUDA support
- NumPy
- SciPy
- tqdm
- setproctitle

## Experimental Results

Performance comparison on TikTok, Amazon-Baby, and Amazon-Sports in terms of **Recall@20**, **Precision@20**, and **NDCG@20**:

<img src="./figures/performance.png" alt="Performance comparison of TVRec, DiffMM, and other baselines on three datasets" width="100%" />

*Results as reported in the supplied comparison figure. The TVRec column contains the results of this adaptation; the figure alone does not establish reproducibility or statistical significance.*

The current training loop selects the best epoch using test Recall. Validation-based model selection must be implemented before using this pipeline for held-out paper evaluation.

## How to Run the Codes

The example commands below train TVRec on the three supported datasets. Unspecified hyperparameters use the defaults in [Params.py](Params.py). These examples are not tuned configurations for reproducing paper results.

Prepare the datasets as described in the **Datasets** section, then run the commands from the repository root.

- **TikTok**

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --anchor_w 2.0
```

- **Baby**

```bash
python Main.py --data baby --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --anchor_w 2.0
```

- **Sports**

```bash
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --anchor_w 2.0
```

These examples retain the dataset-specific settings from the DiffMM README and add an illustrative anchor strength. `--anchor_w` controls the main trajectory's anchor strength; its default is `0.0`, which sets the anchor to zero for both the main and auxiliary trajectories. `--lambda_x`, `--lambda_y`, and `--lambda_z` control the three velocity loss weights and default to `1.0` each.

TVS is always enabled; the previous `--velocity_mode` argument has been removed. Always specify `--data`, since the legacy default `allrecipes` is not supported by the data loader.

To run the numerical and gradient checks on CPU without a dataset:

```bash
python verify_cpu_tvs.py
```

These checks use synthetic data and do not replace full training or recommendation benchmarks.

## Code Structure

```text
.
├── README.md
├── Main.py
├── Model.py
├── Params.py
├── DataHandler.py
├── verify_cpu_tvs.py
├── Utils
│   ├── TimeLogger.py
│   └── Utils.py
├── figures
│   ├── model.png
│   ├── dataset.png
│   └── performance.png
└── Datasets
    ├── README.md
    ├── baby
    │   ├── image_feat.npy.zip
    │   ├── text_feat.npy
    │   ├── trnMat.pkl
    │   ├── tstMat.pkl
    │   └── valMat.pkl
    └── tiktok
        ├── audio_feat.npy
        ├── image_feat.npy
        ├── text_feat.npy
        ├── trnMat.pkl
        ├── tstMat.pkl
        └── valMat.pkl
```

## Datasets

<img src="./figures/dataset.png" alt="Dataset statistics and visual, acoustic, and textual feature dimensions reported by DiffMM" width="100%" />

## Citation
