# PRPH-MIL

Official implementation of **PRPH-MIL**, a prototype-guided multiple-instance
learning framework for weakly supervised whole-slide image (WSI) analysis.

PRPH-MIL combines learnable morphological-prototype-guided feature
re-embedding with prototype-guided hard-instance mining. The framework is
designed to retain diagnostically salient and complementary tissue patterns
while reducing redundant instances before slide-level aggregation.

## Supported tasks

- Cancer diagnosis: CAMELYON16 and CAMELYON17
- Cancer subtyping: TCGA-NSCLC
- Survival analysis: TCGA-BRCA and TCGA-BLCA
- MIL aggregators: ABMIL, SAPMIL, TransMIL, and DSMIL

## Repository structure

```text
PRPH-MIL/
├── classification_subtyping/  # Classification and subtyping training code
│   └── independent_test.py     # C16-to-C17 independent evaluation
├── survival_analysis/         # Discrete-time survival training code
├── labels/                    # Dataset metadata used by the experiments
├── splits/5foldcv/            # Five-fold cross-validation splits
├── scripts/                   # Example commands
├── .gitignore
├── requirements.txt
└── README.md
```

Both task directories provide four explicitly named model files:

```text
models/
├── network_ABMIL.py
├── network_SAPMIL.py
├── network_TransMIL.py
└── network_DSMIL.py
```

The `--aggregator` argument selects the corresponding implementation without
requiring any source-code changes.

## Environment

We recommend using a clean Python environment and installing a PyTorch build
that matches the local CUDA toolkit.

```bash
conda create -n prph python=3.8.18 -y
conda activate prph
pip install -r requirements.txt
torch==1.11.0
torchvision==0.12.0
```

## Data preparation

This repository expects pre-extracted patch-level WSI features. Each slide or
case is represented by a PyTorch tensor:

```text
FEATURE_DIRECTORY/
├── case_or_slide_id_1.pt
├── case_or_slide_id_2.pt
└── ...
```

Each tensor should have shape `[N, D]`, where `N` is the number of patches
in the WSI and `D` is either `512` or `1024`. Set `--input_dim` to the value
matching the extracted features. Raw WSIs may be segmented, tiled into non-overlapping
`256 x 256` patches at `20x` magnification, and encoded using an offline
feature extractor such as ResNet-50, PLIP, or UNI. The feature filename must
match the `case_id` used in the corresponding label CSV.

The label CSV files and cross-validation splits used by the provided code are
included under `labels/` and `splits/`, respectively. Raw WSIs and extracted
feature tensors are not distributed in this repository.

## Classification and subtyping

Run from the repository root. The example below uses CAMELYON16 and the
TransMIL aggregator:

```bash
python classification_subtyping/main.py \
  --dataset Camelyon_16 \
  --data_root_dir /path/to/CAMELYON16/features \
  --label_csv labels/Camelyon16_label.csv \
  --split_dir splits/5foldcv/Camelyon_16 \
  --results_dir results/classification \
  --aggregator TransMIL \
  --input_dim 1024 \
  --optimizer RAdam \
  --lr 2e-4 \
  --weight_decay 1e-5 \
  --num_epoch 200 \
  --early_stop_patience 130
```

Available values for `--aggregator` are `ABMIL`, `SAPMIL`, `TransMIL`, and
`DSMIL`.

### Independent CAMELYON17 testing

After training on the five CAMELYON16 folds, evaluate all fold-specific best
checkpoints on the labelled CAMELYON17 cohort as follows:

```bash
python classification_subtyping/independent_test.py \
  --data_root_dir /path/to/CAMELYON17/features \
  --label_csv labels/Camelyon17_label.csv \
  --checkpoint_root /path/to/CAMELYON16/checkpoints \
  --results_dir results/independent_test \
  --aggregator TransMIL \
  --input_dim 1024
```

The checkpoint directory must contain `fold_0` through `fold_4`, with each
fold containing its validation-selected `model_best_*.pth.tar` file. The
CAMELYON17 cohort is used only for independent testing: it is not used for
training, early stopping, or checkpoint selection. The script reports
fold-level metrics, their mean and standard deviation, and the optional
five-model probability ensemble.

## Survival analysis

The following example runs five-fold survival analysis on TCGA-BRCA using the
DSMIL aggregator:

```bash
python survival_analysis/main.py \
  --dataset tcga_brca \
  --data_root_dir /path/to/TCGA_BRCA/features \
  --label_csv labels/TCGA_BRCA_label.csv \
  --split_dir splits/5foldcv/TCGA_BRCA \
  --results_dir results/survival \
  --aggregator DSMIL \
  --input_dim 1024 \
  --optimizer RAdam \
  --lr 2e-4 \
  --weight_decay 1e-5 \
  --num_epoch 30
```

To select a GPU, expose the desired device before launching training:

```bash
CUDA_VISIBLE_DEVICES=0 python classification_subtyping/main.py ...
```

## Outputs

Each experiment creates a timestamped output directory containing fold-level
checkpoints and evaluation summaries. Classification reports AUC, accuracy,
and F1-score. Survival analysis reports the concordance index and exports
patient-level risk estimates.

## Acknowledgements

Many thanks to [SurMoE](https://github.com/coffeeNtv/SurMoE),
[CLAM](https://github.com/mahmoodlab/CLAM),
[ABMILX](https://github.com/DearCaat/E2E-WSI-ABMILX),
[AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL),
[TransMIL](https://github.com/szc19990412/TransMIL), and
[DSMIL](https://github.com/binli123/dsmil-wsi) for their open-source
contributions to the community.

## Citation

The citation entry will be added after publication. If this repository is
useful in your research, please cite the forthcoming PRPH-MIL paper and the
original works associated with the reused preprocessing and MIL components.
