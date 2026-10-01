# Example commands

Run all commands from the repository root and replace feature paths with local
directories containing the pre-extracted `.pt` tensors.

## CAMELYON16 diagnosis

```bash
CUDA_VISIBLE_DEVICES=0 python classification_subtyping/main.py \
  --dataset Camelyon_16 \
  --data_root_dir /path/to/CAMELYON16/features \
  --label_csv labels/Camelyon16_label.csv \
  --split_dir splits/5foldcv/Camelyon_16 \
  --results_dir results/classification \
  --aggregator TransMIL \
  --input_dim 1024
```

## TCGA-NSCLC subtyping

```bash
CUDA_VISIBLE_DEVICES=0 python classification_subtyping/main.py \
  --dataset TCGA_NSCLC \
  --data_root_dir /path/to/TCGA_NSCLC/features \
  --label_csv labels/TCGA_NSCLC_label.csv \
  --split_dir splits/5foldcv/TCGA_NSCLC \
  --results_dir results/classification \
  --aggregator DSMIL \
  --input_dim 1024
```

## CAMELYON17 independent testing

Use the best checkpoint from each CAMELYON16 fold to evaluate the labelled
CAMELYON17 cohort. Set `--aggregator` to the same aggregator used for training.

```bash
CUDA_VISIBLE_DEVICES=0 python classification_subtyping/independent_test.py \
  --data_root_dir /path/to/CAMELYON17/features \
  --label_csv labels/Camelyon17_label.csv \
  --checkpoint_root /path/to/CAMELYON16/checkpoints \
  --results_dir results/independent_test \
  --aggregator TransMIL \
  --input_dim 1024
```

## TCGA-BRCA survival analysis

```bash
CUDA_VISIBLE_DEVICES=0 python survival_analysis/main.py \
  --dataset tcga_brca \
  --data_root_dir /path/to/TCGA_BRCA/features \
  --label_csv labels/TCGA_BRCA_label.csv \
  --split_dir splits/5foldcv/TCGA_BRCA \
  --results_dir results/survival \
  --aggregator DSMIL \
  --input_dim 1024
```

## TCGA-BLCA survival analysis

```bash
CUDA_VISIBLE_DEVICES=0 python survival_analysis/main.py \
  --dataset tcga_blca \
  --data_root_dir /path/to/TCGA_BLCA/features \
  --label_csv labels/TCGA_BLCA_label.csv \
  --split_dir splits/5foldcv/TCGA_BLCA \
  --results_dir results/survival \
  --aggregator ABMIL \
  --input_dim 1024
```
