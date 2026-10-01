import os
os.environ["OMP_NUM_THREADS"] = "1"
import csv
import time
import numpy as np
import pandas as pd
import torch

from datasets.dataset_classification import Generic_MIL_Classification_Dataset
from models.engine import Engine
from utils.options import parse_args
from utils.util import get_split_loader, set_seed
from torch.nn import CrossEntropyLoss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from models.network_TransMIL import PRPH as PRPHTransMIL
from models.network_ABMIL import PRPH as PRPHABMIL
from models.network_DSMIL import PRPH as PRPHDSMIL
from models.network_SAPMIL import PRPH as PRPHSAPMIL
from datetime import timedelta


MODEL_REGISTRY = {
    "ABMIL": PRPHABMIL,
    "SAPMIL": PRPHSAPMIL,
    "TransMIL": PRPHTransMIL,
    "DSMIL": PRPHDSMIL,
}


def main(args):
    total_start_time = time.time()

    # create results directory
    results_dir = os.path.join(
        args.results_dir,
        args.dataset,
        f"{args.model}_{args.optimizer}_{args.lr}_{time.strftime('%Y%m%d_%H%M')}",
    )
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(args.label_csv):
        raise FileNotFoundError(f"Label CSV not found: {args.label_csv}")
    if not os.path.isdir(args.split_dir):
        raise FileNotFoundError(f"Split directory not found: {args.split_dir}")

    # K-fold cross validation
    all_results = []
    all_epochs = []

    # start 5-fold CV evaluation.
    for fold in range(args.num_folds):
        # 每个 fold 单独固定随机种子
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        print(f"===== Fold {fold}, seed = {fold_seed} =====")
        # build dataset
        dataset = Generic_MIL_Classification_Dataset(
            csv_path=args.label_csv,
            modal=args.modal,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=fold_seed,
            patient_strat=False,
            label_col="label",
        )
        split_csv = os.path.join(args.split_dir, f"splits_{fold}.csv")
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path=split_csv
        )
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size,
        )
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )

        # build model, criterion, optimizer, schedular
        model_dict = {
            "n_classes": args.num_classes,
            "model_size": args.model_size,
            "input_dim": args.input_dim,
        }
        model = MODEL_REGISTRY[args.aggregator](**model_dict)
        criterion = CrossEntropyLoss()
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        engine = Engine(args, results_dir, fold)

        # start training
        best_metrics, best_epoch = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        
        all_results.append(best_metrics)
        all_epochs.append(best_epoch)

    total_end_time = time.time()
    total_duration_seconds = total_end_time - total_start_time
    total_time_str = str(timedelta(seconds=int(total_duration_seconds)))
    print(f"====== Total running time: {total_duration_seconds:.2f} seconds ({total_time_str}) ======")

    # Process and save results
    results_df = pd.DataFrame(all_results)
    results_df['fold'] = range(args.num_folds)
    results_df['best_epoch'] = all_epochs
    
    # Calculate mean and std
    mean_results = results_df.mean().to_dict()
    std_results = results_df.std().to_dict()
    
    summary_df = pd.DataFrame([mean_results, std_results], index=['mean', 'std'])
    
    # Save detailed and summary results
    results_df.to_csv(os.path.join(results_dir, "detailed_results.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "summary_results.csv"))

    print("Full Results:")
    print(results_df)
    print("Summary:")
    print(summary_df)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("finished!")
