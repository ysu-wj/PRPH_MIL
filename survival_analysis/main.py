import os
os.environ["OMP_NUM_THREADS"] = "1"
import csv
import time
import numpy as np
import pandas as pd
import torch
from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from models.engine import Engine
from utils.options import parse_args
from utils.util import get_split_loader, set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from models.network_DSMIL import PRPH as PRPHDSMIL
from models.network_ABMIL import PRPH as PRPHABMIL
from models.network_SAPMIL import PRPH as PRPHSAPMIL
from models.network_TransMIL import PRPH as PRPHTransMIL
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_REGISTRY = {
    "ABMIL": PRPHABMIL,
    "SAPMIL": PRPHSAPMIL,
    "TransMIL": PRPHTransMIL,
    "DSMIL": PRPHDSMIL,
}


def main(args):
    total_start_time = time.time()
    # set random seed for reproduction
    set_seed(args.seed)

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

    # 5-fold cross validation
    header = ["folds"] + [f"fold {fold}" for fold in range(args.num_folds)] + ["mean", "std", "total time"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]
    risk_csv_list = []
    # start 5-fold CV evaluation.
    for fold in range(args.num_folds):
        # build dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path=args.label_csv,
            modal=args.modal,
            apply_sig=True,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            
            patient_strat=False,
            n_bins=4,
            label_col=args.label_col,
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
        if args.aggregator == "TransMIL":
            model = PRPHTransMIL(n_classes=args.num_classes, input_dim=args.input_dim)
        else:
            model = MODEL_REGISTRY[args.aggregator](
                n_classes=args.num_classes,
                model_size=args.model_size,
                input_dim=args.input_dim,
            )
        criterion = define_loss(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        engine = Engine(args, results_dir, fold)

        # start training
        score, epoch, risks = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        risk_csv_list.append(risks)

        # save best score and epoch for each fold
        best_epoch.append(epoch)
        best_score.append(score)

    if risk_csv_list:
        risk_csv = pd.concat(risk_csv_list)
        risk_csv.to_csv(f'{results_dir}/risk.csv', index=False)
    total_end_time = time.time()
    total_duration_seconds = total_end_time - total_start_time
    total_time_str = str(timedelta(seconds=int(total_duration_seconds)))  # 格式化为时:分:秒
    print(f"\n====== 总运行时间: {total_duration_seconds:.2f}秒 ({total_time_str}) ======")
    # finish training
    # mean and std
    fold_scores = best_score[1:1 + args.num_folds]
    best_epoch.extend(["~"] * 3)
    best_score.extend([np.mean(fold_scores), np.std(fold_scores), total_time_str])

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
