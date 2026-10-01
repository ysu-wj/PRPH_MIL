import argparse

def parse_args():
        parser = argparse.ArgumentParser(description='Train PRPH for WSI classification or subtyping.')
        parser.add_argument('--data_root_dir', type=str, required=True,
                            help='Directory containing pre-extracted WSI feature files (*.pt)')
        parser.add_argument('--label_csv', type=str, required=True,
                            help='CSV file containing case_id, slide_id, and label columns')
        parser.add_argument('--split_dir', type=str, required=True,
                            help='Directory containing splits_0.csv, ..., splits_{K-1}.csv')
        parser.add_argument('--results_dir', type=str, default='results',
                            help='Root directory used to save checkpoints and metrics')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--dataset', type=str, default='Camelyon_16', help='Dataset name')
        parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')
        parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
        parser.add_argument('--log_data', action='store_true', default=True, help='Enable TensorBoard logging')
        parser.add_argument('--evaluate', action='store_true', help='Only evaluate')
        parser.add_argument('--resume', type=str, default='', help='Path to checkpoint')
        parser.add_argument('--model', type=str, default='prph')
        parser.add_argument('--aggregator', type=str,
                            choices=['ABMIL', 'SAPMIL', 'TransMIL', 'DSMIL'],
                            default='TransMIL', help='Slide-level MIL aggregator')
        parser.add_argument('--model_size', type=str, choices=['small', 'large'], default='small')
        parser.add_argument('--input_dim', type=int, choices=[512, 1024], default=1024,
                            help='Dimension of each patch-level input feature')
        parser.add_argument('--modal', type=str, choices=['path'], default='path')
        parser.add_argument('--optimizer', type=str, choices=['SGD','Adam','AdamW','RAdam','PlainRAdam','Lookahead'], default='RAdam')
        parser.add_argument('--scheduler', type=str, choices=['None','exp','step','plateau','cosine','linear'], default='None')
        parser.add_argument('--num_epoch', type=int, default=200)
        parser.add_argument('--early_stop_patience', type=int, default=130,help='Early stopping patience based on validation AUC. Set <=0 to disable.')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--loss', type=str, default='nll_surv')
        parser.add_argument('--weighted_sample', action='store_true', default=True)
        parser.add_argument('--lr_step', type=int, default=10)
        parser.add_argument('--lr_gamma', type=float, default=0.1)
        parser.add_argument('--log_dir', type=str, default='logs')
        parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best.pth')

        args = parser.parse_args() 
        return args
