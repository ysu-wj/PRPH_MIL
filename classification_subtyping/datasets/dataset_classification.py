from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path, shuffle=False, seed=7, patient_strat=False, label_col=None, filter_dict={}):
        self.seed = seed
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        
        slide_data = pd.read_csv(csv_path, low_memory=False)

        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        if 'case_id' not in slide_data:
            slide_data['case_id'] = slide_data.index.str[:12]
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'label'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        self.slide_data = slide_data
        self.patient_data = slide_data.drop_duplicates(['case_id']).copy()
        self.num_classes = len(self.slide_data[self.label_col].value_counts())

        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data[self.label_col] == i)[0]

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_col] == i)[0]

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data)
        else:
            return len(self.slide_data)

    def get_split_from_df(self, all_splits: dict, split_key: str = 'train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, num_classes=self.num_classes)
        else:
            split = None
        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data[self.label_col][ids]


class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, modal: str = 'path', **kwargs):
        super(Generic_MIL_Classification_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.modal = modal
        self.use_h5 = False

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data[self.label_col][idx]

        if self.modal == 'path':
            try:
                wsi_path = os.path.join(self.data_dir, f"{case_id}.pt")
                path_features = torch.load(wsi_path)
                if len(path_features.shape) == 1:
                    path_features = path_features.unsqueeze(0)
                return (path_features, label, case_id)
            except FileNotFoundError:
                print(f"Warning: Feature file not found for case_id {case_id} at path: {wsi_path}")
                placeholder = torch.zeros(1, 1024) # Adjust dimension if necessary
                return (placeholder, label, case_id)
            except Exception as e:
                print(f"Error loading features for case_id {case_id}: {e}")
                placeholder = torch.zeros(1, 1024)
                return (placeholder, label, case_id)
        else:
            raise NotImplementedError('Modal [%s] not implemented.' % self.modal)


class Generic_Split(Generic_MIL_Classification_Dataset):
    def __init__(self, slide_data, data_dir=None, label_col=None, num_classes=2):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.label_col = label_col
        self.num_classes = num_classes
        self.modal = 'path' # Assuming 'path' modal for splits

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_col] == i)[0]

    def __len__(self):
        return len(self.slide_data)
