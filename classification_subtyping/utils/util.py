import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

def collate_path(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    # event_time = np.array([item[2] for item in batch])
    # c = torch.FloatTensor([item[3] for item in batch])
    case_id = [item[2] for item in batch]
    return [img, label,  case_id]

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_split_loader(split_dataset, training=False, testing=False, weighted=False, modal='path', batch_size=1):
    collate = collate_path
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(split_dataset)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate, **kwargs)
    else: # For validation or testing
        sampler = SequentialSampler(split_dataset)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate, **kwargs)
    
    return loader

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True