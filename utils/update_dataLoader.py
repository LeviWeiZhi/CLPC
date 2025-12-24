import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import pdb
import numpy as np  
from collections import defaultdict
import math

class PseudoLabeledDataset(Dataset):
    def __init__(self, dataset, pseudo_labels):
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, index):
        data = self.dataset[index]
        data['pseudo_label'] = self.pseudo_labels[index]
        return data

    def __len__(self):
        return len(self.dataset)


def add_pseudo_label(train_loader, image_pseudo_labels):
    image_pseudo_labels = torch.tensor(image_pseudo_labels)
    
    pseudo_labeled_dataset = PseudoLabeledDataset(train_loader.dataset, image_pseudo_labels)
    
    new_train_loader = DataLoader(pseudo_labeled_dataset, 
                                  batch_size=train_loader.batch_size, 
                                  shuffle=True, 
                                  num_workers=train_loader.num_workers,
                                  collate_fn=train_loader.collate_fn)

    return new_train_loader



def update_train_loader(train_loader, pseudo_labels, args, logger):
    valid_indices = np.where(pseudo_labels != -1)[0]

    num_batches = math.ceil(len(valid_indices) / args.batch_size)
    
    new_dataset = Subset(train_loader.dataset, valid_indices)
    

    new_train_loader = DataLoader(
        dataset=new_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=train_loader.collate_fn,
        num_workers=args.num_workers,  
        pin_memory=True                
    )
    
    logger.info(f"New train loader created with {len(valid_indices)} samples, forming {num_batches} batches.")
    
    return new_train_loader


