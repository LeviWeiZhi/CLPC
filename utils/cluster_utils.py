import torch
import pdb
import numpy as np

@torch.no_grad()
def most_similar_index_sort(query_feature, bank_features, matching_indices):
    matching_indices = matching_indices.to(bank_features.device)
    subset_bank_features = bank_features[matching_indices]  
    similarities = torch.matmul(query_feature, subset_bank_features.t()) 
    sorted_indices = torch.argsort(similarities, descending=True)  
    most_similar_index = matching_indices[sorted_indices]
    return most_similar_index


@torch.no_grad()
def pseudo_labels_mining(image_bank, image_pseudo_labels, text_bank, text_pseudo_labels):
    image_labels = torch.tensor(image_pseudo_labels)
    text_labels = torch.tensor(text_pseudo_labels)

    for i in range(len(image_bank)):
        if image_labels[i] == -1:
            if text_labels[i] == -1: 
                continue
            else:

                cluster_indexs = torch.where(text_labels == text_labels[i])
                cluster_index_set = cluster_indexs[0][cluster_indexs[0] != i]

                matching_indices_sort = most_similar_index_sort(image_bank[i], image_bank, cluster_index_set)

                most_similar_index = -1

                if matching_indices_sort.numel() < 1:
                    continue
                else:
                    most_similar_index = next((index for index in matching_indices_sort if text_labels[index] != -1), -1)                                          

                if most_similar_index != -1:
                    image_labels[i] = image_labels[most_similar_index]

    return image_labels.numpy()
