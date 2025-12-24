import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_dmt(image_features, text_features, image_pseudo_labels, margin):
    """
    Dynamic Margin Triplet (DMT) loss
    """
    # normalized features
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    image_mask = (image_pseudo_labels.unsqueeze(0) == image_pseudo_labels.unsqueeze(1)) & (image_pseudo_labels.unsqueeze(0) != -1)
    image_labels = torch.where(image_mask, torch.tensor(1.0), torch.tensor(0.0))
    image_true_label = image_labels.fill_diagonal_(1)

    image_true_label = image_true_label.to(image_features.device)

    similarity_scores1 = torch.matmul(text_norm, image_norm.t())    
    similarity_scores2 = torch.matmul(image_norm, text_norm.t())

    positive_pair_score1 = torch.diag(similarity_scores1)
    positive_pair_score2 = torch.diag(similarity_scores2)

    negative_similarity_scores1 = torch.where(image_true_label == 0, similarity_scores1, torch.tensor(-1.0))
    negative_similarity_scores2 = torch.where(image_true_label == 0, similarity_scores2, torch.tensor(-1.0))

    negative_pair_score1,_ = negative_similarity_scores1.topk(1, dim=1, largest=True, sorted=True)   
    negative_pair_score1 = negative_pair_score1.flatten()   

    negative_pair_score2,_ = negative_similarity_scores2.topk(1, dim=1, largest=True, sorted=True)  
    negative_pair_score2 = negative_pair_score2.flatten()   

    loss1 = torch.sum(torch.clamp(negative_pair_score1 - positive_pair_score1 + margin, min=0))
    loss2 = torch.sum(torch.clamp(negative_pair_score2 - positive_pair_score2 + margin, min=0))

    loss =  (loss1 + loss2) / 2

    return loss


def compute_ndm(image_features, text_features, image_features_m, prompt_features, image_pseudo_labels, logit_scale, alpha=0.9, epsilon=1e-8):
    """
    Normalized Distribution Matching (NDM) loss
    """
    B = image_features.size(0)
    device = image_features.device

    # ========== Step 1: Construct pseudo label distribution ==========
    pseudo_mask = (image_pseudo_labels.unsqueeze(0) == image_pseudo_labels.unsqueeze(1)) & (image_pseudo_labels.unsqueeze(0) != -1)
    pseudo_label_matrix = torch.where(pseudo_mask, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    pseudo_label_matrix.fill_diagonal_(1)
    pseudo_label_dist = pseudo_label_matrix / (pseudo_label_matrix.sum(dim=1, keepdim=True))

    # ========== Step 2: Compute soft label distribution ==========
    tau_soft = 0.0002
    image_norm = F.normalize(image_features, dim=1)
    image_norm_m = F.normalize(image_features_m, dim=1)
    prompt_norm = F.normalize(prompt_features, dim=1)
    sim_prompt = (image_norm_m @ prompt_norm.t()) / tau_soft   # [B, B]
    soft_label_dist = F.softmax(sim_prompt , dim=1)  # Each row is a soft match distribution

    # ========== Step 3: MixUp labels ==========
    labels_distribute = alpha * soft_label_dist + (1 - alpha) * pseudo_label_dist  # [B, B]

    # ========== Step 4: Compute similarities ==========
    text_norm = F.normalize(text_features, dim=1)
    t2i_cosine = text_norm @ image_norm.t()
    i2t_cosine = t2i_cosine.t()

    text_proj_image = logit_scale * t2i_cosine
    image_proj_text = logit_scale * i2t_cosine

    # ========== Step 5: KL divergence ==========
    i2t_pred = F.softmax(image_proj_text, dim=1)
    t2i_pred = F.softmax(text_proj_image, dim=1)

    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return loss


def compute_ccm(image_features, text_features, image_pseudo_labels, logit_scale):
    """
    Improved Image-Text Contrastive Learning (ITC) loss
    """
    # normalized features
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    sim_matrix = torch.matmul(image_norm, text_norm.t()) * logit_scale     

    image_mask = (image_pseudo_labels.unsqueeze(0) == image_pseudo_labels.unsqueeze(1)) & (image_pseudo_labels.unsqueeze(0) != -1)
    image_labels = torch.where(image_mask, torch.tensor(1.0), torch.tensor(0.0))
    image_true_label = image_labels.fill_diagonal_(1)
    labels = image_true_label.to(image_features.device)
    
    # image-text contrastive loss
    pos_sim = torch.exp(sim_matrix) * labels  
    pos_sim_sum = pos_sim.sum(dim=1, keepdim=True)  
    
    all_sim_sum = torch.exp(sim_matrix).sum(dim=1, keepdim=True)  
    loss_i2t = -torch.log(pos_sim_sum / all_sim_sum)  
    loss_i2t = loss_i2t.mean()  
    
    # text-image contrastive loss
    pos_sim_t = torch.exp(sim_matrix.t()) * labels  
    pos_sim_sum_t = pos_sim_t.sum(dim=1, keepdim=True)  
    
    all_sim_sum_t = torch.exp(sim_matrix.t()).sum(dim=1, keepdim=True)  
    loss_t2i = -torch.log(pos_sim_sum_t / all_sim_sum_t) 
    loss_t2i = loss_t2i.mean()  
    
    # total loss
    loss = (loss_i2t + loss_t2i) / 2

    return loss


def compute_fcm(image_features, prompt_features, image_pseudo_labels, logit_scale):
    """
    Image-Prompt Contrastive Learning (IPC) loss
    """
    # normalized features
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    prompt_norm = prompt_features / prompt_features.norm(dim=1, keepdim=True)

    sim_matrix = torch.matmul(image_norm, prompt_norm.t()) * logit_scale  # (64, 64)

    image_mask = (image_pseudo_labels.unsqueeze(0) == image_pseudo_labels.unsqueeze(1)) & (image_pseudo_labels.unsqueeze(0) != -1)
    image_labels = torch.where(image_mask, torch.tensor(1.0), torch.tensor(0.0))
    image_true_label = image_labels.fill_diagonal_(1)
    labels = image_true_label.to(image_features.device)

    # image-prompt contrastive loss
    pos_sim = torch.exp(sim_matrix) * labels  
    pos_sim_sum = pos_sim.sum(dim=1, keepdim=True)  
    
    all_sim_sum = torch.exp(sim_matrix).sum(dim=1, keepdim=True) 
    loss_i2t = -torch.log(pos_sim_sum / all_sim_sum)  
    loss_i2t = loss_i2t.mean() 
    
    # prompt-image contrastive loss
    pos_sim_t = torch.exp(sim_matrix.t()) * labels  
    pos_sim_sum_t = pos_sim_t.sum(dim=1, keepdim=True)  
    
    all_sim_sum_t = torch.exp(sim_matrix.t()).sum(dim=1, keepdim=True)  
    loss_t2i = -torch.log(pos_sim_sum_t / all_sim_sum_t) 
    loss_t2i = loss_t2i.mean() 
    
    # total loss
    loss = (loss_i2t + loss_t2i) / 2

    return 0.5 * loss


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss
