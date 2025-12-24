from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb
import torch.nn.functional as F
import math
import copy


class InversionNetwork(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super(InversionNetwork, self).__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)

        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))

        self.layers = nn.Sequential(*layers)
    

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x
    

class CLPC(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.margin = args.margin
        self.tau = args.tau
        self.num_classes = num_classes
        self.momentum = args.momentum
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        # InversionNetwork 
        self.inversion_network = InversionNetwork(embed_dim=self.embed_dim, output_dim=self.embed_dim)

        # create momentum models

        self.base_model_m, _ = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self._momentum_update(0)  # initialization
        self._freeze_base_model_m()

        # Momentum InversionNetwork
        self.inversion_network_m = copy.deepcopy(self.inversion_network)
        self._momentum_update_inversion_network(0)  # initialization
        self._freeze_momentum_inversion_network()


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    @torch.no_grad()
    def _momentum_update(self, m=None):
        m = self.momentum if m is None else m
        for param_b, param_m in zip(self.base_model.parameters(), self.base_model_m.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    @torch.no_grad()
    def _freeze_base_model_m(self):
        for param in self.base_model_m.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_inversion_network(self, m=None):
        m = self.momentum if m is None else m
        for param, param_m in zip(self.inversion_network.parameters(), self.inversion_network_m.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)

    @torch.no_grad()
    def _freeze_momentum_inversion_network(self):
        for param in self.inversion_network_m.parameters():
            param.requires_grad = False



    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x  = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    
    def encode_text_with_composed_features(self, text, img_tokens):
        x = self.base_model.encode_text_img_composed(text, img_tokens)
        return x

    def forward(self, batch, flag, image_pseudo_labels=None, epoch=None):
        ret = dict()

        self._momentum_update()
        self._momentum_update_inversion_network()

        images = batch['images']
        caption_ids = batch['caption_ids']
        template_tokens = batch['caption_template']

        image_feats, text_feats = self.base_model(images, caption_ids)  

        # [CLS] token and [EOS] token
        i_feats = image_feats[:, 0, :].float()  # Extract [CLS] token for images  
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()  # Extract [EOS] token for text

        with torch.no_grad():
            image_feats_m = self.base_model_m.encode_image(images)   
        i_feats_m = image_feats_m[:, 0, :].float()  # Extract [CLS] token for images 

        copy_i_feats = i_feats.clone().detach()

        pseudo_token = self.inversion_network(i_feats.half())
        pseudo_token_m = self.inversion_network_m(i_feats.half())
        
        p_feats = self.encode_text_with_composed_features(template_tokens, pseudo_token).float()
        pm_feats = self.encode_text_with_composed_features(template_tokens, pseudo_token_m).float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if flag == True:     # calculate loss
            if 'ndm' in self.current_task:
                ret.update({'ndm_loss':objectives.compute_ndm(i_feats, t_feats, i_feats_m, pm_feats.clone().detach(), image_pseudo_labels, logit_scale)})

            if 'dmt' in self.current_task:
                mar = 0.1 + 0.1 / (1 + math.exp(-1 * (epoch - 10)))
                margin = round(mar, 3)
                ret.update({'dmt_loss':objectives.compute_dmt(i_feats, t_feats, image_pseudo_labels, margin)}) 

            if 'ccm' in self.current_task:
                ret.update({'ccm_loss':objectives.compute_ccm(i_feats, t_feats, image_pseudo_labels, logit_scale)})

            if 'fcm' in self.current_task:
                ret.update({'fcm_loss':objectives.compute_fcm(copy_i_feats, p_feats, image_pseudo_labels, logit_scale)})

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})

            return ret
        
        else :  # for clustering
            return i_feats, p_feats
            

def build_model(args, num_classes=11003):
    model = CLPC(args, num_classes)
    # covert model to fp16
    convert_weights(model)

    return model
