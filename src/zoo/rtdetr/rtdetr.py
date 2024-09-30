"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register

class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'affordance_branch']

    def __init__(self, backbone: nn.Module, encoder, decoder, affordance_branch, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.affordance_branch = affordance_branch
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        # Backbone
        x = self.backbone(x)
        
        # Encoder
        encoder_output = self.encoder(x)
        
        # Decoder
        decoder_output = self.decoder(encoder_output, targets)

        # Affordance branch
        affordance_output = self.affordance_branch(decoder_output['features'])

        # Combine outputs
        output = {
            'pred_logits': decoder_output['pred_logits'],
            'pred_boxes': decoder_output['pred_boxes'],
            'pred_affordances': affordance_output
        }

        # Include auxiliary outputs if present
        if 'aux_outputs' in decoder_output:
            output['aux_outputs'] = []
            for aux_out in decoder_output['aux_outputs']:
                aux_affordance = self.affordance_branch(aux_out['features'])
                output['aux_outputs'].append({
                    'pred_logits': aux_out['pred_logits'],
                    'pred_boxes': aux_out['pred_boxes'],
                    'pred_affordances': aux_affordance
                })

        # Include denoising outputs if present
        if 'dn_aux_outputs' in decoder_output:
            output['dn_aux_outputs'] = []
            for dn_aux_out in decoder_output['dn_aux_outputs']:
                dn_aux_affordance = self.affordance_branch(dn_aux_out['features'])
                output['dn_aux_outputs'].append({
                    'pred_logits': dn_aux_out['pred_logits'],
                    'pred_boxes': dn_aux_out['pred_boxes'],
                    'pred_affordances': dn_aux_affordance
                })

        if 'dn_meta' in decoder_output:
            output['dn_meta'] = decoder_output['dn_meta']

        return output
    
    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self