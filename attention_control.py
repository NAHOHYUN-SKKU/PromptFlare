import numpy as np
import torch
import torch.nn.functional as F

class AttnController:
    def __init__(self, mask):
        self.attn2_preds = []
        self.attn2_targets = []
        self.masks = {512: mask}
    
    def __call__(self, hidden_states, module_name):
        hidden_states = hidden_states.clone()
        
        pred, target = hidden_states.chunk(2) # e.g. [1, 256, 1280]
        _, h, _ = pred.shape
        
        self.make_mask(h)
        if module_name.endswith('attn2'):
            self.attn2_preds.append(pred)
            self.attn2_targets.append(target)
    
    def reset(self):
        self.attn2_preds = []
        self.attn2_targets = []
    
    def make_mask(self, h):
        if h not in self.masks:
            size = int(np.sqrt(h))
            
            new_mask = torch.nn.functional.interpolate(self.masks[512], size=(size, size))
            new_mask = new_mask.flatten().unsqueeze(0).unsqueeze(2)
            self.masks[h] = new_mask
    
    def set_mask(self, mask):
        self.masks = {512: mask}
        
    def cal_loss(self, loss_mask, loss_depth):
        text_losses = 0
        
        attn2_preds = self.attn2_preds
        for i, (pred, target) in enumerate(zip(attn2_preds, self.attn2_targets)):
            _, h, _ = pred.shape
            
            text_loss = 0
            if h in loss_depth:
                if loss_mask:
                    pred = pred * self.masks[h]
                    target = target * self.masks[h]
                text_loss = (pred - target.detach()).norm(p=2)
            text_losses += text_loss
        
        self.reset()
        
        return text_losses

class MyAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attn_controller, module_name):
        self.attn_controller = attn_controller
        self.module_name = module_name
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.module_name.endswith('attn2'): # or self.module_name.endswith('attn1'):
            self.attn_controller(hidden_states, self.module_name)
        return hidden_states