from torch import nn as nn
import torch
import numpy as np
import copy
import math
import torch.nn.functional as fn
# import einops
from .sas_modules import TransformerLayer

class SampleNet(nn.Module):
    def __init__(self, hidden_size, max_len, sample_num, layer_norm_eps, reduction=2):
        super(SampleNet, self).__init__()
        L = max_len
        self.N = sample_num
        reduction = reduction
        self.query = nn.Linear(hidden_size, hidden_size)

        self.sample_layer = nn.Sequential(
            nn.Linear(L, L//reduction),
            nn.LayerNorm(L//reduction, eps=layer_norm_eps),
            nn.GELU(),
            nn.Linear(L//reduction, L))

        self.off_set_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            nn.GELU(),
            nn.Linear(hidden_size, sample_num*2))

    def forward(self, x):
        query = self.query(x)
        offset = query.permute(0,2,1)
        offset = self.sample_layer(offset) # B,C,L
        offset = self.off_set_layer(offset.permute(0,2,1)) # B,L, 2*N
        return offset

class DeformableAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12, 
        max_len=50, 
        sample_num =20
    ):
        super(DeformableAttentionEncoder, self).__init__()
        sample_num = sample_num
        layer = DeformableAttentionLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, max_len, sample_num
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

        self.sample_layer = SampleNet(hidden_size, max_len, sample_num, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        offset = self.sample_layer(hidden_states)
        pos= None
        for layer_module in self.layer:
            hidden_states, offset, pos = layer_module(hidden_states, offset, pos)
            # out_pos.append(pos)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class DeformableAttentionLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps, max_len, sample_num
    ):
        super(DeformableAttentionLayer, self).__init__()
        self.multi_head_attention = Deformable_MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, max_len, sample_num
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, offset, pos):
        attention_output, offset, pos = self.multi_head_attention(hidden_states, offset, pos)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, offset, pos



class AdjustNet(nn.Module):
    def __init__(self, hidden_size, max_len, sample_num, layer_norm_eps):
        super(AdjustNet, self).__init__()
        L = max_len
        self.N = sample_num
        self.pre_ref = self._pre_ref_points(L)
        self.pre_mask = self._pre_temporal_mask(L)

        self.sample_layer = nn.Sequential(
            nn.Linear(L, L//4),
            nn.LayerNorm(L//4, eps=layer_norm_eps),
            nn.GELU(),
            nn.Linear(L//4, L))

        reduction = 1
        self.adjust_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//reduction),
            nn.LayerNorm(hidden_size//reduction, eps=layer_norm_eps),
            nn.GELU(),
            nn.Linear(hidden_size//reduction, sample_num*2))

    def forward(self, query, x, offset, ref=None):
        B, L, C = x.size()
        dtype, device = x.dtype, x.device

        if ref is None:
            reference = self._get_ref_points(L, B, dtype, device)
        else:
            reference = ref

        query = query.permute(0,2,1) #B,C,L 
        query = self.sample_layer(query).permute(0,2,1) #  BLC
        adjust = self.adjust_layer(query).tanh() * 0.02

        local_offset = offset + adjust
        
        local_offset = local_offset.reshape(B,L,self.N,2)
        pos = (local_offset + reference).tanh() # B,L,N,2
        t_mask = self._get_temporal_mask(L,B,dtype, device)
       
        x = self.sample(pos, t_mask, x)
        return x, offset, pos
    
    def sample(self, pos, t_mask, x):
        B, L, C = x.size()
        x = x.permute(0,2,1).unsqueeze(-1) #BCL1
        x = x.unsqueeze(1).repeat(1,L,1,1,1) #B,L, C,L,1
        x = x*t_mask
        x = x.reshape(B*L,C,L,1) #B*L, C,L,1
        pos = pos.reshape(B*L,self.N,2)#B*L, N, 2
        x_sampled = fn.grid_sample(
            input=x.contiguous(), 
            grid=pos.unsqueeze(-2).contiguous(), # B*L, N, 1, 2
            mode='bilinear', align_corners=True) # B*L ,C, N,1
        
        x_sampled = x_sampled.squeeze(-1)
        x_sampled = x_sampled.reshape(B,L,C,self.N)
        return x_sampled.permute(0,1,3,2) # B,L,N,C
        
    @torch.no_grad()
    def _get_ref_points(self, L, B, dtype, device):
        ######## pre_ref
        l_ref = self.pre_ref
        l_ref = l_ref[None, ...].expand(B ,-1,-1,-1)
        return l_ref.to(device)
    
    @torch.no_grad()
    def _pre_ref_points(self, L):
        l_key = self.N
        l_ref=[]
        for i in range(L):
            ref = torch.range(0.5, l_key-0.5, 1)
            ref.div_(l_key*L/(i+1e-9)).mul_(2).sub_(1)
            pad = torch.zeros(l_key).float()
            ref = torch.stack([pad, ref],dim=-1)
            l_ref.append(ref)
        l_ref = torch.stack(l_ref,dim=0) #L,N,2
        return l_ref

    @torch.no_grad()
    def _get_temporal_mask(self, L, B, dtype, device):
        mask = self.pre_mask
        return mask.to(device)
    
    @torch.no_grad()
    def _pre_temporal_mask(self, L):
        attn_shape = (L, L)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0)
        subsequent_mask = subsequent_mask.long()
        return subsequent_mask.unsqueeze(1).unsqueeze(0).unsqueeze(-1)


class Deformable_MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, max_len, sample_num):
        super(Deformable_MultiHeadAttention, self).__init__()
        
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.SN = sample_num
        self.sample_layer = AdjustNet(hidden_size, max_len, sample_num= self.SN, layer_norm_eps=layer_norm_eps)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, offset, pos):
        B,L,C = input_tensor.shape
        N = self.SN

        query_layer = self.query(input_tensor)

        sampled_x, offset, pos = self.sample_layer(query_layer, input_tensor, offset, ref = pos) # B, L, N, C
        key_layer = self.key(sampled_x)
        value_layer = self.value(sampled_x)


        query_layer = query_layer.permute(0,2,1).reshape(B*self.num_attention_heads, C//self.num_attention_heads, L)
        value_layer = value_layer.permute(0,3,1,2).reshape(B*self.num_attention_heads, C//self.num_attention_heads,L,N)
        key_layer = key_layer.permute(0,3,1,2).reshape(B*self.num_attention_heads, C//self.num_attention_heads,L,N)


        attention_scores = torch.einsum('b c m, b c m k-> b m k', query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.einsum('b m k, b c m k -> b c m', attention_probs, value_layer).reshape(B,C,L).permute(0,2,1)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, offset, pos

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states




class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        # if hidden_size % n_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (hidden_size, n_heads)
        #     )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states