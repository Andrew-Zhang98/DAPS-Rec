import xdrlib
from .base import BaseModel
from .sas_modules import TransformerEncoder
import torch
import torch.nn as nn

from .dan_modules import DeformableAttentionEncoder

class DANSRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_layers = args.sas_n_layers
        self.n_heads = args.sas_n_heads
        self.hidden_size = args.sas_hidden_size # same as embedding_size
        self.inner_size = args.sas_inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = args.sas_hidden_dropout_prob
        self.attn_dropout_prob = args.sas_attn_dropout_prob
        self.hidden_act = args.sas_hidden_act
        self.layer_norm_eps = args.sas_layer_norm_eps
        self.max_seq_length = args.max_len
        self.n_items = args.num_items
        self.initializer_range = args.sas_initializer_range
        self.sample_num = args.sample_num

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items+1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = DeformableAttentionEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_len = self.max_seq_length,
            sample_num = self.sample_num
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
        
    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb) # BLC

        # extended_attention_mask = None
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        # import pdb; pdb.set_trace()
        trm_output = trm_output[-1]
        return trm_output

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.initializer_range)0.02
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()





class DANRecModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.sasrec = DANSRec(args)
        self.out = nn.Linear(self.sasrec.hidden_size, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'dansrec'

    def forward(self, x):
        x = self.sasrec(x)
        return self.out(x)
