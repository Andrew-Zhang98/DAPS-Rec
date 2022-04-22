from .base import BaseModel
import torch.nn as nn
import torch
from models.bert_modules.embedding.bert import GRUEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class GRU4REC(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.max_len
        num_items = args.num_items
        n_layers = args.gru_layers
        # heads = args.sas_num_heads
        vocab_size = num_items + 1
        hidden = args.gru_hidden_units
        dropout = args.gru_dropout

        self.hidden = hidden
        self.max_len = max_len

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = GRUEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

    
        # define gru4rec layers
        self.emb_dropout = nn.Dropout(dropout)
        self.gru_layers = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            bias=False,
            batch_first=True,
        )
        # self.dense = nn.Linear(hidden, hidden)
        # end
        # self.apply(self._init_weights)
    
    def forward(self, x):
        item_seq_emb = self.embedding(x) # [128, 100, 64]
        # print("item_seq_emb.shape:", item_seq_emb.shape)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        # gru_output = self.dense(gru_output) # maxlen * hidden [128, 100, 64]
        # print("gru_output.shape:", gru_output.shape)
        return gru_output


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


class GRU4RECModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.gru = GRU4REC(args)
        self.out = nn.Linear(self.gru.hidden, args.num_items + 1)

    # def out(self, x):
    #     test_item_emb = self.gru.embedding.token.weight
    #     logits = torch.matmul(x, test_item_emb.transpose(0, 1))
    #     return logits

    @classmethod
    def code(cls):
        return 'gru4rec'

    def forward(self, x):
        x = self.gru(x)
        return self.out(x)
