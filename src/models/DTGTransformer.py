import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

class DTGTransformer(nn.Module):
    def __init__(
            self,
            total_locations, slots_per_day,
            loc_embedding_dim=24, time_embedding_dim=6, poi_embedding_dim=6, commuter_embedding_dim=4, weekend_embedding_dim=2,
            nhead_h=8, dim_feedforward_h=1024, dropout_h=0.1, num_layers_h=2,
            head_dim=512, compress_dim=128
    ):
        # ccg 26, 8, 8, 8, 6
        # la 24, 6, 6, 4, 2
        
        super(DTGTransformer, self).__init__()

        self.total_locations = total_locations+1 # +1 for missing entries
        self.loc_embedding_dim = loc_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.poi_embedding_dim = poi_embedding_dim
        self.commuter_embedding_dim = commuter_embedding_dim
        self.weekend_embedding_dim = weekend_embedding_dim


        self.d_model = self.loc_embedding_dim + self.time_embedding_dim + self.poi_embedding_dim + self.commuter_embedding_dim
        self.head_dim = head_dim

        self.loc_embedding = nn.Embedding(num_embeddings=self.total_locations+1, embedding_dim=self.loc_embedding_dim)
        self.time_embedding = nn.Embedding(num_embeddings=slots_per_day, embedding_dim=self.time_embedding_dim)
        self.poi_embedding = nn.Embedding(num_embeddings=3+1, embedding_dim=self.poi_embedding_dim)
        self.commuter_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.commuter_embedding_dim)

        # Transformer Encoder
        self.encoder_layer_h = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead_h, dim_feedforward=dim_feedforward_h, dropout=dropout_h
        )
        self.encoder_h = TransformerEncoder(self.encoder_layer_h, num_layers=num_layers_h)

        # Output over all locations
        self.head_h = nn.Linear(self.d_model-3, self.head_dim)
        self.compress = nn.Linear(self.head_dim, compress_dim)
        self.final_linear = nn.Linear(compress_dim, self.total_locations)

        # Auxiliary task
        self.aux_linear = nn.Linear(3, 2)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
    
    def forward(self, loc, time, poi, is_commuter):
        device = next(self.parameters()).device
        batch_size = loc.shape[0]
        length_of_sequence = loc.shape[1]

        loc_x_emb = self.loc_embedding(loc).transpose(0,1)
        time_x_emb = self.time_embedding(time).transpose(0,1)
        poi_x_emb = self.poi_embedding(poi).transpose(0,1)
        commuter_x_emb = self.commuter_embedding(is_commuter).transpose(0,1)

        src = torch.cat([loc_x_emb, time_x_emb, poi_x_emb, commuter_x_emb], dim=-1)
        mem = self.encoder_h(src, mask=None)


        main_pred = self.head_h(mem[:,:,:-3])
        aux_pred = mem[:,:,-3:]

        # Main Task
        pred_matrix = F.log_softmax(self.final_linear(self.compress(main_pred)), 2)

        # Auxiliary Task
        aux_matrix = F.log_softmax(self.aux_linear(aux_pred), 2)

        return pred_matrix, aux_matrix
