from __future__ import division
import torch
import torch.nn as nn
import math
from einops import repeat
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv



class ShiftedWindowAttention(nn.Module):

    def __init__(self, d_model=64, n_heads=4, window_size=5, dropout=0.1):
        super(ShiftedWindowAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads


        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape
        residual = x

        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        num_windows = (seq_len + self.window_size - 1) // self.window_size
        pad_len = num_windows * self.window_size - seq_len

        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        padded_seq_len = seq_len + pad_len

        Q = Q.view(batch_size, self.n_heads, num_windows, self.window_size, self.head_dim)
        K = K.view(batch_size, self.n_heads, num_windows, self.window_size, self.head_dim)
        V = V.view(batch_size, self.n_heads, num_windows, self.window_size, self.head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = torch.tanh(attn_output)

        attn_output = attn_output.view(batch_size, self.n_heads, padded_seq_len, self.head_dim)

        if pad_len > 0:
            attn_output = attn_output[:, :, :seq_len, :]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.out_linear(attn_output)
        output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Informer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, window_size=5, dropout=0.1):
        super(Informer, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=False,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.shifted_window_attn = ShiftedWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, space_details, time_details):

        batch_size = space_details.shape[0]
        space_details = space_details.view(batch_size, 39, 64)
        time_details = time_details.view(batch_size, 30, 64)
        memory = space_details.permute(1, 0, 2)
        tgt = time_details.permute(1, 0, 2)

        transformer_output = self.transformer_decoder(tgt, memory)
        transformer_output = self.dropout(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)

        time_with_window_attn = self.shifted_window_attn(time_details)
        space_global = space_details.mean(dim=1, keepdim=True)
        space_global = space_global.expand(-1, 30, -1)

        fused_features = transformer_output + time_with_window_attn + space_global

        fused_output = self.fusion_mlp(fused_features)

        return fused_output


class Graph_Convolution(nn.Module):
    def __init__(self, hidden_channels, out_channels, kernel_size, heads=8):
        super(Graph_Convolution, self).__init__()

        self.activation = nn.ELU()
        self.gru = torch.nn.GRU(16, hidden_size=1, batch_first=True)
        self.gat1 = GATv2Conv((39 * 2), hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 16, kernel_size, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.3, inplace=False)
        )


    def forward(self, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch,
                graph_matrix):
        batch_size = ve_matrix_batch.shape[0]

        edge_index_batch = edge_index_batch.to(device)
        mask_view_batch = mask_view_batch.to(device)
        man_matrix_batch = man_matrix_batch.to(device)
        ac_matrix_batch = ac_matrix_batch.to(device)
        ve_matrix_batch = ve_matrix_batch.to(device)


        has_nan = torch.isnan(man_matrix_batch)
        man_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), man_matrix_batch)
        has_nan = torch.isnan(ac_matrix_batch)
        ac_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), ac_matrix_batch)
        has_nan = torch.isnan(ve_matrix_batch)
        ve_matrix_batch = torch.where(has_nan, torch.tensor(0.0, device=device), ve_matrix_batch)

        man_matrix_batch1 = torch.unsqueeze(man_matrix_batch, dim=1)
        ac_matrix_batch1 = torch.unsqueeze(ac_matrix_batch, dim=1)
        ve_matrix_batch1 = torch.unsqueeze(ve_matrix_batch, dim=1)
        conv_matrix = torch.cat((man_matrix_batch1, ac_matrix_batch1, ve_matrix_batch1), dim=1)
        conv_matrix = self.conv(conv_matrix)

        outputs = []
        for i in range(conv_matrix.size(3)):
            part = conv_matrix[:, :, :, i]
            part = part.permute(0, 2, 1)
            out, _ = self.gru(part)
            outputs.append(out)
        conv_enc1 = torch.cat(outputs, dim=-1)

        mask_view_batch = torch.flatten(mask_view_batch, start_dim=1, end_dim=2)
        mask_view_batch = mask_view_batch.unsqueeze(1)
        conv_enc2 = conv_enc1 * mask_view_batch
        man_matrix_batch2 = man_matrix_batch * mask_view_batch
        graph_matrix = graph_matrix.to(device)
        graph_matrix = torch.cat((man_matrix_batch2, conv_enc2), dim=1)
        graph_matrix = graph_matrix.permute(0, 2, 1)
        x = graph_matrix.reshape(-1, (39 * 2))
        edge_index = edge_index_batch.view(2, -1)

        h = self.gat1(x, edge_index.long())
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index.long())
        h = F.dropout(h, p=0.2, training=self.training)
        output = h.view(batch_size, 39, 64)
        return output


class highwayNet(nn.Module):

    def __init__(self, args):
        super(highwayNet, self).__init__()

        self.args = args
        self.use_cuda = args['use_cuda']
        self.use_maneuvers = args['use_maneuvers']
        self.train_flag = args['train_flag']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth
        self.in_channels = args['in_channels']
        self.out_channels = args['out_channels']
        self.kernel_size = args['kernel_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.dropout = args['dropout']
        self.nbr_max = args['nbr_max']
        self.hidden_channels = args['hidden_channels']
        self.lat_length = 3
        self.lon_length = 3
        self.device=args['device']

        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
        self.up_emb = torch.nn.Linear(1, self.input_embedding_size)
        self.linear1 = nn.Linear(6, 32)
        self.linear2 = nn.Linear(6, 32)
        self.activation = nn.ELU()
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        self.gru = torch.nn.GRU(self.input_embedding_size, self.encoder_size, 2, batch_first=True)
        self.lstm = nn.LSTM(self.input_embedding_size, self.encoder_size)
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        self.gcn = Graph_Convolution(self.in_channels, self.hidden_channels, self.out_channels, self.kernel_size)

        self.informer = Informer(
            d_model=self.encoder_size,
            n_heads=self.n_head,
            window_size=5,
            dropout=self.dropout
        )

        self.qt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.addAndNorm = AddAndNorm(self.encoder_size)
        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout)
        self.second_glu = GLU(
            input_size=self.encoder_size,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout)
        self.normalize = nn.LayerNorm(self.encoder_size)
        self.mu_fc1 = nn.Linear(self.encoder_size, self.n_head * self.att_out)
        self.mu_fc = nn.Linear(self.n_head * self.att_out, self.encoder_size)

        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        self.op_lat = nn.Linear(self.encoder_size, self.num_lat_classes)
        self.op_lon = nn.Linear(self.encoder_size, self.num_lon_classes)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        self.tea_exchange = torch.nn.Linear(31, 30)

    def forward(self, hist, nbrs, masks, lat_enc, lon_enc, lane, nbrslane, cls, nbrscls, va, nbrsva, edge_index_batch,
                ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch, graph_matrix):

        space_details = self.gcn(edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, mask_view_batch,
                                 graph_matrix)

        hist1 = torch.cat((hist, cls, lane, va), -1)
        nbrs1 = torch.cat((nbrs, nbrscls, nbrslane, nbrsva), -1)
        hist_enc = self.activation(self.linear1(hist1))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        time_self_enc = hist_hidden_enc.permute(1, 0, 2)
        time_self_enc1 = hist_hidden_enc.permute(1, 2, 0)
        time_self_enc1 = self.tea_exchange(time_self_enc1)
        time_self_enc = time_self_enc1.permute(0, 2, 1)

        nbrs_enc = self.activation(self.linear1(nbrs1))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc)

        mask = masks.view(masks.size(0), masks.size(1) * masks.size(2), masks.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        soc_enc = torch.zeros_like(mask).float()
        time_nbrs_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)

        query = self.qt(time_self_enc)
        _, _, embed_size = query.shape
        query = torch.cat(torch.split(torch.unsqueeze(query, 2), int(embed_size / self.n_head), -1), 1)
        keys = torch.cat(torch.split(self.kt(time_nbrs_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values = torch.cat(torch.split(self.vt(time_nbrs_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 2,
                                                                                                              3)

        a = torch.matmul(query, keys)
        a /= math.sqrt(self.encoder_size)
        a = torch.softmax(a, -1)
        values = torch.matmul(a, values)

        values = torch.cat(torch.split(values, int(hist.shape[0] - 1), 1), -1)
        values = values.squeeze(2)
        time_values, _ = self.first_glu(values)
        time_detiles = self.addAndNorm(time_self_enc, time_values)

        result = self.informer(space_details, time_detiles)

        enc, _ = self.second_glu(result)
        enc1 = enc[:, -1, :]

        if self.use_maneuvers:
            maneuver_state = self.activation(self.mu_fc1(enc1))
            maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))

            lat_pred = self.softmax(self.op_lat(maneuver_state))
            lon_pred = self.softmax(self.op_lon(maneuver_state))

            return lat_pred, lon_pred

        else:
            return None, None


class GLU(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation, gated), gated


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.add(torch.add(x1, x2), x3)
        else:
            x = torch.add(x1, x2)
        return self.normalize(x)