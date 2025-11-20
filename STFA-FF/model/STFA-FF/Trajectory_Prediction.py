import math
import torch as t
from torch import nn
import torch.nn.functional as F
from einops import  repeat
from MSFDFE import MSFDFE_Module
from FDCS import FDCS_Module
from timm.models.layers import DropPath


class GDEncoder_stu(nn.Module):
    def __init__(self, args,  drop=0.3, drop_path=0.3):
        super(GDEncoder_stu, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length_stu']
        self.in_length_tea = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']
        self.in_length_stu = args['in_length_stu']

        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.qff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)

        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)

        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)
        self.exchange = nn.Linear(self.in_length, self.in_length_stu)

        self.msfdfe = MSFDFE_Module(dim=self.lstm_encoder_size, drop=drop, adaptive_filter=True)

        self.fdcs =FDCS_Module(channel=self.lstm_encoder_size)

        self.norm1 = nn.LayerNorm(self.lstm_encoder_size)
        self.norm2 = nn.LayerNorm(self.lstm_encoder_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0.1 else nn.Identity()

    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):

        if self.f_length == 5:
            hist = t.cat((hist, cls, va), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva), -1)
        elif self.f_length == 6:
            hist = t.cat((hist, cls, va, lane), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva, nbrslane), -1)

        hist = hist.permute(1, 2, 0)
        hist = hist.permute(2, 0, 1)

        hist_enc = self.activation(self.linear1(hist))
        hist_hidden_enc, _ = self.lstm(hist_enc)
        hist_hidden_enc = hist_hidden_enc.permute(1, 0, 2)

        nbrs_enc = self.activation(self.linear1(nbrs))
        nbrs_hidden_enc, _ = self.lstm(nbrs_enc)
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)
        soc_enc = soc_enc.permute(1, 2, 3, 0)
        soc_enc = soc_enc.permute(0, 2, 3, 1)
        soc_enc = soc_enc.permute(0, 3, 1, 2)
        soc_enc_exchange = self.exchange(soc_enc)
        soc_enc_exchange = soc_enc_exchange.permute(3, 0, 1, 2)

        msfdfe_out = hist_hidden_enc + self.drop_path(self.msfdfe(self.norm1(hist_hidden_enc)))

        query = self.qff(msfdfe_out)
        _, _, embed_size = query.shape
        query = t.cat(t.split(t.unsqueeze(query, 2), int(embed_size / self.n_head), -1), 1)
        keys = t.cat(t.split(self.kff(soc_enc_exchange), int(embed_size / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values_att = t.cat(t.split(self.vff(soc_enc_exchange), int(embed_size / self.n_head), -1), 0).permute(1, 0, 2,
                                                                                                              3)

        a = t.matmul(query, keys)
        a /= math.sqrt(self.lstm_encoder_size)
        a = t.softmax(a, -1)
        values_att = t.matmul(a, values_att)
        values_att = t.cat(t.split(values_att, int(hist.shape[0]), 1), -1)
        values_att = values_att.squeeze(2)

        spa_values, _ = self.first_glu(values_att)

        fdcs_out = self.fdcs(msfdfe_out)

        values = self.addAndNorm(msfdfe_out, spa_values, fdcs_out)

        return values


class Decoder_stu(nn.Module):
    def __init__(self, args):
        super(Decoder_stu, self).__init__()
        self.relu_param = args['relu']
        self.use_elu = args['use_elu']
        self.use_maneuvers = args['use_maneuvers']
        self.in_length = args['in_length_stu']
        self.out_length = args['out_length']
        self.encoder_size = args['lstm_encoder_size']
        self.device = args['device']
        self.cat_pred = args['cat_pred']
        self.use_mse = args['use_mse']
        self.lon_length = args['lon_length']
        self.lat_length = args['lat_length']

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.lstm = t.nn.LSTM(self.encoder_size, self.encoder_size)

        if self.use_mse:
            self.linear1 = nn.Linear(self.encoder_size, 2)
        else:
            self.linear1 = nn.Linear(self.encoder_size, 5)

        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):
        if self.use_maneuvers or self.cat_pred:
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = t.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)

        dec = dec.permute(1, 0, 2)
        dec = dec.permute(1, 0, 2)
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)

        if self.use_mse:
            return fut_pred
        else:
            return outputActivation(fut_pred)


class Generator_stu(nn.Module):
    def __init__(self, args):
        super(Generator_stu, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.train_flag = args['train_flag']
        self.use_maneuvers = args['use_maneuvers']
        self.lat_length = args['lat_length']
        self.lon_length = args['lon_length']
        self.use_true_man = args['use_true_man']

        self.Decoder = Decoder_stu(args=args)
        self.in_length_stu = args['in_length_stu']

        self.mapping = t.nn.Parameter(t.Tensor(self.in_length_stu, self.out_length, self.lat_length + self.lon_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)

    def forward(self, values, lat_enc, lon_enc, lat_pred_net, lon_pred_net):

        if self.train_flag:
            if self.use_true_man:
                lat_man = t.argmax(lat_enc, dim=-1).detach()
                lon_man = t.argmax(lon_enc, dim=-1).detach()
            else:
                lat_man = t.argmax(lat_pred_net, dim=-1).detach().unsqueeze(1)
                lon_man = t.argmax(lon_pred_net, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = t.zeros_like(lat_pred_net)
                lon_enc_tmp = t.zeros_like(lon_pred_net)
                lat_man_idx = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man_idx = lon_enc_tmp.scatter_(1, lon_man, 1)
                lat_enc = lat_man_idx
                lon_enc = lon_man_idx
                lat_man = t.argmax(lat_pred_net, dim=-1).detach()
                lon_man = t.argmax(lon_pred_net, dim=-1).detach()

            if lat_man.dim() == 1:
                lat_man = lat_man.unsqueeze(1)
            if lon_man.dim() == 1:
                lon_man = lon_man.unsqueeze(1)

            lat_enc_onehot = t.zeros(lat_enc.shape[0], self.lat_length).to(self.device).scatter_(1, lat_man, 1)
            lon_enc_onehot = t.zeros(lon_enc.shape[0], self.lon_length).to(self.device).scatter_(1, lon_man, 1)

            index = t.cat((lat_enc_onehot, lon_enc_onehot), dim=-1).permute(-1, 0)

            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)

            fut_pred = self.Decoder(dec, lat_enc, lon_enc)
            return fut_pred

        else:
            out = []
            for k in range(self.lon_length):
                for l in range(self.lat_length):
                    lat_enc_tmp = t.zeros(lat_pred_net.shape[0], self.lat_length).to(self.device)
                    lon_enc_tmp = t.zeros(lon_pred_net.shape[0], self.lon_length).to(self.device)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1

                    index = t.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)
                    mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                    dec = t.matmul(mapping, values).permute(1, 0, 2)

                    fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)
                    out.append(fut_pred)

            return out, lat_pred_net, lon_pred_net


# 辅助模块
class GLU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = t.exp(sigX)
    sigY = t.exp(sigY)
    rho = t.tanh(rho)
    out = t.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out
