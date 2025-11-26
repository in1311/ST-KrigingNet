import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class STKrigingNet(nn.Module):
    def __init__(self, d_input, d_model, d_trend, known_num, weight, device):
        super(STKrigingNet, self).__init__()
        """
        The initializer.

        ----------
        Parameters
        d_input: int
            The dimension of input features.
        d_model: int
            The dimension of embedding space (hidden units of each dense layer).
        d_trend: int
            The dimension of trend vector.
        known_num: int
            The number of known points (observed points).
        weight: float
            The weight of position embedding and time embedding.
        device: torch.device
            The device to use (CPU or GPU).
        """
        self.d_model = d_model
        self.d_trend = d_trend
        self.known_num = known_num
        self.AttPrj = Attribute_prj(d_input, d_model)
        self.Scan = AttNet(d_model, weight)
        self.Tcan = AttNet(d_model, weight)
        self.STCovNet = STCovNet(known_num, d_model)
        self.STTN = STTrendNet(d_model, d_trend)
        self.device = device
    
    def STK_decoder(self, z_know, cov_know, cov_unknow, trend_unknow, trend_know):
        
        ##### Spatiotemporal kriging decoder  #####
        device = str(cov_know.device) 
        k = trend_know.shape[-1]
        batch_size = cov_unknow.shape[0]

        # Kriging system matrix
        sys_mat_know = torch.zeros(self.known_num + k, self.known_num + k).to(device)
        sys_mat_unknow = torch.zeros(batch_size, self.known_num + k).to(device)

        sys_mat_know[0:self.known_num, 0:self.known_num] = cov_know
        sys_mat_know[0:self.known_num, self.known_num:self.known_num + k] = trend_know
        sys_mat_know[self.known_num:self.known_num + k, 0:self.known_num] = trend_know.T

        sys_mat_unknow[0:batch_size, 0:self.known_num] = cov_unknow
        sys_mat_unknow[0:batch_size, self.known_num:self.known_num + k] = trend_unknow
        
        ##### Solving the interpolation weights #####
        try:
            sys_mat_know_inv = torch.linalg.inv(sys_mat_know)
        except:
            sys_mat_know_inv = torch.linalg.pinv(sys_mat_know)
        lamda = torch.matmul(sys_mat_unknow, sys_mat_know_inv.T)
        lamda = lamda[:, :-k]
        
        ##### Estimating #####
        trend_unknow_pre = trend_unknow.mean(axis=-1)
        trend_know_pre = trend_know.mean(axis=-1)
        prediction = torch.matmul(lamda, z_know)
        # prediction = torch.matmul(lamda, z_know) + trend_unknow_pre

        return prediction, trend_know_pre, trend_unknow_pre

    def reset_parameters(self):
        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def position_prj(self, coodx, coody):
        """
        Two-dimensional rotational position encoding
        Reference: https://spaces.ac.cn/archives/8397
        """
        d_model = self.d_model
        if d_model % 4 != 0:
            print('d % 4 != 0')
            return 0
        device = self.device
        b = coodx.shape[0]
        pe = torch.zeros(b, d_model, device=device, requires_grad=False).float()
        _4i = torch.arange(0,  d_model, step=4, device=device).float()

        div_term = 10 ** (-_4i / d_model) * math.pi / 2 

        coodx = coodx.unsqueeze(1)
        coody = coody.unsqueeze(1)

        pe[:, 0::4] = torch.cos(coodx * div_term)-torch.sin(coody * div_term)
        pe[:, 1::4] = torch.sin(coodx * div_term)+torch.cos(coody * div_term)
        pe[:, 2::4] = torch.cos(coody * div_term)-torch.sin(coodx * div_term)
        pe[:, 3::4] = torch.sin(coody * div_term)+torch.cos(coodx * div_term)

        return pe

    def time_prj(self, t):
        """
        Sinusoidal position encoding
        Reference: https://github.com/hyunwoongko/transformer/blob/master/models/embedding/positional_encoding.py
        """
        b = t.shape[0]
        device = self.device
        d_model = self.d_model
        # Compute the positional encodings once in log space.
        te = torch.zeros(b, d_model, device=device).float()
        te.require_grad = False

        _2i = torch.arange(0,  d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        div_term = 10 ** (-_2i / d_model) * math.pi / 2
        
        position = torch.Tensor(t).float().unsqueeze(1)

        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)

        return te
    
    def get_pe(self, coods):
        """
        Get the position embedding (pe) and time embedding (te)
        """
        coodx = coods[:, 0]
        coody = coods[:, 1]
        coodt = coods[:, 2]
        pe = self.position_prj(coodx, coody)
        te = self.time_prj(coodt)
        return te, pe
    
    def get_bs_lamda(self, unknow_coods):
        batch_lamda = self.lamda.cpu().clone().detach().numpy()
        unknow_coods = unknow_coods.cpu().clone().detach().numpy()
        return batch_lamda, unknow_coods
    
    def forward(self, input_coods, input_features, input_te, input_pe
                    , know_coods, know_features, know_te, know_pe):
        """ 
        Forward process of STKrigingNet

        ----------
        Parameters
        input_coods: Tensor with shape [batch_size, 3]
        input_features: Tensor with shape [batch_size, d_input]
        input_te: Tensor with shape [batch_size, d_model]
        input_pe: Tensor with shape [batch_size, d_model]
        know_coods: Tensor with shape [know_num, 3]
        know_features: Tensor with shape [know_num, d_input]
        know_te: Tensor with shape [know_num, d_model]
        know_pe: Tensor with shape [know_num, d_model]

        -------
        Returns
        output: Tensor with shape [batch_size, 1]
        cov_know: Tensor with shape [know_num, know_num]
        """

        ##### attribute embedding #####
        ae_unknow = self.AttPrj(input_features)  # attribute embeddings of unknown points
        ae_know = self.AttPrj(know_features)  # attribute embeddings of known points

        ##### spatial cov #####
        spatial_cov_unknow = self.Scan(ae_unknow, ae_know, input_pe, know_pe)  # covariance matrix of unknown points
        spatial_cov_know = self.Scan(ae_know, ae_know, know_pe, know_pe)  # covariance matrix of known points
        
        ##### temporal cov #####
        temporal_cov_unknow = self.Tcan(ae_unknow, ae_know, input_te, know_te)  # temporal covariance of unknown points
        temporal_cov_know = self.Tcan(ae_know, ae_know, know_te, know_te)  # temporal covariance of known points
        
        ##### spatiotemporal cov #####
        cov_unknow = self.STCovNet(spatial_cov_unknow, temporal_cov_unknow)  # spatial covariance of unknown points
        cov_know = self.STCovNet(spatial_cov_know, temporal_cov_know)  # spatial covariance of known points
        
        # Zero the diagonal of the matrix cov_know
        cov_know = cov_know - torch.diag_embed(torch.diag(cov_know))
        
        ##### STTN #####
        if self.d_trend < 1:
            trend_know = torch.ones((cov_know.shape[0], 1)).to(self.device)
            trend_unknow = torch.ones((cov_unknow.shape[0], 1)).to(self.device)
        else:
            trend_know = self.STTN(know_coods, know_pe, know_te)  # trend matrix of known points
            trend_unknow = self.STTN(input_coods, input_pe, input_te)  # trend matrix of unknown points

        ##### spatiotemporal kriging decoder #####
        known_z = know_features[: , -1]
        output, trend_know_pre, trend_unknow_pre = self.STK_decoder(known_z, cov_know, cov_unknow, trend_unknow, trend_know)

        return output, [cov_know, trend_know_pre, trend_unknow_pre]

class STCovNet(nn.Module):
    def __init__(self, known_num, d_model):
        """
        The initializer of spatiotemporal covariance network (block).

        ----------
        Parameters
        known_num: int
            The number of known points (observed points).
        d_model: int
            The dimension of hidden units of each dense layer.
        """
        super(STCovNet, self).__init__()
        self.d_model = d_model
        self.dense1 = Dense(known_num*2, d_model)
        self.dense2 = Dense(d_model, d_model)
        self.dense3 = Dense(d_model, 3)
        self.relu = nn.PReLU()

    def forward(self, temporal_cov, spatial_cov):
        """
        Forward process of STCovNet

        ----------
        Parameters
        temporal_cov: Tensor with shape [batch, 1, know_num] or [batch, know_num, know_num]
        spatial_cov: Tensor with shape [batch, 1, know_num] or [batch, know_num, know_num]
        -------
        Returns
        cov: Tensor with shape [batch, 1, know_num] or [batch, know_num, know_num]
        """

        x = torch.cat([spatial_cov, temporal_cov], -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = F.softmax(x, dim=-1)
        cov = x[:,0].unsqueeze(-1)*spatial_cov + x[:,1].unsqueeze(-1)*temporal_cov + x[:,2].unsqueeze(-1)*spatial_cov*temporal_cov

        return cov


class AttNet(nn.Module):
    def __init__(self, d_model, weight):
        """
        The initializer attention network.

        ----------
        Parameters
        d_model: int
            The dimension of hidden units of each dense layer.
        weight: float
            The weight of position embedding and time embedding.
        """
        super(AttNet, self).__init__()
        self.d_model = d_model
        self.d_q = d_model
        self.d_k = d_model
        self.map_query = nn.Linear(self.d_model, self.d_q, bias=False)
        self.map_key = nn.Linear(self.d_model, self.d_k, bias=False)

        self.weight= nn.Parameter(torch.Tensor([weight]), requires_grad=False)

    def forward(self, ae_q, ae_kv, pe_q, pe_kv):
        """ 
        Forward process of AttNet

        ----------
        Parameters
        ae_q: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        ae_kv: Tensor with shape [know_num, d_model]
        pe_q: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        pe_kv: Tensor with shape [know_num, d_model]

        -------
        Returns
        att_scores: Tensor with shape [batch_size, know_num] or [know_num, know_num]
        """
        pe_q1 = pe_q * self.weight
        pe_kv1 = pe_kv * self.weight
        ae_q = ae_q * torch.abs(1-self.weight)
        ae_kv = ae_kv * torch.abs(1-self.weight)

        residual = ae_q + pe_q1
        query = self.map_query(ae_q + pe_q1) + residual

        residual = ae_kv + pe_kv1
        key = self.map_key(ae_kv + pe_kv1) + residual

        ##### cal attention score #####
        att_scores = torch.matmul(query, key.permute(1, 0)) / math.sqrt(self.d_model)
        att_scores = att_scores.squeeze()

        return att_scores
    

class STTrendNet(nn.Module):
    def __init__(self, d_model, d_trend):
        """
        The initializer of spatiotemporal trend network.

        ----------
        Parameters
        d_model: int
            The dimension of hidden units of each dense layer.
        d_trend: int
            The dimension of trend vector.
        """
        super(STTrendNet, self).__init__()
        self.d_model = d_model
        self.d_trend = d_trend
        self.d_pe = d_model
        self.relu = nn.PReLU()

        self.Sml1_w = Dense(self.d_pe, d_model * 2)
        self.Sml1_b = Dense(self.d_pe, d_model)
        self.Sml2_w = Dense(self.d_pe, d_model * d_model)
        self.Sml2_b = Dense(self.d_pe, d_model)
        self.Sml3_w = Dense(self.d_pe, d_model * d_trend)
        self.Sml3_b = Dense(self.d_pe, d_trend)

        self.Tml1_w = Dense(self.d_pe, d_model * 1)
        self.Tml1_b = Dense(self.d_pe, d_model)
        self.Tml2_w = Dense(self.d_pe, d_model * d_model)
        self.Tml2_b = Dense(self.d_pe, d_model)
        self.Tml3_w = Dense(self.d_pe, d_model * d_trend)
        self.Tml3_b = Dense(self.d_pe, d_trend)
        self.ln = nn.LayerNorm(d_trend)
        
    def forward(self, coods, pe, te):
        """ 
        Forward process of STTrendNet

        ----------
        Parameters
        coods: Tensor with shape [batch_size, 1, 3] or [batch_size, known_num, 3]
        pe: Tensor with shape [batch_size, 1, d_pe] or [batch_size, known_num, d_pe]
        te: Tensor with shape [batch_size, 1, d_pe] or [batch_size, known_num, d_pe]

        -------
        Returns
        output: Tensor with shape [batch_size, 1, d_trend] or [batch_size, known_num, d_trend]
        """
        batch_size, d = pe.shape
        
        # Spatial layer1
        sw1 = self.Sml1_w(pe).reshape(batch_size, 2, self.d_model)
        sb1 = self.Sml1_b(pe).reshape(batch_size, 1, self.d_model)
        x = torch.bmm(coods[:, :-1].unsqueeze(1), sw1) + sb1  
        x = self.relu(x)

        # Spatial layer2
        sw2 = self.Sml2_w(pe).reshape(batch_size, self.d_model, self.d_model)
        sb2 = self.Sml2_b(pe).reshape(batch_size, 1, self.d_model)
        x = torch.bmm(x, sw2) + sb2
        x = self.relu(x)

        # Spatial layer3
        sw3 = self.Sml3_w(pe).reshape(batch_size, self.d_model, self.d_trend)
        sb3 = self.Sml3_b(pe).reshape(batch_size, 1, self.d_trend)
        x = torch.bmm(x, sw3) + sb3
        x = x.squeeze()

        # Temporal layer1
        tw1 = self.Tml1_w(te).reshape(batch_size, 1, self.d_model)
        tb1 = self.Tml1_b(te).reshape(batch_size, 1, self.d_model)
        y = torch.bmm(coods[:, -1].reshape(-1, 1, 1), tw1) + tb1
        y = self.relu(y)

        # Temporal layer2
        tw2 = self.Tml2_w(te).reshape(batch_size, self.d_model, self.d_model)
        tb2 = self.Tml2_b(te).reshape(batch_size, 1, self.d_model)
        y = torch.bmm(y, tw2) + tb2
        y = self.relu(y)

        # Temporal layer3
        tw3 = self.Tml3_w(te).reshape(batch_size, self.d_model, self.d_trend)
        tb3 = self.Tml3_b(te).reshape(batch_size, 1, self.d_trend)
        y = torch.bmm(y, tw3) + tb3
        y = y.squeeze()

        output = x * y
        output = output/te.shape[-1]

        return output
    
        
class Attribute_prj(nn.Module):
    def __init__(self, d_input, d_model):
        """
        The initializer of Attribute projector.

        ----------
        Parameters
        d_input: int
            The dimension of input features.
        d_model: int
            The dimension of embedding space.
        """
        super(Attribute_prj, self).__init__()
        self.dense1 = Dense(d_input, d_model)
        self.dense2 = Dense(d_model, d_model)
        self.dense3 = Dense(d_model, d_model)
        self.relu = nn.PReLU()

    def forward(self, input_features):
        """ 
        Forward process of Attribute_prj

        ----------
        Parameters
        input_features: Tensor with shape [batch_size, d_input] or [know_num, d_input]

        -------
        Returns
        output: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        """
        x = self.dense1(input_features)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        output = self.dense3(x)
        return output


class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Dense, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc = nn.Linear(self.in_feature, self.out_feature)

    def forward(self, x):
        x = self.fc(x)
        return x