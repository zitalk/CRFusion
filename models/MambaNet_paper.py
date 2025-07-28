import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer
import math
def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )
class CRFusion(nn.Module):
    def __init__(self):
        super(CRFusion, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.CSCM_1 = RSFM(2048, 2048)
        self.CSCM_2 = RSFM(1024, 1024)
        self.CSCM_3 = RSFM(512, 512)
        self.CSCM_4 = RSFM(256, 256)

        self.FA_Block2 = Block(dim=256)
        self.FA_Block3 = Block(dim=128)
        self.FA_Block4 = Block(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 =  nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.dwc3 = conv3x3_bn_relu(256, 128)
        self.dwc2 = conv3x3_bn_relu(512, 256)
        self.dwc1 = conv3x3_bn_relu(1024, 512)
        self.dwcon_1 = conv3x3_bn_relu(2048, 1024)
        self.dwcon_2 = conv3x3_bn_relu(1024, 512)
        self.dwcon_3 = conv3x3_bn_relu(512, 256)
        self.dwcon_4 = conv3x3_bn_relu(256, 128)
        self.conv43 = conv3x3_bn_relu(128, 256, s=2)
        self.conv32 = conv3x3_bn_relu(256, 512, s=2)
        self.conv21 = conv3x3_bn_relu(512, 1024, s=2)


    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0] # torch.Size([1, 128, 96, 96])   
        r3 = rgb_list[1] # torch.Size([1, 256, 48, 48])
        r2 = rgb_list[2] # torch.Size([1, 512, 24, 24])
        r1 = rgb_list[3] # torch.Size([1, 1024, 12, 12])
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]


        r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=96, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(d2), size=48, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(d1), size=24, mode='bilinear')

        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        d1_con = torch.cat((d1, d1), 1)
        d1_con = self.dwcon_1(d1_con)

        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        d2_con = torch.cat((d2, d1_up), 1)
        d2_con = self.dwcon_2(d2_con)

        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)

        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)
        

        xf_1,h1 = self.CSCM_1(r1_con, d1_con)  # 1024,12,12
        xf_2,h2 = self.CSCM_2(r2_con, d2_con, h1)  # 512,24,24
        xf_3,h3 = self.CSCM_3(r3_con, d3_con, h2)  # 256,48,48
        xf_4,h4 = self.CSCM_4(r4_con, d4_con, h3)  # 128,96,96


        df_f_1 = self.deconv_layer_1(xf_1)

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.FA_Block4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1,y2,y3,y4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class RSFM(nn.Module):
    def __init__(self, inp, oup, d_model=64):
        super().__init__()
        self.conv1 = nn.Conv2d(inp//2, d_model, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inp//2, d_model, kernel_size=1, stride=1, padding=0)
        self.rgb_mamba = ResidualBlock(d_model=d_model)
        self.conv3 = nn.Conv2d(d_model, oup//2, kernel_size=1, stride=1, padding=0)

    def forward(self,  rgb_feat, dte_feat, h=None):
        rgb_feat = self.conv1(rgb_feat)
        dte_feat = self.conv2(dte_feat)
        
        out1, h1 = self.rgb_mamba(rgb_feat, h = h)
        out2, h2 = self.rgb_mamba(dte_feat+out1, h = h1)
        
        out3, h3 = self.rgb_mamba(rgb_feat+out2, h = h2)
        out4, h4 = self.rgb_mamba(dte_feat+out3, h = h3)
        
        out5, h5 = self.rgb_mamba(rgb_feat+out4, h = h4)
        out6, h6 = self.rgb_mamba(dte_feat+out5, h = h5)
        
        return self.conv3(out6), h6
    

class ResidualBlock(nn.Module):
    def __init__(self,dt_scale=1.0, d_model=512,dt_rank=32,d_state=16,bias=False,d_conv=3,conv_bias=True,dt_init='random',dt_max=0.1,dt_min=0.001,dt_init_floor=0.0001):
        super().__init__()
        d_inner = d_model*2
        self.mixer = MambaBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h=None):
        #  x : (B, C, H, W )
        #  h : (B, L, ED, N)
        #  output : (B, C, H, W )
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, L, C]
        x = self.norm(x)
        
        output, h = self.mixer(x, h)
        output = output + x
        # 恢复形状 [B, C, H, W]
        B, L, C_out = output.shape
        H_out = W_out = int(L ** 0.5)
        assert H_out * W_out == L, f"L={L} 不是一个正方形"
        output = output.transpose(1, 2).view(B, C_out, H_out, W_out)
        return output, h



class MambaBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()
        #  projects block input from D to 2*ED (two branches)
        self.dt_scale = dt_scale
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, bias=conv_bias,
                                groups=self.d_inner,
                                padding=(d_conv - 1)//2)
        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(self.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    def forward(self, x, h):
        #  x : (B, L, D)
        #  h : (B, L, ED, N)
        #  y : (B, L, D)


        xz = self.in_proj(x)  # (B, L,2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B,L, ED), (B,L, ED)
        x_cache = x.permute(0,2,1)#(B, ED,L)

        #  x branch
        x = self.conv1d( x_cache).permute(0,2,1) #  (B,L , ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        #y->B,L,ED;h->B,L,ED,N

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output, h

    def ssm_step(self, x, h):
        #  x : (B, L, ED)
        #  h : (B, L, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],
                                  dim=-1)  #  (B, L,dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B,L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B,L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L,ED, N)

        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.d_inner, self.d_state, device=deltaA.device)  #  (B, L, ED, N)

        ###################
        # 上采样h
        if h is not None and h.size(1) != x.size(1):
            #  h : (B, L, ED, N)
            h = F.interpolate(h.permute(0, 3, 1, 2), size=(x.size(1), self.d_inner), mode='bilinear').permute(0, 2, 3, 1)

        ###################

        h = deltaA * h + BX  #  (B, L, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x#B,L,ED

        #  todo : pq h.squeeze(1) ??
        return y, h


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
