import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class PixelVAE(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim=32,  hidden_dim=128, act='silu'):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_channels, z_dim) 

        self.enc_net = nn.Sequential(
            BaseConv(in_channels, hidden_dim, 1, 1, act=act), 
            BaseConv(hidden_dim, hidden_dim, 1, 1, act=act), 
        )
        self.mu_conv = nn.Conv2d(hidden_dim, z_dim, kernel_size=1)
        self.logvar_conv = nn.Conv2d(hidden_dim, z_dim, kernel_size=1)

        self.dec_net = nn.Sequential(
            BaseConv(z_dim, hidden_dim, 1, 1, act=act), 
            BaseConv(hidden_dim, hidden_dim, 1, 1, act=act), 
            BaseConv(hidden_dim, out_channels, 1, 1, act=act), 
        )

    def encode(self, x):
        h = self.enc_net(x)              
        mu = self.mu_conv(h)              
        logvar = self.logvar_conv(h)      
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  

    def decode(self, z):
        return self.dec_net(z) 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    

################################################
class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        P5          = self.lateral_conv0(feat3)

        P5_upsample = self.upsample(P5)  
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4          = self.reduce_conv1(P5_upsample) 

        P4_upsample = self.upsample(P4) 
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        P3_out      = self.C3_p3(P4_upsample)  
        # return P3_out
        return P5, P4, P3_out
    

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


################################################
class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)
        return x



class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.conv = BaseConv(c2, c2, 3, 1, act="silu")

        self.alpha = nn.Parameter(torch.tensor(1.0))   
        self.beta = nn.Parameter(torch.tensor(0.0))    

        self.adaptive = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // 4, 2, 1)  
        )

    def forward(self, x):
        b, c, h, w = x.shape

        stats = self.adaptive(x).view(b, 2)
        dyn_alpha = F.softplus(stats[:, 0:1]) + self.alpha  
        dyn_beta = stats[:, 1:2] + self.beta

        x_flat = x.view(b, c, -1).transpose(1, 2).contiguous()  # (B, N, C)
        feature = x_flat.clone()
        distance = torch.cdist(feature, feature)  # (B, N, N)

        hg = torch.sigmoid(-dyn_alpha.view(b, 1, 1) * distance + dyn_beta.view(b, 1, 1))

        x_flat = self.hgconv(x_flat, hg) + x_flat

        x = x_flat.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))
        x = self.conv(x)
        return x
    
################################################
class Channel_Attention(nn.Module):
    def __init__(self, in_nc, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_nc, in_nc // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_nc // ratio, in_nc, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x
################################################
import torch.nn as nn
import torch
class EUGA(nn.Module):
    def __init__(self, in_channel=3, mid_channel=None, ratio=4):
        super(EUGA, self).__init__()
        self.chanel_in = in_channel

        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = BaseConv(self.chanel_in, self.chanel_in, 3, 1, act="silu")


    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        proj_query = self.query_conv(fb).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, out_channel

        proj_key = self.key_conv(fb).view(B, -1, H * W)  # B, out_channel, HW

        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW

        attention = self.softmax(energy)  # B, HW, HW
        proj_value = self.value_conv(fa).view(B, -1, H * W)  # B, in_channel, HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, in_channel, HW

        out = out.view(B, C, H, W)

        out = out + fa

        out = self.out_conv(out)
        return out

################################################
class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=10):
        super(Network, self).__init__()
        self.num_frame = num_frame
        self.backbone = Feature_Extractor(0.33, 0.50)  
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")  
        
        ################ 
        fea_ext_out_nc = 128
        act = "silu"

        self.HCM_list = nn.ModuleList([])
        for i in range(self.num_frame-1):
            self.HCM_list.append(
                nn.Sequential(
                    BaseConv(fea_ext_out_nc*2, fea_ext_out_nc, 1, 1, act=act),
                    HyperComputeModule(fea_ext_out_nc, fea_ext_out_nc)
                )
            )
        self.fuse_st = nn.Sequential(
                BaseConv(fea_ext_out_nc*self.num_frame, fea_ext_out_nc, 1, 1, act=act),
                Channel_Attention(fea_ext_out_nc, ratio=8),
                BaseConv(fea_ext_out_nc, fea_ext_out_nc, 3, 1, act=act),
            )
        ############## 
        self.mask_conv =  nn.Sequential(
            BaseConv(1, 64, 3, 2, act=act),
            BaseConv(64, 64, 3, 1, act=act),
            BaseConv(64, 64, 3, 2, act=act),
            BaseConv(64, 64, 3, 1, act=act),
            BaseConv(64, 64, 3, 2, act=act),
            BaseConv(64, 128, 3, 1, act=act),
        )
        self.EUGA = EUGA(in_channel=128, mid_channel=128)
        ############## 
        self.fusion_conv = BaseConv(fea_ext_out_nc*2, fea_ext_out_nc, 1, 1, act=act)
        self.loss_function = nn.L1Loss()

        self.vae = PixelVAE(128, 128)

    def forward(self, inputs, depths_list=None):  
        ################ 
        feats = []
        for i in range(self.num_frame):
            P5, P4, P3 = self.backbone(inputs[:, :, i, :, :])
            feats.append(P3)

        ################ 
        key_feature = feats[-1]
        align_feats = []
        for i in range(0, self.num_frame-1):
            fuse_feature = torch.cat((key_feature, feats[i]), dim=1)
            align_feat = self.HCM_list[i](fuse_feature)
            align_feats.append(align_feat)

        align_feats.append(key_feature)

        feats_ST = torch.cat(align_feats, dim=1)
        feats_ST = self.fuse_st(feats_ST)

        ############ DAM
        if self.training:
            attention = self.mask_conv(depths_list)
            feats_S = self.EUGA(feats[-1], attention)
        
        mask_hat, mu, logvar = self.vae(feats[-1]) 

        if self.training:
            recon_loss = self.loss_function(mask_hat, feats_S)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        ############### 
        feat_fusion = torch.cat((feats_ST, mask_hat), dim=1)
        feat_fusion = self.fusion_conv(feat_fusion)

        ############### 
        outputs  = self.head([feat_fusion])
        if self.training:
            return outputs, recon_loss, kl_loss
        else:
            return  outputs

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
