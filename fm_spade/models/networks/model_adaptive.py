# 文件名: imf_model.py
# (主要修改点: FrameDecoder)

import torch
import torch.nn as nn
import argparse

# --- 核心依赖导入 ---
from models.utils.modules import (
    DownConvResBlock, ResBlock, UpConvResBlock, ConvResBlock, 
)
# 导入所有注意力模块，包括新增的 SwinTransformerBlock
from models.utils.attention_modules import AttentionLayerFactory, ThreeCrossAttentionLayerFactory
from models.utils.lia_resblocks import StyledConv,EqualConv2d,EqualLinear
# ... LatentTokenEncoder, DenseFeatureEncoder, LatentTokenDecoder 的代码保持不变 ...
class DenseFeatureEncoder(nn.Module):
    # ... (代码不变) ...
    def __init__(self, in_channels=3, output_channels=[64, 128, 256, 512, 512, 512], initial_channels=32, dm=512):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        for out_channels in output_channels:
            if out_channels==32:continue
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

        # Equal convolution and linear layers
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        # Apply linear layers
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        # Final linear layer
        x = self.final_linear(x)
        return features[::-1], x

class LatentTokenEncoder(nn.Module):
    def __init__(self, initial_channels=64, output_channels=[64, 128, 256, 512, 512, 512], dm=32):
        super(LatentTokenEncoder, self).__init__()

        # Initial convolution followed by LeakyReLU activation
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        # Dynamically create ResBlocks
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        # Equal convolution and linear layers
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)

    def forward(self, x):
        # Initial convolution and activation
        x = self.activation(self.conv1(x))
        
        # Apply ResBlocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply equalconv
        x = self.equalconv(x)
        
        # Global average pooling
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        
        # Apply linear layers
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
            
        
        # Final linear layer
        x = self.final_linear(x)
        
        return x

class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, const_dim=32):
        super().__init__()
        # Constant input for the decoder
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        # StyleConv layers
        self.style_conv_layers = nn.ModuleList([
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 128, 3, latent_dim, upsample=True),
            StyledConv(128, 128, 3, latent_dim),
            StyledConv(128, 128, 3, latent_dim)  
        ])

    def forward(self, t):
        # Repeat constant input for batch size
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        #import pdb;pdb.set_trace()
        # Store feature maps
        m1, m2, m3, m4 = None, None, None, None
        # Apply style convolution layers
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            
            if i == 3:
                m1 = x
            elif i == 6:
                m2 = x
            elif i == 9:
                m3 = x
            elif i == 12:
                m4 = x
        
        # Return the feature maps in reverse order
        return m1, m2, m3, m4

class IdTokenDecoder(nn.Module):
    def __init__(self, latent_dim=512, const_dim=512):
        super().__init__()
        # Constant input for the decoder
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        # StyleConv layers
        self.style_conv_layers = nn.ModuleList([
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 128, 3, latent_dim, upsample=True),
            StyledConv(128, 128, 3, latent_dim),
            StyledConv(128, 128, 3, latent_dim)  
        ])

    def forward(self, t):
        # Repeat constant input for batch size
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        #import pdb;pdb.set_trace()
        # Store feature maps
        m1, m2, m3, m4 = None, None, None, None
        # Apply style convolution layers
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            
            if i == 3:
                m1 = x
            elif i == 6:
                m2 = x
            elif i == 9:
                m3 = x
            elif i == 12:
                m4 = x
        
        # Return the feature maps in reverse order
        return m1, m2, m3, m4

# ============================================================================
# 主模型 (最终简化版)
# ============================================================================
class FrameDecoder(nn.Module):
    def __init__(self, args, feature_dims, spatial_dims):
        super().__init__()
        self.args = args
        
        feature_dims_rev = feature_dims[::-1]
        spatial_dims_rev = spatial_dims[::-1]

        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(feature_dims_rev[i], feature_dims_rev[i+1]) for i in range(len(feature_dims_rev) - 1)
        ])
        self.resblocks = nn.ModuleList([
            ConvResBlock(feature_dims_rev[i+1]*2, feature_dims_rev[i+1]) for i in range(len(feature_dims_rev) - 1)
        ])
        
        self.transformer_blocks = nn.ModuleList()
        print("🔧 正在通过工厂构建解码器中的统一自注意力层:")
        for i in range(len(spatial_dims_rev) - 1):
            s_dim = spatial_dims_rev[i+1]
            f_dim = feature_dims_rev[i+1]
            self.transformer_blocks.append(
                AttentionLayerFactory(args=args, dim=f_dim, resolution=(s_dim, s_dim))
            )

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dims_rev[-1], 3*4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Sigmoid()
        )

    def forward(self, features_align):
        x = features_align[0]
        #import pdb;pdb.set_trace()
        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x)
            x = torch.cat([x, features_align[i + 1]], dim=1)
            x = self.resblocks[i](x)
            x = self.transformer_blocks[i](x)
        return self.final_conv(x)

class IdAdaptive(nn.Module):
    def __init__(self, dim_mot=32, dim_app=512, depth=4):
        super().__init__()
        self.in_layer = EqualLinear(dim_app+dim_mot, dim_app)
        self.linear_layers = nn.ModuleList([EqualLinear(dim_app, dim_app) for _ in range(depth)])
        self.final_linear = EqualLinear(dim_app, dim_mot)  # 杈撳嚭 shift + scale
        self.activation = nn.LeakyReLU(0.2)
        self.scale_activation = nn.Sigmoid()  # 闄愬埗 scale 鍦?[0,1]

    def modulate(self, x, shift, scale) -> torch.Tensor:
        # x: (B, dim_mot)
        # shift/scale: (B, dim_mot)
        return x * scale + shift

    def forward(self, mot, app):
        """
        mot: (B, dim_mot)
        app: (B, dim_app)
        """
        x = torch.cat((mot, app), dim=-1)
        x = self.in_layer(x)
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        # 鏈€鍚庝竴灞傝緭鍑?shift 鍜?scale
        out = self.final_linear(x)
        #scale = self.scale_activation(scale)  # 闄愬埗 scale
        #out = self.modulate(mot, shift, scale)
        return out

class IMFModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_dims = [32, 64, 128, 256, 512, 512]
        self.motion_dims = self.feature_dims
        self.spatial_dims = [256, 128, 64, 32, 16, 8]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels=self.feature_dims)
        self.latent_token_encoder = LatentTokenEncoder(initial_channels=64, output_channels=[128, 256, 512, 512, 512])
        self.latent_token_decoder = LatentTokenDecoder()
        #self.id_token_decoder = IdTokenDecoder()
        
        self.frame_decoder = FrameDecoder(args, self.feature_dims, self.spatial_dims)

        self.implicit_motion_alignment = nn.ModuleList()
        print("🔧 正在通过工厂构建对齐阶段的统一交叉注意力层:")
        for dim, s_dim in zip(self.feature_dims[::-1], self.spatial_dims[::-1]):
            self.implicit_motion_alignment.append(
                ThreeCrossAttentionLayerFactory(args=args, dim=dim, resolution=(s_dim, s_dim))
            )
        self.adapt = IdAdaptive()

    def decode(self, A, B, C):
        num_levels = len(self.spatial_dims)
        aligned_features = [None] * num_levels
        attention_map = None # 初始化 attention_map 为 None
        for i in range(num_levels):
            attention_block = self.implicit_motion_alignment[i]
            if attention_block.is_standard_attention:
                aligned_feature, attention_map = attention_block.coarse_warp(A[i], B[i], C[i])
                aligned_features[i] = aligned_feature
            else:
                aligned_feature = attention_block.fine_warp(C[i], attn=attention_map)
                aligned_features[i] = aligned_feature
        output_frame = self.frame_decoder(aligned_features)
        return output_frame
    

    def app_encode(self, x):
        f_r, id = self.dense_feature_encoder(x)
        return f_r, id
    
    def mot_encode(self, x):
        mot_latent = self.latent_token_encoder(x)
        return mot_latent
    
    def mot_decode(self, x):
        mot_map = self.latent_token_decoder(x)
        return mot_map
    
    def id_adapt(self, t, id):
        return self.adapt(t, id)
    
    def forward(self, x_current, x_reference):
        f_r, i_r = self.app_encode(x_reference)
        t_r = self.mot_encode(x_reference)
        t_c = self.mot_encode(x_current)
        ta_r = self.adapt(t_r, i_r)
        ta_c = self.adapt(t_c, i_r)
        ma_r = self.mot_decode(ta_r)
        ma_c = self.mot_decode(ta_c)
        output_frame = self.decode(ma_c, ma_r, f_r)
        return output_frame