# 文件名: imf_model.py
# (主要修改点: FrameDecoder)

import torch
import torch.nn as nn
import argparse

# --- 核心依赖导入 ---
from utils.modules import (
    DownConvResBlock, ResBlock, UpConvResBlock, ConvResBlock, 
)
# 导入所有注意力模块，包括新增的 SwinTransformerBlock
from utils.attention_modules import AttentionLayerFactory, ThreeCrossAttentionLayerFactory
from utils.lia_resblocks import StyledConv,EqualConv2d,EqualLinear
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

import torch
import torch.nn as nn
import time
from collections import defaultdict
import numpy as np

# ============================================================================
# Part 1: 新增的模块计时工具类
# ============================================================================
class ModuleTimer:
    """
    一个使用 PyTorch Hooks 来为 nn.Module 计时的工具类。
    它不修改模型接口，并且能精确测量 CUDA 上的执行时间。
    """
    def __init__(self, modules_to_time, device):
        self.modules_to_time = modules_to_time
        self.device = device
        self.timings = defaultdict(list)
        self.start_events = {}

        self._register_hooks()

    def _start_timing(self, name):
        """记录开始时间点。"""
        if self.device.type == 'cuda':
            # 确保之前的CUDA操作完成
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.start_events[name] = start_event
        else:
            self.start_events[name] = time.time()

    def _stop_timing(self, name):
        """记录结束时间点并计算耗时。"""
        if name not in self.start_events:
            return  # 如果没有开始事件，则跳过

        start_event_or_time = self.start_events[name]

        if self.device.type == 'cuda':
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            # 确保结束事件也完成
            torch.cuda.synchronize()
            # elapsed_time 返回毫秒 (ms)
            duration_ms = start_event_or_time.elapsed_time(end_event)
            self.timings[name].append(duration_ms)
        else:
            end_time = time.time()
            duration_ms = (end_time - start_event_or_time) * 1000
            self.timings[name].append(duration_ms)

        # 清理已使用的事件
        del self.start_events[name]

    def _make_pre_hook(self, name):
        """创建一个前向传播前的钩子函数。"""
        def pre_hook(module, input):
            self._start_timing(name)
        return pre_hook

    def _make_forward_hook(self, name):
        """创建一个前向传播后的钩子函数。"""
        def forward_hook(module, input, output):
            self._stop_timing(name)
        return forward_hook

    def _register_hooks(self):
        """为所有指定的模块注册钩子。"""
        for name, module in self.modules_to_time.items():
            if isinstance(module, nn.ModuleList) or isinstance(module, list):
                 # 如果是列表，为列表中的每个子模块注册钩子
                for i, sub_module in enumerate(module):
                    sub_name = f"{name}_{i}"
                    sub_module.register_forward_pre_hook(self._make_pre_hook(sub_name))
                    sub_module.register_forward_hook(self._make_forward_hook(sub_name))
            else:
                 module.register_forward_pre_hook(self._make_pre_hook(name))
                 module.register_forward_hook(self._make_forward_hook(name))

    def reset(self):
        """重置计时器，清除所有记录的时间。"""
        self.timings.clear()

    def summary(self):
        """打印计时结果的摘要表格。"""
        if not self.timings:
            print("No timings recorded.")
            return

        print("\n" + "="*80)
        print("📊 平均运行速度分析 (ms)")
        print("="*80)
        print(f"{'模块名':<40} {'平均耗时 (ms)':<20} {'总耗时 (ms)':<20}")
        print("-"*80)

        # 用于聚合 alignment 的总时间
        alignment_times = []
        
        for name, times in sorted(self.timings.items()):
            if "alignment" in name:
                alignment_times.extend(times)

            avg_time = np.mean(times)
            total_time = np.sum(times)
            
            # 只打印非聚合的结果
            if "alignment_" in name:
                print(f"  - {name:<36} {avg_time:<20.3f} {total_time:<20.3f}")
            elif "alignment" not in name:
                 print(f"{name:<40} {avg_time:<20.3f} {total_time:<20.3f}")

        # 计算并打印聚合的 alignment 时间
        if alignment_times:
            avg_alignment_time = np.mean(alignment_times) * len(self.modules_to_time['alignment'])
            total_alignment_time = np.sum(alignment_times)
            print("-"*80)
            print(f"{'alignment (聚合)':<40} {avg_alignment_time:<20.3f} {total_alignment_time:<20.3f}")

        print("="*80)
# ============================================================================
# Part 3: 训练显存测试主执行块
# ============================================================================
def main():
    # --- 1. 参数配置 ---
    parser = argparse.ArgumentParser(description="IMFModel Training Memory Test")
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent tokens.')
    # 阈值设为 16*16=256，这样 8x8 和 16x16 层会用标准注意力
    parser.add_argument('--swin_res_threshold', type=int, default=128, help='Resolution threshold to switch to Swin Attention.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Attention.')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Stochastic depth rate for Swin.')
    parser.add_argument('--low_res_depth', type=int, default=2, help='Number of TransformerBlocks for low-res features.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size.')
    
    args, _ = parser.parse_known_args()

    # --- 2. 模型初始化 (保持不变) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMFModel(args).to(device)
    model.eval()

    # --- 3. 定义要计时的模块 (保持不变) ---
    modules_to_time = {
        'dense_feature_encoder': model.dense_feature_encoder,
        'latent_token_encoder': model.latent_token_encoder,
        'latent_token_decoder': model.latent_token_decoder,
        'alignment': model.implicit_motion_alignment, # 这是一个 ModuleList
        'frame_decoder': model.frame_decoder,

    }

    # --- 4. 初始化计时器并注册钩子 (保持不变) ---
    print("\n" + "="*50)
    print("⏱️  正在初始化模块计时器并注册钩子...")
    timer = ModuleTimer(modules_to_time, device)
    print("✅ 钩子注册完成!")
    print("="*50)

    # --- 5. 准备输入数据 (保持不变) ---
    batch_size = 1
    img_size = 256
    x_current = torch.randn(batch_size, 3, img_size, img_size).to(device)
    x_reference = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # --- 6. 运行速度测试 ---
    with torch.no_grad():
        # --- 模型预热 ---
        print("\n" + "-"*50)
        print("正在预热模型 (运行10次)...")
        for _ in range(10):
            _ = model(x_current, x_reference)
        print("预热完成。")
        print("-"*50)

        # 清除预热期间的计时数据
        timer.reset()

        # --- 性能测试 ---
        num_runs = 1000
        print(f"\n正在精确测试 {num_runs} 次运行的模块和总速度...")
        
        # <--- 新增：为总时间计时做准备 ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_total = time.time()
        # <--- 新增结束 ---

        for _ in range(num_runs):
            _ = model(x_current, x_reference)
            
        # <--- 新增：记录总时间结束并计算 ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_total = time.time()
        
        total_elapsed_time = end_time_total - start_time_total
        avg_time_per_run_ms = (total_elapsed_time / num_runs) * 1000
        fps = num_runs / total_elapsed_time
        # <--- 新增结束 ---
            
        print("✅ 速度测试完成!")

    # --- 7. 打印模块计时结果 ---
    timer.summary()

    # --- 8. 新增：打印总时间计时结果 ---
    print("\n" + "="*80)
    print("🚀 模型总体性能 (端到端)")
    print("="*80)
    print(f"{'指标':<40} {'数值':<20}")
    print("-"*80)
    print(f"{'总耗时 (秒)':<40} {total_elapsed_time:<20.4f}")
    print(f"{'平均每次前向传播耗时 (毫秒)':<40} {avg_time_per_run_ms:<20.3f}")
    print(f"{'模型吞吐率 (FPS)':<40} {fps:<20.2f}")
    print("="*80)

    # --- (可选) 打印模型总参数 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总可训练参数: {total_params / 1e6:.2f} M")


if __name__ == "__main__":
    main()