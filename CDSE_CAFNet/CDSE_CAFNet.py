import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sam2.build_sam import build_sam2
from timm.models.layers import trunc_normal_

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CDEFM(nn.Module):
    def __init__(self, in_channels_sam, in_channels_dino, out_channels, num_heads=8):
        super(CDEFM, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        self.align_sam = nn.Conv2d(in_channels_sam, out_channels, 1)
        self.align_dino = nn.Conv2d(in_channels_dino, out_channels, 1)
        
        self.q_sam = nn.Conv2d(out_channels, out_channels, 1)
        self.k_dino = nn.Conv2d(out_channels, out_channels, 1)
        self.v_dino = nn.Conv2d(out_channels, out_channels, 1)
        self.q_dino = nn.Conv2d(out_channels, out_channels, 1)
        self.k_sam = nn.Conv2d(out_channels, out_channels, 1)
        self.v_sam = nn.Conv2d(out_channels, out_channels, 1)
        self.attn_out = nn.Conv2d(out_channels * 2, out_channels, 1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_conv1 = nn.Conv2d(out_channels, out_channels // 4, 1)
        self.context_conv2 = nn.Conv2d(out_channels // 4, out_channels, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels * 2),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x_sam, x_dino):
        if x_sam.shape[2:] != x_dino.shape[2:]:
            x_dino = F.interpolate(x_dino, size=x_sam.shape[2:], mode='bilinear', align_corners=True)
            
        B, _, H, W = x_sam.size()
        head_dim = self.out_channels // self.num_heads
        
        x_sam = self.align_sam(x_sam)
        x_dino = self.align_dino(x_dino)
        
        q_s = self.q_sam(x_sam).view(B, self.num_heads, head_dim, H * W)
        k_d = self.k_dino(x_dino).view(B, self.num_heads, head_dim, H * W)
        v_d = self.v_dino(x_dino).view(B, self.num_heads, head_dim, H * W)
        q_d = self.q_dino(x_dino).view(B, self.num_heads, head_dim, H * W)
        k_s = self.k_sam(x_sam).view(B, self.num_heads, head_dim, H * W)
        v_s = self.v_sam(x_sam).view(B, self.num_heads, head_dim, H * W)
        
        attn_sd = F.softmax((q_s @ k_d.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        attn_ds = F.softmax((q_d @ k_s.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        
        out_sd = (attn_sd @ v_d).view(B, self.out_channels, H, W)
        out_ds = (attn_ds @ v_s).view(B, self.out_channels, H, W)
        x_attn = self.attn_out(torch.cat([out_sd, out_ds], dim=1))
        
        context = self.global_pool(x_sam + x_dino)
        context = F.relu(self.context_conv1(context))
        context = torch.sigmoid(self.context_conv2(context))
        x_context = x_attn * context
        
        pool_sam = self.global_pool(x_sam).squeeze(-1).squeeze(-1)
        pool_dino = self.global_pool(x_dino).squeeze(-1).squeeze(-1)
        pool_combined = self.global_pool(x_sam + x_dino).squeeze(-1).squeeze(-1)
        weights = self.mlp(torch.cat([pool_sam, pool_dino, pool_combined], dim=1))
        w_sam, w_dino = weights[:, :self.out_channels], weights[:, self.out_channels:]
        x_fused = w_sam.view(B, self.out_channels, 1, 1) * x_sam + w_dino.view(B, self.out_channels, 1, 1) * x_dino
        
        x_out = self.norm((x_fused + x_context + x_sam + x_dino).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x_out

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        prompted = x + prompt
        return self.block(prompted)
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # 添加1x1卷积用于调整残差通道数（关键修复）
        self.residual_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2=None):
        if x2 is not None:
            # 对齐x2与x1的尺寸
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        
        # 保存残差并调整通道数（关键修复）
        residual = self.residual_adjust(x)
        # 上采样后再卷积
        x = self.up(x)
        x = self.conv(x)
        
        # 确保残差与当前特征尺寸匹配（关键修复）
        if x.shape[2:] != residual.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 残差连接
        x = x + residual
        return x

class CDSE_CAFNet(nn.Module):
    def __init__(self, checkpoint_path=None, dinov2_path=None):
        super(CDSE_CAFNet, self).__init__()

        # SAM编码器
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:  
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder, model.sam_prompt_encoder
        del model.memory_encoder, model.memory_attention, model.mask_downsample
        del model.obj_ptr_tpos_proj, model.obj_ptr_proj, model.image_encoder.neck
        self.sam = model.image_encoder.trunk
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.blocks = nn.Sequential(*[Adapter(block) for block in self.sam.blocks])

        # DINOv2编码器
        if dinov2_path:
            self.dino = timm.create_model(
                'vit_large_patch14_dinov2',
                features_only=True,
                img_size=(448, 448),
                pretrained=True,
                pretrained_cfg_overlay=dict(file=dinov2_path)
            )
        else:
            self.dino = timm.create_model(
                'vit_large_patch14_dinov2',
                features_only=True,
                img_size=(448, 448)
            )
        for param in self.dino.parameters():
            param.requires_grad = False

        # CDEFM模块
        self.CDEFM1 = CDEFM(in_channels_sam=144, in_channels_dino=1024, out_channels=128, num_heads=8)
        self.CDEFM2 = CDEFM(in_channels_sam=288, in_channels_dino=1024, out_channels=128, num_heads=8)
        self.CDEFM3 = CDEFM(in_channels_sam=576, in_channels_dino=1024, out_channels=128, num_heads=8)
        self.CDEFM4 = CDEFM(in_channels_sam=1152, in_channels_dino=1024, out_channels=128, num_heads=8)

        # 解码器（使用修复后的Up模块）
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)

        # 输出头
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        # 特征提取
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=True))[-1]
        
        # 对齐DINOv2特征尺寸
        x_d1 = F.interpolate(x_d, size=x1_s.shape[2:], mode='bilinear', align_corners=True)
        x_d2 = F.interpolate(x_d, size=x2_s.shape[2:], mode='bilinear', align_corners=True)
        x_d3 = F.interpolate(x_d, size=x3_s.shape[2:], mode='bilinear', align_corners=True)
        x_d4 = F.interpolate(x_d, size=x4_s.shape[2:], mode='bilinear', align_corners=True)

        # 特征融合
        x1 = self.CDEFM1(x1_s, x_d1)
        x2 = self.CDEFM2(x2_s, x_d2)
        x3 = self.CDEFM3(x3_s, x_d3)
        x4 = self.CDEFM4(x4_s, x_d4)

        # 解码过程
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # 输出
        seg_out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear', align_corners=True)
        return seg_out