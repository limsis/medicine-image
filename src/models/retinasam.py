"""
RetinaSAM - 眼底视网膜专用SAM模型
基于SAM架构，增加眼底专用的特征适配器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleASPP(nn.Module):
    """
    多尺度空洞空间金字塔池化模块
    用于捕获不同大小的病灶
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        dilations = [1, 6, 12, 18]
        
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        
        # 全局分支
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, (H, W), mode='bilinear', align_corners=True)
        outputs.append(global_feat)
        
        # 融合
        x = torch.cat(outputs, dim=1)
        x = self.fusion(x)
        
        return x


class SlenderStructureAttention(nn.Module):
    """
    细长结构感知注意力模块
    借鉴裂缝检测领域，用于增强血管和微动脉瘤等细长结构
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 细长结构卷积核（水平、垂直、对角）
        self.kernels = nn.Parameter(torch.tensor([
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],  # 水平
            [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],  # 垂直
            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],  # 对角1
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]   # 对角2
        ], dtype=torch.float32).view(4, 1, 3, 3), requires_grad=False)
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 对每个通道应用细长结构检测
        slender_feats = []
        for i in range(C):
            channel_feat = x[:, i:i+1, :, :]
            with torch.no_grad():
                edges = F.conv2d(channel_feat, self.kernels, padding=1)
            slender_feats.append(edges)
        
        # 融合
        slender_feat = torch.mean(torch.stack(slender_feats, dim=1), dim=1)
        attention = self.conv(slender_feat)
        
        return x * attention


class LesionContextFusion(nn.Module):
    """
    病灶上下文融合模块
    融合血管、病灶和背景的上下文信息
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 多尺度特征提取
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3 // 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_feat = self.low_conv(x)
        mid_feat = self.mid_conv(x)
        high_feat = self.high_conv(x)
        
        fused = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        out = self.fusion(fused)
        
        return out + x  # 残差连接


class FundusFeatureAdapter(nn.Module):
    """
    眼底专用特征适配器
    包含多尺度ASPP、细长结构注意力和病灶上下文融合
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.aspp = MultiScaleASPP(in_channels, out_channels)
        self.slender_attn = SlenderStructureAttention(out_channels)
        self.context_fusion = LesionContextFusion(out_channels)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.slender_attn(x)
        x = self.context_fusion(x)
        x = self.final_conv(x)
        return x


class ImprovedMaskDecoder(nn.Module):
    """
    改进的掩膜解码器
    增加边界感知分支
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        # 主分割分支
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 边界感知分支
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2, num_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_feat = self.main_branch(x)
        boundary_feat = self.boundary_branch(x)
        
        fused = torch.cat([main_feat, boundary_feat], dim=1)
        out = self.fusion(fused)
        
        return out


class RetinaSAM(nn.Module):
    """
    RetinaSAM - 眼底视网膜专用SAM模型
    """
    def __init__(
        self,
        image_encoder: nn.Module = None,
        prompt_encoder: nn.Module = None,
        mask_decoder: nn.Module = None,
        in_channels: int = 256,
        num_classes: int = 5  # 背景、微动脉瘤、出血、硬性渗出、软性渗出
    ):
        super().__init__()
        
        # 使用预训练的SAM组件或简化版本
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        
        # 眼底专用特征适配器
        self.fundus_adapter = FundusFeatureAdapter(in_channels, in_channels)
        
        # 改进的掩膜解码器
        self.mask_decoder = ImprovedMaskDecoder(in_channels, num_classes)
        
        # 如果没有提供预训练组件，创建简单的替代
        if self.image_encoder is None:
            self.image_encoder = self._create_simple_encoder()
        
        if self.prompt_encoder is None:
            self.prompt_encoder = nn.Identity()
    
    def _create_simple_encoder(self) -> nn.Module:
        """创建简单的图像编码器作为替代"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            prompts: 提示信息 (可选)
        
        Returns:
            分割掩膜 (B, num_classes, H, W)
        """
        # 图像编码
        img_feat = self.image_encoder(x)
        
        # 眼底专用特征适配
        adapted_feat = self.fundus_adapter(img_feat)
        
        # 提示编码（可选）
        if prompts is not None:
            prompt_feat = self.prompt_encoder(prompts)
            adapted_feat = adapted_feat + prompt_feat
        
        # 掩膜解码
        mask = self.mask_decoder(adapted_feat)
        
        return mask


if __name__ == '__main__':
    # 测试RetinaSAM
    model = RetinaSAM(num_classes=5)
    
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
