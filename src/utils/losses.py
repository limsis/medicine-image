"""
医学图像分割损失函数
包括Dice损失、边界感知损失、骨架加权损失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 可选导入scipy
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DiceLoss(nn.Module):
    """
    Dice损失函数
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测结果 (B, C, H, W) 或 (B, H, W)
            target: 真实标签 (B, C, H, W) 或 (B, H, W)
        
        Returns:
            Dice损失
        """
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        if len(target.shape) == 3:
            if pred.shape[1] > 1:
                target = F.one_hot(target.long(), pred.shape[1]).permute(0, 3, 1, 2).float()
            else:
                target = target.unsqueeze(1).float()
        
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - torch.mean(dice)


class FocalLoss(nn.Module):
    """
    Focal损失函数，用于处理类别不平衡
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        pred = torch.sigmoid(pred)
        
        bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
        
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return torch.mean(focal_loss)


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失函数
    使用Sobel算子提取边界
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def get_boundary(self, x: torch.Tensor) -> torch.Tensor:
        # 转换为float类型
        x = x.float()
        
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # 如果输入是多通道，对每个通道分别处理
        if x.shape[1] > 1:
            edges = []
            for i in range(x.shape[1]):
                channel = x[:, i:i+1, :, :]
                edge_x = F.conv2d(channel, self.sobel_x, padding=1)
                edge_y = F.conv2d(channel, self.sobel_y, padding=1)
                edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
                edges.append(edge)
            edge = torch.cat(edges, dim=1)
        else:
            edge_x = F.conv2d(x, self.sobel_x, padding=1)
            edge_y = F.conv2d(x, self.sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        edge = edge / (edge.max() + 1e-8)
        
        return edge
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        pred_boundary = self.get_boundary(pred)
        
        # 处理target可能是单通道的情况
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 如果pred和target通道数不同，对target进行one-hot编码
        if pred.shape[1] != target.shape[1]:
            target = F.one_hot(target.squeeze(1).long(), pred.shape[1]).permute(0, 3, 1, 2).float()
        
        target_boundary = self.get_boundary(target)
        
        # 如果通道数仍然不同，取平均值
        if pred_boundary.shape[1] != target_boundary.shape[1]:
            pred_boundary = pred_boundary.mean(dim=1, keepdim=True)
            target_boundary = target_boundary.mean(dim=1, keepdim=True)
        
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return self.weight * boundary_loss


class SkeletonWeightedLoss(nn.Module):
    """
    骨架加权损失函数
    借鉴骨架检测领域，对细长结构给予更高权重
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def get_skeleton_weight(self, target: torch.Tensor) -> torch.Tensor:
        """获取骨架权重"""
        if not SCIPY_AVAILABLE:
            # 不使用scipy的简化版本，直接返回全1权重
            return torch.ones_like(target)
        
        target_np = target.cpu().numpy()
        
        weights = []
        for i in range(target_np.shape[0]):
            dist = ndimage.distance_transform_edt(target_np[i] > 0.5)
            if dist.max() > 0:
                dist = dist / dist.max()
            weight = 1.0 + dist
            weights.append(weight)
        
        weights = np.stack(weights, axis=0)
        return torch.from_numpy(weights).float().to(target.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        # 处理target可能是单通道的情况
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        # 如果pred和target通道数不同，对target进行one-hot编码
        if pred.shape[1] != target.shape[1]:
            target = F.one_hot(target.squeeze(1).long(), pred.shape[1]).permute(0, 3, 1, 2).float()
        
        skeleton_weight = self.get_skeleton_weight(target)
        
        # 确保skeleton_weight的形状与pred匹配
        if skeleton_weight.dim() == 3:
            skeleton_weight = skeleton_weight.unsqueeze(1)
        
        # 如果通道数不匹配，扩展skeleton_weight
        if skeleton_weight.shape[1] == 1 and pred.shape[1] > 1:
            skeleton_weight = skeleton_weight.repeat(1, pred.shape[1], 1, 1)
        
        bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
        weighted_bce = bce * skeleton_weight
        
        return self.weight * torch.mean(weighted_bce)


class RetinaSAMLoss(nn.Module):
    """
    RetinaSAM的总损失函数
    结合Dice损失、边界感知损失、骨架加权损失和分类损失
    """
    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_skeleton: float = 0.3,
        lambda_classification: float = 0.2
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary
        self.lambda_skeleton = lambda_skeleton
        self.lambda_classification = lambda_classification
        
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryAwareLoss()
        self.skeleton_loss = SkeletonWeightedLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        pred_class: torch.Tensor = None,
        target_class: torch.Tensor = None
    ) -> torch.Tensor:
        loss_dice = self.dice_loss(pred_mask, target_mask)
        loss_boundary = self.boundary_loss(pred_mask, target_mask)
        loss_skeleton = self.skeleton_loss(pred_mask, target_mask)
        
        total_loss = (
            self.lambda_dice * loss_dice +
            self.lambda_boundary * loss_boundary +
            self.lambda_skeleton * loss_skeleton
        )
        
        if pred_class is not None and target_class is not None:
            loss_class = self.ce_loss(pred_class, target_class)
            total_loss += self.lambda_classification * loss_class
        
        return total_loss


class EnhanceSegLoss(nn.Module):
    """
    EnhanceSegNet的总损失函数
    结合增强损失和分割损失
    """
    def __init__(
        self,
        lambda_enhance: float = 0.3,
        lambda_segment: float = 1.0,
        lambda_adversarial: float = 0.1
    ):
        super().__init__()
        self.lambda_enhance = lambda_enhance
        self.lambda_segment = lambda_segment
        self.lambda_adversarial = lambda_adversarial
        
        self.segment_loss = RetinaSAMLoss()
        self.enhance_loss = nn.MSELoss()
    
    def forward(
        self,
        pred_enhanced: torch.Tensor,
        target_enhanced: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        disc_real: torch.Tensor = None,
        disc_fake: torch.Tensor = None
    ) -> torch.Tensor:
        loss_enhance = self.enhance_loss(pred_enhanced, target_enhanced)
        loss_segment = self.segment_loss(pred_mask, target_mask)
        
        total_loss = (
            self.lambda_enhance * loss_enhance +
            self.lambda_segment * loss_segment
        )
        
        if disc_real is not None and disc_fake is not None:
            loss_adv = -torch.mean(torch.log(disc_fake + 1e-8))
            total_loss += self.lambda_adversarial * loss_adv
        
        return total_loss


if __name__ == '__main__':
    batch_size = 2
    height, width = 256, 256
    
    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    dice_loss = DiceLoss()
    print(f'Dice Loss: {dice_loss(pred, target):.4f}')
    
    boundary_loss = BoundaryAwareLoss()
    print(f'Boundary Loss: {boundary_loss(pred, target):.4f}')
    
    retina_loss = RetinaSAMLoss()
    print(f'RetinaSAM Loss: {retina_loss(pred, target):.4f}')
