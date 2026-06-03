"""
医学图像分割评估指标
包括Dice、IoU、Hausdorff距离等常用指标
"""
import numpy as np

# 可选导入scipy
try:
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算Dice相似系数
    
    Args:
        pred: 预测分割掩膜 (H, W)
        target: 真实分割掩膜 (H, W)
        smooth: 平滑项，防止除零
    
    Returns:
        Dice系数 [0, 1]
    """
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    
    return (2.0 * intersection + smooth) / (union + smooth)


def iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算IoU（交并比）
    
    Args:
        pred: 预测分割掩膜 (H, W)
        target: 真实分割掩膜 (H, W)
        smooth: 平滑项，防止除零
    
    Returns:
        IoU [0, 1]
    """
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    return (intersection + smooth) / (union + smooth)


def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算灵敏度（召回率）
    
    Args:
        pred: 预测分割掩膜 (H, W)
        target: 真实分割掩膜 (H, W)
    
    Returns:
        灵敏度 [0, 1]
    """
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    true_positive = np.sum(pred * target)
    total_positive = np.sum(target)
    
    if total_positive == 0:
        return 1.0
    
    return true_positive / total_positive


def precision(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算精确率
    
    Args:
        pred: 预测分割掩膜 (H, W)
        target: 真实分割掩膜 (H, W)
    
    Returns:
        精确率 [0, 1]
    """
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    true_positive = np.sum(pred * target)
    total_predicted = np.sum(pred)
    
    if total_predicted == 0:
        return 1.0
    
    return true_positive / total_predicted


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算Hausdorff距离
    
    Args:
        pred: 预测分割掩膜 (H, W)
        target: 真实分割掩膜 (H, W)
    
    Returns:
        Hausdorff距离
    """
    if not SCIPY_AVAILABLE:
        # 不依赖scipy的简化实现
        pred = (pred > 0.5).astype(np.float32)
        target = (target > 0.5).astype(np.float32)
        
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return 0.0
        
        pred_coords = np.argwhere(pred)
        target_coords = np.argwhere(target)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0
        
        # 计算最大最小距离（简化版）
        max_dist = 0.0
        for pc in pred_coords:
            min_dist = np.min(np.sqrt(np.sum((target_coords - pc)**2, axis=1)))
            max_dist = max(max_dist, min_dist)
        
        for tc in target_coords:
            min_dist = np.min(np.sqrt(np.sum((pred_coords - tc)**2, axis=1)))
            max_dist = max(max_dist, min_dist)
        
        return max_dist
    
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 0.0
    
    return max(
        directed_hausdorff(pred, target)[0],
        directed_hausdorff(target, pred)[0]
    )


def evaluate_multi_class(pred: np.ndarray, target: np.ndarray, num_classes: int = 5) -> dict:
    """
    评估多类别分割结果
    
    Args:
        pred: 预测分割结果 (H, W) 或 (C, H, W)
        target: 真实分割标签 (H, W) 或 (C, H, W)
        num_classes: 类别数
    
    Returns:
        包含各类别指标的字典
    """
    results = {
        'dice': [],
        'iou': [],
        'sensitivity': [],
        'precision': [],
        'hausdorff': []
    }
    
    if len(pred.shape) == 3:
        pred = pred.argmax(axis=0)
    if len(target.shape) == 3:
        target = target.argmax(axis=0)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        if np.sum(target_cls) == 0:
            results['dice'].append(1.0 if np.sum(pred_cls) == 0 else 0.0)
            results['iou'].append(1.0 if np.sum(pred_cls) == 0 else 0.0)
            results['sensitivity'].append(1.0)
            results['precision'].append(1.0 if np.sum(pred_cls) == 0 else 0.0)
            results['hausdorff'].append(0.0)
        else:
            results['dice'].append(dice_coefficient(pred_cls, target_cls))
            results['iou'].append(iou(pred_cls, target_cls))
            results['sensitivity'].append(sensitivity(pred_cls, target_cls))
            results['precision'].append(precision(pred_cls, target_cls))
            results['hausdorff'].append(hausdorff_distance(pred_cls, target_cls))
    
    # 计算平均指标
    results['mean_dice'] = np.mean(results['dice'])
    results['mean_iou'] = np.mean(results['iou'])
    results['mean_sensitivity'] = np.mean(results['sensitivity'])
    results['mean_precision'] = np.mean(results['precision'])
    results['mean_hausdorff'] = np.mean(results['hausdorff'])
    
    return results
