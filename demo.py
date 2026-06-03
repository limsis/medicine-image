
"""
医学图像技术调研 - 演示脚本
展示如何使用RetinaSAM和EnhanceSegNet
"""
import sys
print("=== 医学图像技术调研演示 ===\n")

# 检查PyTorch
try:
    import torch
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch未安装")
    sys.exit(1)

# 导入我们的模型
print("\n=== 导入模型 ===")
try:
    sys.path.append('src')
    from models.retinasam import RetinaSAM
    from models.enhancesegnet import EnhanceSegNet
    print("✅ 模型导入成功")
except Exception as e:
    print(f"❌ 模型导入失败: {e}")

# 创建并测试RetinaSAM
print("\n=== 测试RetinaSAM模型 ===")
try:
    model = RetinaSAM(num_classes=5)
    print("✅ RetinaSAM创建成功")
    
    # 测试前向传播
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"✅ 前向传播成功 - 输出形状: {output.shape}")
except Exception as e:
    print(f"❌ RetinaSAM测试失败: {e}")

# 创建并测试EnhanceSegNet
print("\n=== 测试EnhanceSegNet模型 ===")
try:
    model = EnhanceSegNet(num_classes=5)
    print("✅ EnhanceSegNet创建成功")
    
    # 测试前向传播
    x = torch.randn(1, 3, 512, 512)
    enhanced_img, mask, quality = model(x)
    print(f"✅ 前向传播成功")
    print(f"   - 增强图像形状: {enhanced_img.shape}")
    print(f"   - 分割掩膜形状: {mask.shape}")
    print(f"   - 质量分数形状: {quality.shape}")
except Exception as e:
    print(f"❌ EnhanceSegNet测试失败: {e}")

# 测试损失函数
print("\n=== 测试损失函数 ===")
try:
    from utils.losses import RetinaSAMLoss, EnhanceSegLoss
    
    retina_loss = RetinaSAMLoss()
    print("✅ RetinaSAMLoss创建成功")
    
    enhance_loss = EnhanceSegLoss()
    print("✅ EnhanceSegLoss创建成功")
except Exception as e:
    print(f"❌ 损失函数测试失败: {e}")

# 测试评估指标
print("\n=== 测试评估指标 ===")
try:
    from utils.metrics import dice_coefficient, iou
    import numpy as np
    
    # 创建测试数据
    pred = np.random.rand(256, 256) > 0.5
    target = np.random.rand(256, 256) > 0.5
    
    dice = dice_coefficient(pred, target)
    iou_score = iou(pred, target)
    
    print(f"✅ Dice系数: {dice:.4f}")
    print(f"✅ IoU: {iou_score:.4f}")
except Exception as e:
    print(f"❌ 评估指标测试失败: {e}")

print("\n" + "="*40)
print("🎉 演示完成！")
print("\n接下来你可以：")
print("1. 阅读医学图像技术调研报告.md")
print("2. 查看src目录下的代码")
print("3. 使用论文目录中的相关论文")
print("4. 运行训练脚本: python src/train.py --model retinasam")

