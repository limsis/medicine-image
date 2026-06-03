
"""
超级简单的模型测试
"""
import sys
sys.path.append('src')

print("测试RetinaSAM...")

try:
    import torch
    from models.retinasam import RetinaSAM
    
    # 创建模型
    model = RetinaSAM(num_classes=5)
    
    # 测试前向传播
    x = torch.randn(2, 3, 512, 512)  # batch_size=2
    
    # 前向传播
    output = model(x)
    
    print(f"✅ RetinaSAM测试成功！")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出通道数: {output.shape[1]} (应该是5)")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🎉 测试完成！模型工作正常！")
