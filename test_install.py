
"""
简单的安装测试脚本
"""
import sys
print("Python 版本:", sys.version)

# 检查基础依赖
try:
    import numpy as np
    print(f"numpy 版本: {np.__version__}")
except ImportError:
    print("numpy 未安装")

try:
    from PIL import Image
    print("Pillow 已安装")
except ImportError:
    print("Pillow 未安装")

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
except ImportError:
    print("PyTorch 未安装 - 这没关系，我们可以先测试其他部分")

print("\n=== 测试完成 ===")
print("即使没有PyTorch，你也可以：")
print("1. 阅读技术报告")
print("2. 查看代码结构")
print("3. 使用已有论文进行学习")

