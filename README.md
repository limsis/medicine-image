# 医学图像技术调研 - SAM在眼底视网膜分割的应用

> AI开发技术 - 实验4.1 医学图像技术调研

## 项目简介

本项目研究如何将Segment Anything Model (SAM)扩展应用到眼底视网膜病变分割领域，同时解决低质量医学图像分割的挑战。

## 主要内容

### 1. 研究背景
- 眼底视网膜病变在糖尿病诊断中的重要性
- SAM在医学图像分割的应用现状
- 当前研究存在的问题与不足

### 2. 研究方案

#### RetinaSAM - 眼底视网膜专用SAM模型
- 多尺度空洞空间金字塔池化 (ASPP)
- 细长结构感知注意力机制
- 病灶上下文融合模块
- 边界感知损失 + 骨架加权损失

#### EnhanceSegNet - 低质量图像增强与分割联合网络
- 质量评估网络
- 自适应增强模块
- 端到端联合训练

### 3. 实验规划
- 基线模型：MedSAM
- 分阶段实验设计
- 评估指标：Dice系数、IoU、Hausdorff距离等

## 项目结构

```
医学图像技术调研/
├── README.md                          # 项目说明
├── 医学图像技术调研报告.md             # 完整技术报告 (Markdown)
├── 医学图像技术调研报告.docx           # 完整技术报告 (Word)
├── requirements.txt                   # Python依赖
├── demo_eval.py                       # 模型演示脚本
├── convert_to_word.py                 # Markdown转Word脚本
├── src/                               # 源代码目录
│   ├── models/                        # 模型实现
│   │   ├── __init__.py
│   │   ├── retinasam.py              # RetinaSAM模型
│   │   └── enhancesegnet.py          # EnhanceSegNet模型
│   ├── utils/                         # 工具函数
│   │   ├── __init__.py
│   │   ├── losses.py                 # 损失函数
│   │   └── metrics.py                # 评估指标
│   ├── data/                          # 数据处理
│   │   ├── __init__.py
│   │   └── dataloader.py             # 数据加载器
│   └── train.py                      # 训练脚本
└── 眼底视网膜病变分割-论文PDF/          # 参考文献集合
    ├── SAM相关/
    ├── 眼底视网膜病变分割/
    ├── 医学图像增强/
    └── SAM模型蒸馏/
```

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8.0
- numpy
- pillow

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行演示

```bash
python demo_eval.py
```

### 训练模型

```bash
cd src
python train.py --model retinasam --epochs 10 --batch_size 4 --device cpu
```

## 模型说明

### RetinaSAM
专门针对眼底视网膜分割优化的SAM变体，包含多尺度特征提取和细长结构感知。

### EnhanceSegNet
结合图像质量增强与分割的联合网络，自动调整增强策略以适应不同质量的输入。

## 技术报告

详细的技术报告请查看：
- [医学图像技术调研报告.md](医学图像技术调研报告.md) (Markdown)
- [医学图像技术调研报告.docx](医学图像技术调研报告.docx) (Word)

## 参考文献

本项目基于以下研究工作：

- [SAM] Kirillov et al. "Segment Anything"
- [MedSAM] Ma et al. "Segment Anything Model for Medical Image Segmentation"
- 以及30+篇相关领域论文

详细参考文献列表请见技术报告。

## 目录说明

- `src/models/` - RetinaSAM和EnhanceSegNet模型实现
- `src/utils/` - 损失函数和评估指标
- `src/data/` - 数据加载和预处理
- `src/train.py` - 训练脚本
- `眼底视网膜病变分割-论文PDF/` - 相关论文集合

## 许可证

本项目仅供学习交流使用。

## 联系方式

如有问题或建议，欢迎提交Issue或PR。
