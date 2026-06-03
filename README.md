# 医学图像技术调研

AI开发技术 - 实验4.1

## 项目简介

一个基于SAM架构的眼底视网膜图像分割demo项目。

## 项目结构

```
├── README.md
├── requirements.txt
├── demo_eval.py
├── convert_to_word.py
├── src/
│   ├── models/
│   │   ├── retinasam.py
│   │   └── enhancesegnet.py
│   ├── utils/
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── data/
│   │   └── dataloader.py
│   └── train.py
└── 眼底视网膜病变分割-论文PDF/
```

## 快速开始

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

## 环境要求

- Python >= 3.8
- PyTorch >= 1.8.0
- numpy
- Pillow
