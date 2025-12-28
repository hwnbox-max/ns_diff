# Neuro-Symbolic Diffusion (NS-Diff)

**Official Implementation of "Neuro-Symbolic Diffusion: Bridging Interpretable Classification and Generative Verification via Manifold-Aligned Concepts"**

## 📋 概述

NS-Diff是一个统一的框架,将扩散模型的流形学习能力与可微神经符号逻辑的演绎严谨性相结合。通过端到端训练,实现了高精度分类和可解释的生成验证。

### 核心创新

1. **Semantic Manifold Alignment (SMA)**: 通过Jacobian正交正则化,将潜在子梯度投影到正交的可解释概念度量空间
2. **Differentiable Neuro-Symbolic Logic (DNSL)**: 基于Product T-Norms的可微模糊逻辑,实现显式IF-THEN规则的端到端优化
3. **Generative Counterfactual Verification**: 通过合成因果干预,视觉化验证模型的推理过程

## 🔧 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (用于GPU训练)
- 16GB+ RAM (推荐32GB)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/ns-diff.git
cd ns-diff

# 创建虚拟环境
conda create -n nsdiff python=3.9
conda activate nsdiff

# 安装依赖
pip install -r requirements.txt

# 安装基线模型依赖 (可选)
# Concept Bottleneck Models
git clone https://github.com/yewsiang/ConceptBottleneck.git
cd ConceptBottleneck && pip install -e . && cd ..
```

## 📁 项目结构

```
ns-diff/
├── models/
│   ├── ns_diff.py              # NS-Diff核心实现
│   └── baselines.py            # 基线模型 (ResNet, CBM, etc.)
├── data/
│   └── datasets.py             # 数据加载器 (Shapes3D, CelebA-HQ)
├── evaluation/
│   ├── metrics.py              # 评估指标 (MIG, ISR, etc.)
│   └── visualization.py        # 可视化工具
├── experiments/
│   ├── config.json             # 实验配置
│   └── run_all_experiments.py  # 完整实验脚本
├── train.py                    # 训练脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 本文档
```

## 📊 数据集准备

### CelebA-HQ

1. 下载CelebA-HQ数据集:
```bash
# 下载图像
wget https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view

# 下载属性文件
wget https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view
```

2. 整理目录结构:
```
/path/to/celeba-hq/
├── images/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── list_attr_celeba.txt
```

### Shapes3D

1. 下载Shapes3D数据集:
```bash
# 从DeepMind下载
gsutil -m cp -r gs://3d-shapes/3dshapes.h5 /path/to/data/
```

## 🚀 快速开始

### 训练NS-Diff

```bash
python train.py \
    --model ns_diff \
    --dataset celeba-hq \
    --data_path /path/to/celeba-hq \
    --image_dir /path/to/celeba-hq/images \
    --attr_file /path/to/celeba-hq/list_attr_celeba.txt \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_concepts 8 \
    --num_classes 2 \
    --lambda_cls 1.0 \
    --lambda_ortho 0.1 \
    --device cuda
```

### 训练基线模型

#### ResNet-50
```bash
python train.py \
    --model resnet50 \
    --dataset celeba-hq \
    --data_path /path/to/celeba-hq \
    --image_dir /path/to/celeba-hq/images \
    --attr_file /path/to/celeba-hq/list_attr_celeba.txt \
    --epochs 100 \
    --batch_size 32
```

#### Standard CBM
```bash
python train.py \
    --model standard_cbm \
    --dataset celeba-hq \
    --data_path /path/to/celeba-hq \
    --image_dir /path/to/celeba-hq/images \
    --attr_file /path/to/celeba-hq/list_attr_celeba.txt \
    --num_concepts 8 \
    --epochs 100
```

## 🧪 复现论文实验

### 运行完整实验套件

```bash
# 1. 修改实验配置
vim experiments/config.json

# 2. 运行所有实验
python experiments/run_all_experiments.py --config experiments/config.json
```

这将自动运行:
- 基线对比实验 (Table 1)
- 消融研究 (Table 1 底部)
- Shapes3D验证实验
- 生成反事实可视化 (Figure 2)

### 单独运行实验

#### 基线对比
```bash
# NS-Diff
python train.py --model ns_diff --dataset celeba-hq ...

# ResNet-50
python train.py --model resnet50 --dataset celeba-hq ...

# Standard CBM
python train.py --model standard_cbm --dataset celeba-hq ...

# Post-hoc CBM
python train.py --model posthoc_cbm --dataset celeba-hq ...

# DisDiff-FNNC
python train.py --model disdiff_fnnc --dataset celeba-hq ...
```

#### 消融研究
```bash
# 完整NS-Diff
python train.py --model ns_diff --lambda_cls 1.0 --lambda_ortho 0.1 ...

# w/o 正交正则化
python train.py --model ns_diff --lambda_cls 1.0 --lambda_ortho 0.0 ...

# 其他消融变体需要修改代码
```

## 📈 评估和可视化

### 计算评估指标

```python
from evaluation.metrics import compute_metrics
from models.ns_diff_error import NSDiff
import torch

# 加载模型
model = NSDiff(num_concepts=8, num_classes=2)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 计算指标
metrics = compute_metrics(
    concepts=predicted_concepts,
    labels=true_labels,
    model=model,
    test_loader=test_loader,
    device=device,
    ground_truth_factors=ground_truth_factors  # 如果可用
)

print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"MIG: {metrics['mig']:.4f}")
print(f"ISR: {metrics['isr']:.2f}%")
```

### 生成可视化

```python
from evaluation.visualization import (
    visualize_counterfactual_comparison,
    plot_performance_comparison,
    create_concept_intervention_grid
)

# 反事实对比
visualize_counterfactual_comparison(
    original=original_images,
    counterfactual=counterfactual_images,
    concept_idx=2,
    concept_names=['Bangs', 'Beard', 'Smiling', ...],
    original_concepts=original_concepts,
    cf_concepts=cf_concepts,
    save_path='results/counterfactual.png'
)

# 性能对比
results = {
    'ResNet-50': {'accuracy': 90.2, 'mig': 0.0, 'isr': 0.0},
    'NS-Diff': {'accuracy': 89.3, 'mig': 0.78, 'isr': 91.4}
}
plot_performance_comparison(results, save_path='results/comparison.png')

# 概念干预网格
create_concept_intervention_grid(
    model=model,
    image=test_image,
    concept_idx=0,
    concept_name='Bangs',
    num_steps=7,
    save_path='results/intervention_grid.png'
)
```

## 📊 预期结果

根据论文,在CelebA-HQ上的预期结果:

| Model | Acc (%) | MIG | ISR (%) |
|-------|---------|-----|---------|
| ResNet-50 | 90.2 | N/A | N/A |
| Standard CBM | 86.5 | 0.42 | 23.5 |
| Post-hoc CBM | 85.8 | 0.48 | N/A |
| DisDiff-FNNC | 87.1 | 0.55 | 65.2 |
| **NS-Diff (Ours)** | **89.3** | **0.78** | **91.4** |

消融研究结果:

| Variant | Acc (%) | MIG | ISR (%) |
|---------|---------|-----|---------|
| NS-Diff | 89.3 | 0.78 | 91.4 |
| w/o SMA | 86.1 | 0.45 | 35.8 |
| w/o Ortho | 88.5 | 0.62 | 72.1 |
| w/o DNSL | 89.8 | 0.76 | 90.5 |

## 🔍 核心算法解析

### Algorithm 1: Neuro-Symbolic Manifold Alignment & Joint Training

```
输入: 数据集 D = {(x, y)}, 预训练扩散编码器 E_φ, 解码器 D_ψ
输出: 优化的SMA投影器 P_θ, DNSL逻辑权重 W_rule

1. 初始化 P_θ, W_rule
2. While not converged:
    // Phase 1: 流形感知与对齐
    z ← E_φ(x)                           # 提取潜在子梯度
    c ← P_θ(z)                           # 投影到概念度量空间
    
    // Phase 2: 几何正则化 (Theorem 1)
    L_ortho ← ∑_{i≠j} ‖∇_z c_i · (∇_z c_j)^T‖²_F
    
    // Phase 3: 可微逻辑推理 (DNSL)
    μ ← exp(-(c - m)²/2σ²)               # 语义模糊化
    α ← ∏_{k∈I} μ_k                      # Product T-Norm规则推理
    ŷ ← Softmax(W^T_rule · α)            # 逻辑聚合
    
    // Phase 4: 端到端优化
    L_cls ← CrossEntropy(y, ŷ)
    L_total ← λ_cls · L_cls + λ_ortho · L_ortho
    Backward: ∇L_total 通过 α → μ → c → z 反向传播
    Update: φ, θ, W_rule
    
    // Phase 5: 生成验证 (周期性)
    If iteration % N_check == 0:
        c' ← Intervene(c, target_concept)
        x_cf ← D_ψ(c')                   # 生成反事实
```

## 🛠️ 自定义和扩展

### 添加新的数据集

```python
from data.datasets import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data_path, ...):
        # 加载数据
        pass
    
    def __getitem__(self, idx):
        image = ...  # 加载图像
        target = ...  # 类别标签
        concepts = ...  # 概念向量
        return image, target, concepts
```

### 修改概念数量

```python
# 在训练时指定
python train.py \
    --model ns_diff \
    --num_concepts 12 \  # 增加到12个概念
    --num_classes 5      # 5类分类
```

### 调整损失权重

根据任务特性调整λ_cls和λ_ortho:

```bash
# 更注重分类性能
python train.py --lambda_cls 1.0 --lambda_ortho 0.05

# 更注重概念解耦
python train.py --lambda_cls 0.8 --lambda_ortho 0.2
```

## 🐛 常见问题

### Q1: CUDA内存不足
```bash
# 减小批次大小
--batch_size 16

# 或使用梯度累积
--accumulation_steps 2
```

### Q2: 训练不收敛
- 降低学习率: `--learning_rate 5e-5`
- 增加正则化: `--weight_decay 1e-4`
- 调整损失权重: `--lambda_ortho 0.05`

### Q3: ISR分数过低
- 增加正交正则化权重: `--lambda_ortho 0.2`
- 检查解码器训练是否充分
- 确保概念对齐质量

## 📝 引用

如果你使用了本代码,请引用:

```bibtex
@inproceedings{nsdiff2025,
  title={Neuro-Symbolic Diffusion: Bridging Interpretable Classification and Generative Verification via Manifold-Aligned Concepts},
  author={Anonymous},
  booktitle={Conference},
  year={2025}
}
```


Qickstart
# 1. 安装
pip install -r requirements.txt
python test_installation.py

# 2. 准备数据
# 下载CelebA-HQ和Shapes3D

# 3. 训练NS-Diff
python train.py --model ns_diff --dataset celeba-hq \
    --data_path /path/to/data --epochs 100

# 4. 运行完整实验
python experiments/run_all_experiments.py

## 📧 联系方式

如有问题或建议,请通过以下方式联系:
- GitHub Issues: [项目Issues页面]
- Email: [your-email@domain.com]

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- Concept Bottleneck Models (Koh et al., ICML 2020)
- Diffusion Models (Ho et al., NeurIPS 2020)
- Post-hoc CBM (Yuksekgonul et al., ICLR 2023)
- Shapes3D Dataset (DeepMind)
- CelebA-HQ Dataset (Nvidia)