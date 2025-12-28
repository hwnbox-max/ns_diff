# NS-Diff 项目实现概览

## 📚 项目结构

本项目完整实现了论文 "Neuro-Symbolic Diffusion: Bridging Interpretable Classification and Generative Verification via Manifold-Aligned Concepts" 的所有核心算法和实验。

### 核心模块

#### 1. **models/ns_diff.py** - NS-Diff核心实现

包含五个主要类:

```python
# 1. DiffusionEncoder: 扩散编码器 E_φ
#    - 从图像x提取潜在表示z
#    - 基于U-Net bottleneck架构
#    - 捕获数据流形的score function

# 2. SemanticManifoldAlignment (SMA): 语义流形对齐 P_θ
#    - 非线性MLP投影器
#    - Jacobian正交正则化 (Theorem 1)
#    - 将z投影到正交概念空间c

# 3. DifferentiableNeuroSymbolicLogic (DNSL): 可微神经符号逻辑
#    - 高斯隶属函数 (Eq. 2)
#    - Product T-Norm规则推理 (Eq. 3-4)
#    - Softmax去模糊化 (Eq. 5)

# 4. DiffusionDecoder: 扩散解码器 D_ψ
#    - 从修改的概念c'生成反事实x'
#    - 用于生成验证

# 5. NSDiff: 完整框架
#    - 整合所有模块
#    - 实现Algorithm 1的完整训练流程
#    - 提供反事实生成接口
```

**关键方法**:
- `compute_jacobian_orthogonality_loss()`: 实现Eq. 1的Jacobian正交正则化
- `fuzzify()`: 实现Eq. 2的语义模糊化
- `product_tnorm()`: 实现Eq. 3的Product T-Norm推理
- `generate_counterfactual()`: 生成反事实验证
- `compute_total_loss()`: 计算联合损失 (Algorithm 1 Phase 4)

#### 2. **models/baselines.py** - 基线模型

实现了论文Table 1中的所有基线:

```python
# 1. ResNet50BlackBox
#    - 标准ResNet-50分类器
#    - 黑盒模型,无可解释性

# 2. StandardCBM (Koh et al., ICML 2020)
#    - 线性概念投影 (存在Linear Expressiveness Bottleneck)
#    - x -> features -> concepts -> predictions

# 3. PostHocCBM (Yuksekgonul et al., ICLR 2023)
#    - 冻结backbone + 线性概念探针
#    - 残差连接: features + concepts

# 4. DisDiffFNNC
#    - 简化版NS-Diff (无端到端训练)
#    - 冻结扩散特征 + 简单模糊分类器
```

#### 3. **data/datasets.py** - 数据集加载

支持两个标准数据集:

```python
# 1. Shapes3DDataset
#    - 480,000张图像
#    - 6个ground-truth因子
#    - 完美控制变量,用于定量验证

# 2. CelebAHQDataset
#    - 30,000张256x256人脸图像
#    - 8个主要属性作为概念
#    - 真实复杂流形,测试鲁棒性
```

**返回格式**: `(image, target, concepts)`
- `image`: 归一化的图像tensor [3, H, W]
- `target`: 类别标签 (int)
- `concepts`: 概念向量 [num_concepts]

#### 4. **evaluation/metrics.py** - 评估指标

实现论文4.1.3节定义的所有指标:

```python
# 1. compute_mig()
#    - Mutual Information Gap
#    - 衡量概念解耦程度
#    - MIG = (1/K) * ∑_k [I(c_k; v_k^*) - max_{j≠k^*} I(c_k; v_j)]

# 2. compute_intervention_success_rate()
#    - Intervention Success Rate (ISR)
#    - 验证反事实生成质量
#    - ISR = (成功干预次数 / 总干预次数) * 100%

# 3. compute_disentanglement_score()
#    - 多个解耦指标: MIG, SAP, Modularity
#    - 全面评估概念质量
```

#### 5. **evaluation/visualization.py** - 可视化工具

生成论文中的所有图表:

```python
# 1. visualize_counterfactual_comparison()
#    - 对应论文Figure 2
#    - 展示原始图像 vs 反事实图像
#    - 显示概念变化

# 2. plot_performance_comparison()
#    - 对应论文Table 1
#    - 多模型性能对比柱状图

# 3. visualize_concept_manifold()
#    - t-SNE/PCA可视化概念空间
#    - 验证流形结构

# 4. create_concept_intervention_grid()
#    - 展示概念连续干预效果
#    - 7步插值网格
```

#### 6. **train.py** - 训练脚本

统一的训练框架,支持所有模型:

```python
class Trainer:
    def train_epoch():
        # Algorithm 1的完整实现
        # Phase 1-4: 前向传播 + 损失计算 + 反向传播
        # Phase 5: 周期性反事实验证

    def evaluate():
        # 计算所有评估指标
        # Acc, MIG, ISR

    def _visualize_counterfactuals():
        # 生成TensorBoard可视化
```

**命令行接口**:
```bash
python train.py \
    --model {ns_diff, resnet50, standard_cbm, posthoc_cbm, disdiff_fnnc} \
    --dataset {shapes3d, celeba-hq} \
    --epochs 100 \
    --batch_size 32 \
    --lambda_cls 1.0 \
    --lambda_ortho 0.1
```

#### 7. **experiments/run_all_experiments.py** - 完整实验套件

自动化运行所有实验:

```python
class ExperimentRunner:
    def run_baseline_comparison():
        # 运行5个模型对比 (Table 1)
        # 生成对比图和LaTeX表格

    def run_ablation_study():
        # NS-Diff消融研究 (Table 1底部)
        # w/o SMA, w/o Ortho, w/o DNSL

    def run_shapes3d_experiments():
        # Shapes3D数据集验证

    def generate_counterfactual_visualizations():
        # 生成Figure 2的可视化
```

## 🔬 实验复现流程

### Step 1: 环境准备

```bash
# 创建环境
conda create -n nsdiff python=3.9
conda activate nsdiff

# 安装依赖
pip install -r requirements.txt

# 测试安装
python test_installation.py
```

### Step 2: 数据准备

```bash
# CelebA-HQ
./scripts/download_celeba_hq.sh /path/to/data

# Shapes3D
./scripts/download_shapes3d.sh /path/to/data
```

### Step 3: 运行单个实验

```bash
# NS-Diff
python train.py --model ns_diff --dataset celeba-hq \
    --data_path /path/to/celeba-hq \
    --image_dir /path/to/celeba-hq/images \
    --attr_file /path/to/celeba-hq/list_attr_celeba.txt \
    --epochs 100 --batch_size 32

# 基线模型
python train.py --model resnet50 --dataset celeba-hq ...
python train.py --model standard_cbm --dataset celeba-hq ...
```

### Step 4: 运行完整实验套件

```bash
# 修改配置文件
vim experiments/config.json

# 运行所有实验
python experiments/run_all_experiments.py --config experiments/config.json
```

### Step 5: 结果分析

```bash
# 查看TensorBoard日志
tensorboard --logdir ./experimental_results/logs

# 生成最终报告
# 自动生成在 ./experimental_results/EXPERIMENTAL_REPORT.md
```

## 📊 预期输出

### 训练输出示例

```
==================================================
Epoch 50/100
==================================================
Train - Loss: 0.3245, Acc: 89.3%, Ortho Loss: 0.0012
Test  - Loss: 0.3567, Acc: 87.8%, MIG: 0.78, ISR: 91.4%
✓ New best model saved! Acc: 87.8%
```

### 生成的文件

```
experimental_results/
├── baseline_comparison.csv         # 基线对比数据
├── baseline_comparison.png         # 对比图表
├── ablation_study.csv             # 消融研究数据
├── ablation_study.png             # 消融热图
├── counterfactual_concept_0.png   # 反事实可视化
├── counterfactual_concept_1.png
├── shapes3d_results.csv           # Shapes3D结果
├── EXPERIMENTAL_REPORT.md         # 最终报告
└── logs/                          # TensorBoard日志
    ├── ns_diff/
    ├── resnet50/
    └── ...
```

## 🎯 关键实现细节

### 1. Jacobian正交正则化 (Theorem 1)

```python
# 计算每个概念对z的梯度
jacobians = []
for k in range(num_concepts):
    grad = torch.autograd.grad(
        outputs=c[:, k].sum(),
        inputs=z,
        create_graph=True  # 需要二阶导数
    )[0]
    jacobians.append(grad)

# 归一化并计算Gram矩阵
J_norm = F.normalize(torch.stack(jacobians, dim=1), p=2, dim=2)
gram = torch.bmm(J_norm, J_norm.transpose(1, 2))

# 最小化非对角线元素
mask = ~torch.eye(num_concepts, dtype=torch.bool)
L_ortho = (gram[:, mask] ** 2).mean()
```

### 2. Product T-Norm推理 (Eq. 3-4)

```python
# 对数空间实现以保持数值稳定
log_mu = torch.log(mu + 1e-8)
weighted_log = torch.matmul(log_mu, rule_weights.t())
alpha = torch.exp(weighted_log / rule_weights.sum(dim=1).t())

# 保证梯度流通
# ∂α_l/∂c_k = (∏_{m≠k} μ_m) · ∂μ_k/∂c_k ≠ 0
```

### 3. 反事实生成

```python
# 1. 获取原始概念
z = encoder(x)
c = sma(z)

# 2. 干预概念
c_prime = c.clone()
c_prime[:, target_idx] = target_value

# 3. 生成反事实
x_cf = decoder(c_prime)

# 4. 验证干预效果
z_cf = encoder(x_cf)
c_cf = sma(z_cf)
success = |c_cf[:, target_idx] - target_value| < threshold
```

## 🔧 调试技巧

### 1. 监控梯度流

```python
# 在训练循环中添加
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 2. 可视化概念空间

```python
from evaluation.visualization import visualize_concept_manifold

visualize_concept_manifold(
    concepts=learned_concepts,
    labels=true_labels,
    concept_names=concept_names,
    method='tsne'
)
```

### 3. 检查正交性

```python
# 计算概念间的余弦相似度
c_norm = F.normalize(concepts, p=2, dim=0)
similarity = torch.matmul(c_norm.t(), c_norm)
print(f"Off-diagonal max: {similarity.fill_diagonal_(0).abs().max():.4f}")
# 应该接近0
```

## 📈 性能优化

### 1. 数据加载优化

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,        # 多进程加载
    pin_memory=True,      # 加速GPU传输
    prefetch_factor=2     # 预取batch
)
```

### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 梯度累积

```python
# 模拟更大的batch size
accumulation_steps = 4
for i, (images, targets) in enumerate(train_loader):
    loss = model.compute_loss(images, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🎓 学术用途

如果您使用本代码进行研究,请确保:

1. ✅ 引用原论文
2. ✅ 报告所有超参数
3. ✅ 使用相同的数据划分
4. ✅ 运行多次实验并报告标准差
5. ✅ 开源您的代码和结果

## 📞 获取帮助

- **GitHub Issues**: 报告bug或请求功能
- **Discussions**: 技术讨论和问答
- **Email**: 紧急问题联系作者

## 🚀 后续工作方向

1. **扩展到更多模态**: 文本、音频等
2. **大规模数据集**: ImageNet-1K
3. **在线学习**: 增量概念学习
4. **多任务学习**: 共享概念空间
5. **因果推断**: 更严格的因果验证

---

**祝您实验顺利! 🎉**