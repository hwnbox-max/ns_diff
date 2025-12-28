"""
Visualization Tools
用于生成论文中的图表和反事实验证可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from PIL import Image


def denormalize(tensor: torch.Tensor,
                mean: List[float] = [0.5, 0.5, 0.5],
                std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
    """
    反归一化图像tensor用于显示

    Args:
        tensor: 归一化的图像tensor [C, H, W] 或 [B, C, H, W]
        mean: 归一化均值
        std: 归一化标准差
    Returns:
        image: 反归一化的numpy数组
    """
    if tensor.dim() == 4:
        # Batch of images
        images = []
        for t in tensor:
            img = denormalize(t, mean, std)
            images.append(img)
        return np.stack(images)

    # Single image [C, H, W]
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and transpose to [H, W, C]
    image = tensor.cpu().numpy().transpose(1, 2, 0)

    return image


def visualize_counterfactual_comparison(original: torch.Tensor,
                                        counterfactual: torch.Tensor,
                                        concept_idx: int,
                                        concept_names: List[str],
                                        original_concepts: torch.Tensor,
                                        cf_concepts: torch.Tensor,
                                        save_path: Optional[str] = None):
    """
    可视化原始图像和反事实图像的对比 (对应论文Figure 2)

    Args:
        original: 原始图像 [B, 3, H, W]
        counterfactual: 反事实图像 [B, 3, H, W]
        concept_idx: 被干预的概念索引
        concept_names: 概念名称列表
        original_concepts: 原始概念值 [B, num_concepts]
        cf_concepts: 反事实概念值 [B, num_concepts]
        save_path: 保存路径
    """
    batch_size = original.size(0)
    num_concepts = len(concept_names)

    # 创建图表
    fig = plt.figure(figsize=(20, 4 * batch_size))

    for b in range(batch_size):
        # 原始图像
        ax1 = plt.subplot(batch_size, 3, b * 3 + 1)
        orig_img = denormalize(original[b])
        ax1.imshow(orig_img)
        ax1.set_title(f'Original Image\n{concept_names[concept_idx]} = {original_concepts[b, concept_idx]:.2f}',
                      fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 反事实图像
        ax2 = plt.subplot(batch_size, 3, b * 3 + 2)
        cf_img = denormalize(counterfactual[b])
        ax2.imshow(cf_img)
        ax2.set_title(f'Counterfactual Image\n{concept_names[concept_idx]} = {cf_concepts[b, concept_idx]:.2f}',
                      fontsize=12, fontweight='bold', color='red')
        ax2.axis('off')

        # 概念变化条形图
        ax3 = plt.subplot(batch_size, 3, b * 3 + 3)

        x = np.arange(num_concepts)
        width = 0.35

        orig_vals = original_concepts[b].cpu().numpy()
        cf_vals = cf_concepts[b].cpu().numpy()

        bars1 = ax3.bar(x - width / 2, orig_vals, width, label='Original', alpha=0.8)
        bars2 = ax3.bar(x + width / 2, cf_vals, width, label='Counterfactual', alpha=0.8)

        # 高亮被干预的概念
        bars1[concept_idx].set_color('blue')
        bars1[concept_idx].set_alpha(1.0)
        bars2[concept_idx].set_color('red')
        bars2[concept_idx].set_alpha(1.0)

        ax3.set_ylabel('Concept Value', fontsize=10)
        ax3.set_title('Concept Changes', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(concept_names, rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved counterfactual comparison to {save_path}")

    plt.show()


def visualize_concept_manifold(concepts: np.ndarray,
                               labels: np.ndarray,
                               concept_names: List[str],
                               method: str = 'tsne',
                               save_path: Optional[str] = None):
    """
    可视化概念空间的流形结构 (使用t-SNE或PCA)

    Args:
        concepts: 概念表示 [N, num_concepts]
        labels: 类别标签 [N]
        concept_names: 概念名称
        method: 降维方法 ('tsne' 或 'pca')
        save_path: 保存路径
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # 降维到2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Concept Manifold'
    else:
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Concept Manifold'

    concepts_2d = reducer.fit_transform(concepts)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 按类别着色
    ax1 = axes[0]
    scatter1 = ax1.scatter(concepts_2d[:, 0], concepts_2d[:, 1],
                           c=labels, cmap='viridis',
                           alpha=0.6, s=20)
    ax1.set_title(f'{title}\nColored by Class Label', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Component 1', fontsize=12)
    ax1.set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Class')

    # 右图: 按第一个概念着色
    ax2 = axes[1]
    scatter2 = ax2.scatter(concepts_2d[:, 0], concepts_2d[:, 1],
                           c=concepts[:, 0], cmap='RdYlBu',
                           alpha=0.6, s=20)
    ax2.set_title(f'{title}\nColored by {concept_names[0]}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Component 1', fontsize=12)
    ax2.set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label=concept_names[0])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved manifold visualization to {save_path}")

    plt.show()


def plot_performance_comparison(results: Dict[str, Dict[str, float]],
                                metrics: List[str] = ['accuracy', 'mig', 'isr'],
                                save_path: Optional[str] = None):
    """
    绘制不同模型的性能对比图 (对应论文Table 1)

    Args:
        results: 字典 {model_name: {metric: value}}
        metrics: 要显示的指标列表
        save_path: 保存路径
    """
    model_names = list(results.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        values = [results[model].get(metric, 0) for model in model_names]
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # 设置y轴范围
        if metric == 'accuracy' or metric == 'isr':
            ax.set_ylim([0, 100])
        else:
            ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")

    plt.show()


def plot_training_curves(log_dir: str,
                         metrics: List[str] = ['loss', 'accuracy'],
                         save_path: Optional[str] = None):
    """
    绘制训练曲线

    Args:
        log_dir: TensorBoard日志目录
        metrics: 要显示的指标
        save_path: 保存路径
    """
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing import event_accumulator

    # 读取TensorBoard日志
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 尝试读取训练和测试指标
        for split in ['train', 'test']:
            tag = f'{split}/{metric}'
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                ax.plot(steps, values, label=split.capitalize(), linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Over Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def visualize_ablation_study(ablation_results: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None):
    """
    可视化消融研究结果

    Args:
        ablation_results: {variant_name: {metric: value}}
        save_path: 保存路径
    """
    variants = list(ablation_results.keys())
    metrics = list(ablation_results[variants[0]].keys())

    # 创建热图数据
    data = []
    for variant in variants:
        row = [ablation_results[variant][m] for m in metrics]
        data.append(row)

    data = np.array(data)

    # 归一化到[0, 1]以便比较
    data_norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.max() > col.min():
            data_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            data_norm[:, j] = 1.0

    # 创建热图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 原始值
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=metrics, yticklabels=variants,
                ax=ax1, cbar_kws={'label': 'Value'})
    ax1.set_title('Ablation Study Results (Absolute Values)', fontsize=14, fontweight='bold')

    # 归一化值
    sns.heatmap(data_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=metrics, yticklabels=variants,
                ax=ax2, cbar_kws={'label': 'Normalized'}, vmin=0, vmax=1)
    ax2.set_title('Ablation Study Results (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation study to {save_path}")

    plt.show()


def create_concept_intervention_grid(model,
                                     image: torch.Tensor,
                                     concept_idx: int,
                                     concept_name: str,
                                     num_steps: int = 7,
                                     device: torch.device = torch.device('cpu'),
                                     save_path: Optional[str] = None):
    """
    创建概念干预的网格可视化
    展示从0到1连续改变一个概念时图像的变化

    Args:
        model: NS-Diff模型
        image: 输入图像 [1, 3, H, W]
        concept_idx: 概念索引
        concept_name: 概念名称
        num_steps: 插值步数
        device: 计算设备
        save_path: 保存路径
    """
    model.eval()

    # 生成不同概念值
    concept_values = np.linspace(0, 1, num_steps)

    generated_images = []
    generated_concepts = []

    with torch.no_grad():
        for val in concept_values:
            x_cf, info = model.generate_counterfactual(
                image.to(device),
                concept_idx,
                float(val)
            )
            generated_images.append(x_cf.cpu())
            generated_concepts.append(info['generated_concepts'].cpu())

    # 创建网格
    fig, axes = plt.subplots(2, num_steps, figsize=(3 * num_steps, 6))

    for i, (img, concepts) in enumerate(zip(generated_images, generated_concepts)):
        # 图像
        ax_img = axes[0, i]
        img_np = denormalize(img[0])
        ax_img.imshow(img_np)
        ax_img.set_title(f'{concept_name}={concept_values[i]:.2f}', fontsize=10)
        ax_img.axis('off')

        # 概念值
        ax_bar = axes[1, i]
        concept_vals = concepts[0].numpy()
        colors = ['red' if j == concept_idx else 'gray' for j in range(len(concept_vals))]
        ax_bar.bar(range(len(concept_vals)), concept_vals, color=colors, alpha=0.7)
        ax_bar.set_ylim([0, 1])
        ax_bar.set_xticks([])
        if i == 0:
            ax_bar.set_ylabel('Concept Values', fontsize=10)
        ax_bar.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Concept Intervention: {concept_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved intervention grid to {save_path}")

    plt.show()


# 示例用法
if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")

    # 创建示例数据用于测试
    print("\nTesting performance comparison plot...")

    results = {
        'ResNet-50': {'accuracy': 90.2, 'mig': 0.0, 'isr': 0.0},
        'Standard CBM': {'accuracy': 86.5, 'mig': 0.42, 'isr': 23.5},
        'Post-hoc CBM': {'accuracy': 85.8, 'mig': 0.48, 'isr': 0.0},
        'DisDiff-FNNC': {'accuracy': 87.1, 'mig': 0.55, 'isr': 65.2},
        'NS-Diff (Ours)': {'accuracy': 89.3, 'mig': 0.78, 'isr': 91.4}
    }

    plot_performance_comparison(results, save_path='./test_comparison.png')