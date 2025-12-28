"""
Evaluation Metrics Implementation
实现论文4.1.3节中定义的评估指标:
- Classification Accuracy (Acc)
- Mutual Information Gap (MIG)
- Intervention Success Rate (ISR)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个离散变量之间的互信息

    Args:
        x: 第一个变量 [N]
        y: 第二个变量 [N]
    Returns:
        mi: 互信息值
    """
    # 离散化连续值
    if x.dtype == np.float32 or x.dtype == np.float64:
        x = np.digitize(x, bins=np.linspace(x.min(), x.max(), 20))
    if y.dtype == np.float32 or y.dtype == np.float64:
        y = np.digitize(y, bins=np.linspace(y.min(), y.max(), 20))

    return mutual_info_score(x, y)


def compute_mig(concepts: np.ndarray,
                ground_truth_factors: np.ndarray,
                num_bins: int = 20) -> float:
    """
    计算Mutual Information Gap (MIG)
    MIG衡量概念的解耦程度

    MIG = (1/K) * ∑_k [I(c_k; v_k^*) - max_{j≠k^*} I(c_k; v_j)]

    其中:
    - c_k: 学习到的第k个概念
    - v_j: 第j个ground-truth因子
    - k^*: 对于c_k具有最高互信息的因子

    Args:
        concepts: 学习到的概念 [N, num_concepts]
        ground_truth_factors: 真实因子 [N, num_factors]
        num_bins: 离散化的bin数量
    Returns:
        mig: MIG分数 (越高越好,最大为1)
    """
    num_concepts = concepts.shape[1]
    num_factors = ground_truth_factors.shape[1]

    # 计算所有概念-因子对的互信息矩阵
    mi_matrix = np.zeros((num_concepts, num_factors))

    for k in range(num_concepts):
        c_k = concepts[:, k]
        # 离散化概念
        c_k_discrete = np.digitize(c_k, bins=np.linspace(0, 1, num_bins))

        for j in range(num_factors):
            v_j = ground_truth_factors[:, j]
            # 因子通常已经是离散的,但为了一致性也进行处理
            if v_j.dtype == np.float32 or v_j.dtype == np.float64:
                v_j_discrete = np.digitize(v_j, bins=np.linspace(v_j.min(), v_j.max(), num_bins))
            else:
                v_j_discrete = v_j

            mi_matrix[k, j] = mutual_info_score(c_k_discrete, v_j_discrete)

    # 对每个概念,计算gap
    gaps = []
    for k in range(num_concepts):
        mi_k = mi_matrix[k, :]
        # 排序互信息值
        sorted_mi = np.sort(mi_k)[::-1]

        if len(sorted_mi) >= 2:
            # Gap = 最大MI - 次大MI
            gap = sorted_mi[0] - sorted_mi[1]
        else:
            gap = sorted_mi[0]

        gaps.append(gap)

    # 归一化到[0, 1]
    # 理论最大gap是当一个概念完全编码一个因子时的MI值
    max_possible_gap = np.log(num_bins)  # 最大互信息

    # 计算平均gap并归一化
    mig = np.mean(gaps) / max_possible_gap if max_possible_gap > 0 else 0.0

    return float(np.clip(mig, 0, 1))


def compute_intervention_success_rate(model: nn.Module,
                                      test_loader,
                                      device: torch.device,
                                      num_interventions: int = 100,
                                      oracle_classifier=None) -> float:
    """
    计算Intervention Success Rate (ISR)
    ISR衡量模型生成的反事实是否真正改变了概念

    流程:
    1. 对测试集中的样本,随机选择一个概念进行干预
    2. 使用模型生成反事实图像
    3. 使用oracle分类器验证反事实是否具有目标概念
    4. 计算成功率

    Args:
        model: 待评估模型 (必须有generate_counterfactual方法)
        test_loader: 测试数据加载器
        device: 计算设备
        num_interventions: 测试的干预次数
        oracle_classifier: 用于验证的oracle分类器 (如果None,使用模型自己)
    Returns:
        isr: 干预成功率 (百分比)
    """
    # 检查模型是否支持反事实生成
    if not hasattr(model, 'generate_counterfactual'):
        print("Warning: Model does not support counterfactual generation. Returning ISR=0")
        return 0.0

    model.eval()

    success_count = 0
    total_count = 0

    # 如果没有提供oracle,使用模型自己作为oracle
    if oracle_classifier is None:
        oracle_classifier = model

    with torch.no_grad():
        for images, targets, concepts in tqdm(test_loader, desc="Computing ISR"):
            if total_count >= num_interventions:
                break

            images = images.to(device)
            batch_size = images.size(0)

            for i in range(min(batch_size, num_interventions - total_count)):
                # 随机选择一个概念进行干预
                num_concepts = model.num_concepts if hasattr(model, 'num_concepts') else 8
                target_concept_idx = np.random.randint(0, num_concepts)

                # 目标值: 翻转当前值 (0->1 或 1->0)
                current_value = concepts[i, target_concept_idx].item()
                target_value = 1.0 if current_value < 0.5 else 0.0

                try:
                    # 生成反事实
                    x_cf, info = model.generate_counterfactual(
                        images[i:i + 1],
                        target_concept_idx,
                        target_value
                    )

                    # 使用oracle验证反事实的概念
                    if hasattr(oracle_classifier, 'forward'):
                        outputs = oracle_classifier(x_cf)
                        if isinstance(outputs, dict):
                            cf_concepts = outputs['concepts']
                        else:
                            # 如果oracle不返回概念,无法验证
                            continue
                    else:
                        continue

                    # 检查干预是否成功
                    cf_value = cf_concepts[0, target_concept_idx].item()
                    error = abs(cf_value - target_value)

                    # 如果生成的概念值接近目标值 (容差0.2)
                    if error < 0.2:
                        success_count += 1

                    total_count += 1

                except Exception as e:
                    print(f"Error in counterfactual generation: {e}")
                    continue

    isr = 100.0 * success_count / total_count if total_count > 0 else 0.0
    return isr


def compute_concept_accuracy(predicted_concepts: np.ndarray,
                             true_concepts: np.ndarray,
                             threshold: float = 0.5) -> float:
    """
    计算概念预测准确率
    将连续的概念分数二值化后与真实概念比较

    Args:
        predicted_concepts: 预测的概念 [N, num_concepts]
        true_concepts: 真实概念 [N, num_concepts]
        threshold: 二值化阈值
    Returns:
        accuracy: 概念准确率
    """
    pred_binary = (predicted_concepts > threshold).astype(int)
    true_binary = (true_concepts > threshold).astype(int)

    # 计算每个概念的准确率,然后取平均
    accuracies = []
    for k in range(predicted_concepts.shape[1]):
        acc = accuracy_score(true_binary[:, k], pred_binary[:, k])
        accuracies.append(acc)

    return float(np.mean(accuracies))


def compute_disentanglement_score(concepts: np.ndarray,
                                  factors: np.ndarray) -> Dict[str, float]:
    """
    计算多个解耦指标

    包括:
    - MIG: Mutual Information Gap
    - SAP: Separated Attribute Predictability
    - Modularity

    Args:
        concepts: 学习到的概念 [N, num_concepts]
        factors: 真实因子 [N, num_factors]
    Returns:
        scores: 解耦分数字典
    """
    scores = {}

    # MIG
    scores['mig'] = compute_mig(concepts, factors)

    # SAP: 使用分类器预测因子
    sap_scores = []
    for j in range(factors.shape[1]):
        # 训练一个简单分类器
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

        # 划分训练测试集
        n_train = int(0.8 * len(concepts))
        X_train, X_test = concepts[:n_train], concepts[n_train:]
        y_train, y_test = factors[:n_train, j], factors[n_train:, j]

        # 训练并预测
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # 计算准确率
        acc = accuracy_score(y_test, y_pred)
        sap_scores.append(acc)

    scores['sap'] = float(np.mean(sap_scores))

    # Modularity: 每个概念是否主要对应一个因子
    num_concepts = concepts.shape[1]
    num_factors = factors.shape[1]

    mi_matrix = np.zeros((num_concepts, num_factors))
    for k in range(num_concepts):
        for j in range(num_factors):
            mi_matrix[k, j] = compute_mutual_information(concepts[:, k], factors[:, j])

    # 归一化
    mi_matrix_norm = mi_matrix / (mi_matrix.sum(axis=1, keepdims=True) + 1e-10)

    # Modularity: 每行的最大值的平均
    modularity = np.mean(np.max(mi_matrix_norm, axis=1))
    scores['modularity'] = float(modularity)

    return scores


def compute_metrics(concepts: np.ndarray,
                    labels: np.ndarray,
                    model: nn.Module,
                    test_loader,
                    device: torch.device,
                    ground_truth_factors: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        concepts: 预测的概念 [N, num_concepts]
        labels: 类别标签 [N]
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        ground_truth_factors: 真实因子 [N, num_factors] (如果可用)
    Returns:
        metrics: 所有指标的字典
    """
    metrics = {}

    # 如果有真实因子,计算解耦指标
    if ground_truth_factors is not None:
        print("Computing disentanglement metrics...")
        disentangle_scores = compute_disentanglement_score(concepts, ground_truth_factors)
        metrics.update(disentangle_scores)

    # 计算ISR (仅对支持反事实生成的模型)
    if hasattr(model, 'generate_counterfactual'):
        print("Computing Intervention Success Rate...")
        isr = compute_intervention_success_rate(
            model=model,
            test_loader=test_loader,
            device=device,
            num_interventions=100
        )
        metrics['isr'] = isr

    return metrics


# 单元测试
if __name__ == "__main__":
    # 测试MIG计算
    print("Testing MIG computation...")

    # 创建模拟数据: 3个概念, 3个因子
    # 理想情况: 每个概念完美编码一个因子
    n_samples = 1000
    n_concepts = 3
    n_factors = 3

    # 模拟完美解耦的概念
    factors = np.random.randint(0, 10, size=(n_samples, n_factors))
    concepts = np.zeros((n_samples, n_concepts))

    # 每个概念对应一个因子
    for i in range(n_concepts):
        concepts[:, i] = factors[:, i] / 10.0  # 归一化到[0, 1]
        # 添加一些噪声
        concepts[:, i] += np.random.normal(0, 0.05, n_samples)
        concepts[:, i] = np.clip(concepts[:, i], 0, 1)

    mig = compute_mig(concepts, factors)
    print(f"MIG for perfectly disentangled concepts: {mig:.4f}")
    print("(Should be close to 1.0)")

    # 测试完全纠缠的概念
    concepts_entangled = np.random.rand(n_samples, n_concepts)
    mig_entangled = compute_mig(concepts_entangled, factors)
    print(f"\nMIG for random (entangled) concepts: {mig_entangled:.4f}")
    print("(Should be close to 0.0)")