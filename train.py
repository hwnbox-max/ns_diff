"""
Training Script for NS-Diff and Baseline Models
实现Algorithm 1的完整训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np

# 导入自定义模块
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ns_diff_error import NSDiff
from models.baselines import build_model
from data.datasets import get_dataloader
from evaluation.metrics import compute_metrics


class Trainer:
    """统一的训练器类,支持所有模型"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            model: 待训练模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置字典
            device: 计算设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # 最佳模型跟踪
        self.best_acc = 0.0
        self.best_epoch = 0

        # 损失权重 (针对NS-Diff)
        self.lambda_cls = config.get('lambda_cls', 1.0)
        self.lambda_ortho = config.get('lambda_ortho', 0.1)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch数
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_ortho_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, targets, concepts) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播 - 根据模型类型选择不同的处理方式
            if isinstance(self.model, NSDiff):
                # NS-Diff需要特殊的损失计算
                losses = self.model.compute_total_loss(
                    images, targets,
                    lambda_cls=self.lambda_cls,
                    lambda_ortho=self.lambda_ortho
                )
                loss = losses['total']
                cls_loss = losses['classification']
                ortho_loss = losses['orthogonality']

                # 用于计算准确率
                outputs = self.model(images)
                predictions = outputs['predictions']

            else:
                # 基线模型 - 标准分类损失
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['predictions']
                    logits = outputs.get('logits', predictions)
                else:
                    predictions = outputs
                    logits = outputs

                loss = nn.CrossEntropyLoss()(logits, targets)
                cls_loss = loss
                ortho_loss = torch.tensor(0.0)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_ortho_loss += ortho_loss.item()

            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

            # 周期性验证生成 (仅NS-Diff, Algorithm 1 Phase 5)
            if isinstance(self.model, NSDiff) and batch_idx % self.config.get('check_interval', 500) == 0:
                self._visualize_counterfactuals(images[:4], epoch, batch_idx)

        # 计算平均指标
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'cls_loss': total_cls_loss / len(self.train_loader),
            'ortho_loss': total_ortho_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }

        return metrics

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        评估模型

        Args:
            epoch: 当前epoch数
        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_concepts = []

        with torch.no_grad():
            for images, targets, concepts in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                if isinstance(self.model, NSDiff):
                    outputs = self.model(images)
                    predictions = outputs['predictions']
                    concept_preds = outputs['concepts']
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['predictions']
                        concept_preds = outputs.get('concepts', None)
                    else:
                        predictions = outputs
                        concept_preds = None

                # 计算损失
                loss = nn.CrossEntropyLoss()(predictions, targets)
                total_loss += loss.item()

                # 统计准确率
                _, predicted = predictions.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 收集用于详细评估
                all_predictions.append(predicted.cpu())
                all_targets.append(targets.cpu())
                if concept_preds is not None:
                    all_concepts.append(concept_preds.cpu())

        # 基本指标
        metrics = {
            'loss': total_loss / len(self.test_loader),
            'accuracy': 100. * correct / total
        }

        # 如果有概念,计算高级指标
        if len(all_concepts) > 0:
            all_concepts = torch.cat(all_concepts, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()

            # 计算MIG等指标 (见metrics.py)
            advanced_metrics = compute_metrics(
                concepts=all_concepts,
                labels=all_targets,
                model=self.model,
                test_loader=self.test_loader,
                device=self.device
            )
            metrics.update(advanced_metrics)

        return metrics

    def _visualize_counterfactuals(self,
                                   images: torch.Tensor,
                                   epoch: int,
                                   batch_idx: int):
        """
        可视化反事实生成 (NS-Diff专用)

        Args:
            images: 输入图像 [B, 3, H, W]
            epoch: 当前epoch
            batch_idx: 当前batch索引
        """
        if not isinstance(self.model, NSDiff):
            return

        self.model.eval()
        with torch.no_grad():
            # 随机选择一个概念进行干预
            target_concept_idx = np.random.randint(0, self.model.num_concepts)
            target_value = np.random.choice([0.0, 1.0])

            # 生成反事实
            x_cf, info = self.model.generate_counterfactual(
                images, target_concept_idx, target_value
            )

            # 记录到TensorBoard
            self.writer.add_images(
                f'counterfactuals/concept_{target_concept_idx}',
                torch.cat([images, x_cf], dim=0),
                epoch * len(self.train_loader) + batch_idx
            )

            # 记录干预效果
            self.writer.add_scalar(
                f'intervention/concept_{target_concept_idx}_error',
                info['intervention_success'].item(),
                epoch * len(self.train_loader) + batch_idx
            )

        self.model.train()

    def train(self):
        """完整训练流程"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'=' * 50}")

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 记录训练指标
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")

            if 'ortho_loss' in train_metrics and train_metrics['ortho_loss'] > 0:
                print(f"Ortho Loss: {train_metrics['ortho_loss']:.6f}")

            # 评估
            eval_metrics = self.evaluate(epoch)

            # 记录评估指标
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f'test/{key}', value, epoch)

            print(f"Test  - Loss: {eval_metrics['loss']:.4f}, "
                  f"Acc: {eval_metrics['accuracy']:.2f}%")

            if 'mig' in eval_metrics:
                print(f"MIG: {eval_metrics['mig']:.4f}")
            if 'isr' in eval_metrics:
                print(f"ISR: {eval_metrics['isr']:.2f}%")

            # 学习率调度
            self.scheduler.step()
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if eval_metrics['accuracy'] > self.best_acc:
                self.best_acc = eval_metrics['accuracy']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, eval_metrics)
                print(f"✓ New best model saved! Acc: {self.best_acc:.2f}%")

            # 定期保存检查点
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, eval_metrics)

        print(f"\n{'=' * 50}")
        print(f"Training completed!")
        print(f"Best accuracy: {self.best_acc:.2f}% at epoch {self.best_epoch}")
        print(f"{'=' * 50}")

        self.writer.close()

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train NS-Diff or baseline models')

    # 模型参数
    parser.add_argument('--model', type=str, default='ns_diff',
                        choices=['ns_diff', 'resnet50', 'standard_cbm', 'posthoc_cbm', 'disdiff_fnnc'],
                        help='Model to train')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='celeba-hq',
                        choices=['shapes3d', 'celeba-hq'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory for CelebA-HQ')
    parser.add_argument('--attr_file', type=str, default=None,
                        help='Attribute file for CelebA-HQ')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # NS-Diff特定参数
    parser.add_argument('--num_concepts', type=int, default=8,
                        help='Number of concepts')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--lambda_ortho', type=float, default=0.1,
                        help='Orthogonality loss weight')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--check_interval', type=int, default=500,
                        help='Check counterfactuals every N batches')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    train_loader, test_loader = get_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_dir=args.image_dir,
        attr_file=args.attr_file
    )

    # 构建模型
    print(f"Building model: {args.model}...")
    if args.model == 'ns_diff':
        model = NSDiff(
            num_concepts=args.num_concepts,
            num_classes=args.num_classes,
            latent_dim=args.latent_dim
        )
    else:
        model = build_model(
            model_name=args.model,
            num_concepts=args.num_concepts,
            num_classes=args.num_classes
        )

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # 配置字典
    config = vars(args)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()