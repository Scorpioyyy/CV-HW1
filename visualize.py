"""
可视化脚本
生成：
  1. 训练/验证 Loss 曲线 + 验证集 Accuracy 曲线
  2. 第一层隐藏层权重可视化
  3. 混淆矩阵热力图
  4. 错误样本分析图
  5. 超参数搜索结果汇总图
用法:
    python visualize.py [--model_path ./models/best_model.npy]
                        [--history_path ./models/history.json]
                        [--search_path ./results/search/grid_search_results.json]
                        [--data_dir ./data] [--out_dir ./results/figures]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合脚本生成图片
import matplotlib.pyplot as plt

from src.data_loader import FashionMNISTLoader
from src.model       import MLP
from src.evaluator   import Evaluator

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='可视化脚本')
    parser.add_argument('--model_path',   type=str, default='./models/best_model.npy')
    parser.add_argument('--history_path', type=str, default='./models/history.json')
    parser.add_argument('--search_path',  type=str, default='./results/search/grid_search_results.json')
    parser.add_argument('--data_dir',     type=str, default='./data')
    parser.add_argument('--out_dir',      type=str, default='./results/figures')
    return parser.parse_args()


#  1. 训练曲线
def plot_training_curves(history: dict, out_dir: str):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('训练过程曲线', fontsize=15, fontweight='bold')

    # Loss 曲线
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='训练集 Loss', color='#2196F3', linewidth=2)
    ax.plot(epochs, history['val_loss'],   label='验证集 Loss', color='#FF5722', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Loss 曲线', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in history['val_acc']],
            label='验证集 Accuracy', color='#4CAF50', linewidth=2)
    best_epoch = int(np.argmax(history['val_acc'])) + 1
    best_acc   = max(history['val_acc']) * 100
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.6, label=f'最优 Epoch={best_epoch}')
    ax.annotate(f'{best_acc:.2f}%', xy=(best_epoch, best_acc),
                xytext=(best_epoch + 1, best_acc - 1.5),
                fontsize=10, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('验证集 Accuracy 曲线', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  训练曲线 → {path}')
    return path


#  2. 第一层权重可视化
def plot_weight_visualization(model: MLP, out_dir: str, n_show: int = 64):
    """将第一层权重矩阵的每列恢复成 28×28 图像并排列展示"""
    W1 = model.linears[0].W            # 形状 (784, hidden_size)
    n_neurons = min(n_show, W1.shape[1])
    cols = 8
    rows = (n_neurons + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    fig.suptitle(f'第一层隐藏层权重可视化（前 {n_neurons} 个神经元）\n'
                 '每个子图对应一个神经元在输入空间的感受野',
                 fontsize=12, fontweight='bold')

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')
        if i < n_neurons:
            w = W1[:, i].reshape(28, 28)
            # 对每个权重图像做独立归一化，便于观察
            vmax = max(abs(w.min()), abs(w.max()))
            ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, 'weight_visualization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  权重可视化 → {path}')
    return path


#  3. 混淆矩阵热力图
def plot_confusion_matrix(cm: np.ndarray, out_dir: str):
    class_names = FashionMNISTLoader.CLASS_NAMES
    n = len(class_names)

    # 归一化（按行，即真实类别）
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / np.maximum(row_sum, 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('混淆矩阵', fontsize=14, fontweight='bold')

    for ax, data, title, fmt in [
        (axes[0], cm,      '原始计数',     'd'),
        (axes[1], cm_norm, '行归一化（准确率）', '.2f'),
    ]:
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('预测类别', fontsize=11)
        ax.set_ylabel('真实类别', fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 在格子中写数值
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                text = f'{val:{fmt}}' if fmt != 'd' else f'{int(val)}'
                color = 'white' if val > thresh else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  混淆矩阵 → {path}')
    return path


#  4. 错例分析
def plot_error_analysis(X_test, y_test, evaluator: Evaluator, out_dir: str,
                        n_samples: int = 16, mean: float = 0.0, std: float = 1.0):
    """展示分类错误的图像，标注真实标签和预测标签"""
    error_idx, true_labels, pred_labels = evaluator.get_error_samples(
        X_test, y_test, n=n_samples, seed=0
    )

    class_names = FashionMNISTLoader.CLASS_NAMES
    cols = 4
    rows = (len(error_idx) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle('错误分类样本分析', fontsize=14, fontweight='bold')

    for i, (idx, true_lbl, pred_lbl) in enumerate(zip(error_idx, true_labels, pred_labels)):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]

        # 反标准化后显示图像
        img = X_test[idx].reshape(28, 28) * std + mean
        ax.imshow(img, cmap='gray', interpolation='nearest')
        ax.axis('off')

        # 标题：真实 vs 预测
        ax.set_title(
            f'真: {class_names[true_lbl]}\n预: {class_names[pred_lbl]}',
            fontsize=9,
            color='red' if true_lbl != pred_lbl else 'green'
        )

    # 隐藏多余的子图
    for i in range(len(error_idx), rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'error_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  错例分析 → {path}')
    return path, error_idx, true_labels, pred_labels


#  5. 超参数搜索结果
def plot_search_results(search_results: list, out_dir: str):
    """绘制超参数搜索结果：按验证集准确率排序的条形图"""
    if not search_results:
        print('  无搜索结果，跳过')
        return None

    # 按准确率降序排列，只展示前 20 组
    results = sorted(search_results, key=lambda r: r['val_acc'], reverse=True)[:20]
    labels  = [
        f"lr={r['params'].get('lr', '?'):.3f}\n"
        f"h={r['params'].get('hidden_sizes', '?')}\n"
        f"wd={r['params'].get('weight_decay', '?')}\n"
        f"act={r['params'].get('activation', '?')}"
        for r in results
    ]
    accs = [r['val_acc'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(max(14, len(results) * 0.8), 6))
    bars = ax.bar(range(len(results)), accs, color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)

    # 最优的用不同颜色
    bars[0].set_color('#FF5722')
    bars[0].set_alpha(1.0)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel('验证集准确率 (%)', fontsize=12)
    ax.set_title('超参数搜索结果（Top-20）', fontsize=13, fontweight='bold')
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱子顶部写数值
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, 'search_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  搜索结果 → {path}')
    return path


def plot_search_heatmap(search_results: list, out_dir: str):
    """绘制 lr × weight_decay 准确率热力图（固定其他参数为最优值）"""
    if not search_results:
        return None

    # 提取所有 lr 和 weight_decay 的组合
    lrs  = sorted(set(r['params']['lr'] for r in search_results))
    wds  = sorted(set(r['params']['weight_decay'] for r in search_results))

    # 对每个 (lr, wd) 组合，取 relu + [256,128] 的结果（若存在）
    def get_acc(lr, wd):
        candidates = [r for r in search_results
                      if abs(r['params']['lr'] - lr) < 1e-9
                      and abs(r['params']['weight_decay'] - wd) < 1e-9
                      and r['params'].get('activation') == 'relu']
        if not candidates:
            candidates = [r for r in search_results
                          if abs(r['params']['lr'] - lr) < 1e-9
                          and abs(r['params']['weight_decay'] - wd) < 1e-9]
        if candidates:
            return max(c['val_acc'] for c in candidates) * 100
        return np.nan

    matrix = np.array([[get_acc(lr, wd) for wd in wds] for lr in lrs])

    if np.all(np.isnan(matrix)):
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(lrs)))
    ax.set_xticklabels([f'{wd}' for wd in wds], fontsize=10)
    ax.set_yticklabels([f'{lr}' for lr in lrs], fontsize=10)
    ax.set_xlabel('Weight Decay (L2)', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('超参数热力图：验证集准确率 (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)

    for i in range(len(lrs)):
        for j in range(len(wds)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'search_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  搜索热力图 → {path}')
    return path


#  主入口
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'\n生成可视化图片，输出目录: {args.out_dir}')

    # 加载数据
    loader = FashionMNISTLoader(data_dir=args.data_dir)
    _, _, (X_test, y_test) = loader.load_data(val_ratio=0.1, normalize=True)
    mean = getattr(loader, 'mean', 0.0)
    std  = getattr(loader, 'std',  1.0)

    # 加载模型
    if not os.path.exists(args.model_path):
        print(f'警告：未找到模型文件 {args.model_path}，跳过需要模型的图')
        model = None
    else:
        model = MLP.from_file(args.model_path)
        print(f'模型加载成功: {args.model_path}')

    # 加载训练历史
    if os.path.exists(args.history_path):
        with open(args.history_path, encoding='utf-8') as f:
            history = json.load(f)
        print(f'历史记录加载成功: {args.history_path}')
        plot_training_curves(history, args.out_dir)
    else:
        print(f'警告：未找到历史文件 {args.history_path}')

    if model is not None:
        evaluator = Evaluator(model)

        # 权重可视化
        plot_weight_visualization(model, args.out_dir)

        # 混淆矩阵
        cm = evaluator.confusion_matrix(X_test, y_test)
        plot_confusion_matrix(cm, args.out_dir)

        # 错例分析
        plot_error_analysis(X_test, y_test, evaluator, args.out_dir,
                            n_samples=16, mean=mean, std=std)

    # 超参数搜索结果
    if os.path.exists(args.search_path):
        with open(args.search_path, encoding='utf-8') as f:
            search_results = json.load(f)
        print(f'搜索结果加载成功: {args.search_path}')
        plot_search_results(search_results, args.out_dir)
        plot_search_heatmap(search_results, args.out_dir)
    else:
        print(f'警告：未找到搜索结果文件 {args.search_path}')

    print('\n所有图片生成完毕！')


if __name__ == '__main__':
    main()
