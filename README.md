# HW1：从零开始构建三层神经网络分类器

Fashion-MNIST 图像分类，三层 MLP，纯 NumPy 实现。

---

## 目录结构

```
HW1/
├── data/                       # Fashion-MNIST 原始数据（自动下载）
├── models/                     # 训练好的模型权重
│   ├── best_model.npy          # 最优模型权重
│   └── history.json            # 训练历史记录
├── results/
│   ├── figures/                # 可视化图片
│   └── search/                 # 超参数搜索结果
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载与预处理
│   ├── model.py                # MLP 模型定义（含手动反向传播）
│   ├── trainer.py              # 训练循环（SGD + LR Decay + L2 正则化）
│   ├── evaluator.py            # 测试评估（准确率 + 混淆矩阵）
│   └── hyperparameter_search.py # 超参数搜索（网格搜索 / 随机搜索）
├── train.py                    # 训练脚本
├── test.py                     # 测试脚本
├── search.py                   # 超参数搜索脚本
├── visualize.py                # 可视化脚本
├── report.md                   # 实验报告
└── README.md
```

---

## 环境依赖

```
Python >= 3.8
numpy
matplotlib
```

安装依赖：
```bash
pip install numpy matplotlib
```

---

## 快速开始

### 1. 训练模型

```bash
python train.py
```

常用参数：
```bash
python train.py \
  --epochs 50 \
  --lr 0.05 \
  --hidden 256 128 \
  --activation relu \
  --weight_decay 0.001 \
  --batch_size 128 \
  --lr_decay 0.97 \
  --data_dir ./data \
  --save_dir ./models
```

训练脚本会：
- 自动下载 Fashion-MNIST 数据集到 `./data/`
- 按验证集准确率自动保存最优权重到 `./models/best_model.npy`
- 保存训练历史到 `./models/history.json`

### 2. 测试模型

```bash
python test.py --model_path ./models/best_model.npy --data_dir ./data
```

输出测试集准确率和混淆矩阵。

### 3. 超参数搜索

**网格搜索**（遍历所有组合，约 36 组）：
```bash
python search.py --mode grid --epochs 20
```

**随机搜索**（随机采样 20 组）：
```bash
python search.py --mode random --epochs 20 --n_iter 20
```

结果保存至 `./results/search/`。

### 4. 生成可视化图片

```bash
python visualize.py \
  --model_path ./models/best_model.npy \
  --history_path ./models/history.json \
  --search_path ./results/search/grid_search_results.json \
  --data_dir ./data \
  --out_dir ./results/figures
```

生成图片：
- `training_curves.png`：训练/验证 Loss 曲线 + 验证集 Accuracy 曲线
- `weight_visualization.png`：第一层隐藏层权重可视化
- `confusion_matrix.png`：混淆矩阵热力图
- `error_analysis.png`：错误分类样本分析
- `search_results.png`：超参数搜索结果柱状图
- `search_heatmap.png`：超参数热力图

---

## 模型下载

训练好的最优模型权重可从以下链接下载：

[Google Drive 模型权重](https://drive.google.com/file/d/1gYOVv5CpU0LPRcfJ4NvFV4kp2doXD_5h/view?usp=drive_link)

下载后放置于 `./models/best_model.npy` 即可直接运行 `test.py`。

---

## 实验结果

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 见 report.md |
| 最优验证集准确率 | 见 report.md |

详细实验报告见 [report.md](report.md)。
