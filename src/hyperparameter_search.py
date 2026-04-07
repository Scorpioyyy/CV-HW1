"""
超参数查找模块
支持网格搜索（Grid Search）和随机搜索（Random Search）
"""

import numpy as np
import json
import os
import itertools
import time

from .model   import MLP
from .trainer import SGD, ExponentialLRScheduler, Trainer


#  搜索器基类
class BaseSearch:
    """超参数搜索基类"""

    def __init__(self, save_dir: str = './results/search'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results: list = []

    def _run_one(self, params: dict, X_train, y_train, X_val, y_val) -> tuple:
        """
        用给定超参数训练一个短周期的模型，返回 (best_val_acc, history)
        """
        model = MLP(
            input_size   = X_train.shape[1],
            hidden_sizes = params['hidden_sizes'],
            output_size  = 10,
            activation   = params['activation'],
            weight_decay = params['weight_decay'],
        )
        optimizer = SGD(lr=params['lr'], momentum=params.get('momentum', 0.9))
        scheduler = ExponentialLRScheduler(optimizer, gamma=params.get('lr_decay', 0.97))

        trainer = Trainer(model, optimizer, scheduler, save_dir=self.save_dir)
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs     = params.get('epochs', 20),
            batch_size = params.get('batch_size', 128),
            verbose    = False,
        )
        return trainer.best_val_acc, history

    def _save_results(self, filename: str = 'search_results.json'):
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f'搜索结果已保存至: {path}')

    def best_params(self):
        """返回验证集准确率最高的参数组合"""
        if not self.results:
            return None, 0.0
        best = max(self.results, key=lambda r: r['val_acc'])
        return best['params'], best['val_acc']

    def summary(self):
        """打印搜索结果汇总"""
        if not self.results:
            print('尚无搜索结果')
            return

        sorted_results = sorted(self.results, key=lambda r: r['val_acc'], reverse=True)
        print('\n' + '='*70)
        print(f'超参数搜索结果（共 {len(self.results)} 组，按验证集准确率排序）')
        print('='*70)
        for i, r in enumerate(sorted_results[:10], 1):  # 只展示 Top-10
            print(f'  #{i:2d}  Val Acc: {r["val_acc"]:.4f}  | {r["params"]}')


#  网格搜索
class GridSearch(BaseSearch):
    """
    网格搜索：遍历所有参数组合

    参数:
        param_grid: 字典，key 为参数名，value 为候选值列表
            例:
            {
                'lr':           [0.01, 0.05, 0.1],
                'hidden_sizes': [[256, 128], [512, 256]],
                'weight_decay': [0.0, 0.001],
                'activation':   ['relu', 'tanh'],
                'epochs':       [20],
            }
    """

    def __init__(self, param_grid: dict, save_dir: str = './results/search'):
        super().__init__(save_dir)
        self.param_grid = param_grid

    def _get_combinations(self):
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def run(self, X_train, y_train, X_val, y_val):
        combos = self._get_combinations()
        print(f'\n网格搜索开始，共 {len(combos)} 个参数组合')

        for idx, params in enumerate(combos, 1):
            print(f'\n[{idx}/{len(combos)}] {params}')
            t0 = time.time()
            try:
                val_acc, history = self._run_one(params, X_train, y_train, X_val, y_val)
                self.results.append({
                    'params':  params,
                    'val_acc': float(val_acc),
                    'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                })
                print(f'  Val Acc: {val_acc:.4f}  ({time.time()-t0:.1f}s)')
            except Exception as e:
                print(f'  失败: {e}')

        self._save_results('grid_search_results.json')
        self.summary()
        return self.results


#  随机搜索
class RandomSearch(BaseSearch):
    """
    随机搜索：从参数分布中随机采样

    参数:
        param_distributions: 字典，value 为列表（均匀离散采样）或
            {'type': 'log_uniform', 'low': 1e-4, 'high': 1.0}（对数均匀连续采样）
        n_iter: 采样次数
    """

    def __init__(self, param_distributions: dict, n_iter: int = 20,
                 save_dir: str = './results/search', seed: int = 42):
        super().__init__(save_dir)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.seed   = seed

    def _sample_params(self, rng: np.random.Generator) -> dict:
        params = {}
        for key, dist in self.param_distributions.items():
            if isinstance(dist, list):
                params[key] = rng.choice(dist).tolist() if hasattr(dist[0], '__len__') \
                              else dist[rng.integers(len(dist))]
            elif isinstance(dist, dict) and dist.get('type') == 'log_uniform':
                log_val = rng.uniform(np.log(dist['low']), np.log(dist['high']))
                params[key] = float(np.exp(log_val))
            else:
                raise ValueError(f'不支持的分布类型: {dist}')
        return params

    def run(self, X_train, y_train, X_val, y_val):
        rng = np.random.default_rng(self.seed)
        print(f'\n随机搜索开始，共 {self.n_iter} 次采样')

        for idx in range(1, self.n_iter + 1):
            params = self._sample_params(rng)
            print(f'\n[{idx}/{self.n_iter}] {params}')
            t0 = time.time()
            try:
                val_acc, history = self._run_one(params, X_train, y_train, X_val, y_val)
                self.results.append({
                    'params':  params,
                    'val_acc': float(val_acc),
                    'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                })
                print(f'  Val Acc: {val_acc:.4f}  ({time.time()-t0:.1f}s)')
            except Exception as e:
                print(f'  失败: {e}')

        self._save_results('random_search_results.json')
        self.summary()
        return self.results
