"""
feature_matrix.py

提供一个特征矩阵收集器和装饰器，用于将每次计算得到的 6 个特征实时追加为特征矩阵的列。

接口：
- `FeatureMatrixCollector(n_features=5, as_columns=True)`
    - `append(features)`：追加单次特征（长度须等于 n_features）
    - `matrix`：当前矩阵（numpy.ndarray）。如果 `as_columns=True`，形状为 `(n_features, N)`，否则为 `(N, n_features)`。
    - `reset()`：清空矩阵

- `collect_features(collector)` 装饰器：
    - 被装饰函数应返回长度为 n_features 的可迭代特征（tuple/list/np.ndarray）。
    - 装饰器会在函数执行后将返回值追加到 `collector` 中并返回原始返回值。

示例用法：
    from feature_matrix import FeatureMatrixCollector, collect_features

    collector = FeatureMatrixCollector()

    @collect_features(collector)
    def compute_features(...):
        # 返回 (f1,f2,f3,f4,f5)
        return [f1,f2,f3,f4,f5]

    compute_features(...)  # 调用时会自动把返回的六个特征追加为新的一列

"""

from typing import Callable, Iterable
import numpy as np
import os


class FeatureMatrixCollector:
    def __init__(self, n_features: int = 6, as_columns: bool = True, dtype=float):
        self.n_features = int(n_features)
        self.as_columns = bool(as_columns)
        self.dtype = dtype
        # 初始化空矩阵
        if self.as_columns:
            # 每列是一次观测，行是各特征
            self._matrix = np.empty((self.n_features, 0), dtype=self.dtype)
        else:
            # 每行是一次观测
            self._matrix = np.empty((0, self.n_features), dtype=self.dtype)

    def append(self, features: Iterable[float]):
        arr = np.asarray(list(features), dtype=self.dtype)
        if arr.size != self.n_features:
            raise ValueError(f"输入特征长度 {arr.size} 与期望 {self.n_features} 不符")
        if self.as_columns:
            col = arr.reshape((self.n_features, 1))
            self._matrix = np.hstack([self._matrix, col]) if self._matrix.size else col
        else:
            row = arr.reshape((1, self.n_features))
            self._matrix = np.vstack([self._matrix, row]) if self._matrix.size else row

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def reset(self):
        if self.as_columns:
            self._matrix = np.empty((self.n_features, 0), dtype=self.dtype)
        else:
            self._matrix = np.empty((0, self.n_features), dtype=self.dtype)

    def save_csv(self, filepath: str, fmt: str = '%.6e', delimiter: str = ','):
        """把当前矩阵保存为 CSV 文件，按样本为行、特征为列的格式写入。

        如果矩阵为空，会创建一个空文件并写入表头。
        """
        # 将矩阵转换为 (N, n_features)
        if self.as_columns:
            mat = self._matrix.T  # shape (N, n_features)
        else:
            mat = self._matrix

        # 准备表头
        header = ','.join([f'f{i+1}' for i in range(self.n_features)])

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # 写入整个矩阵（覆盖）
        with open(filepath, 'w', newline='') as f:
            f.write(header + '\n')
            if mat.size:
                np.savetxt(f, mat, fmt=fmt, delimiter=delimiter)

    def save_incremental(self, filepath: str, fmt: str = '%.6e', delimiter: str = ','):
        """以增量方式追加新样本到 CSV（每次追加新列对应的行）。

        - 第一次调用会创建文件并写入全部已有数据。
        - 后续调用只会追加自上次保存后新增的列数据。
        """
        if self.as_columns:
            total_cols = self._matrix.shape[1]
        else:
            total_cols = self._matrix.shape[0]

        # 记录上次已保存的列索引
        last = getattr(self, '_last_saved_cols', 0)
        if self.as_columns:
            new_cols = self._matrix[:, last:total_cols]
            mat = new_cols.T  # shape (new_N, n_features)
        else:
            new_rows = self._matrix[last:total_cols, :]
            mat = new_rows

        # 如果没有新数据，直接返回
        if mat.size == 0:
            self._last_saved_cols = total_cols
            return

        header = ','.join([f'f{i+1}' for i in range(self.n_features)])
        exists = os.path.exists(filepath)
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        mode = 'a' if exists else 'w'
        with open(filepath, mode, newline='') as f:
            if not exists:
                f.write(header + '\n')
            np.savetxt(f, mat, fmt=fmt, delimiter=delimiter)

        # 更新已保存计数
        self._last_saved_cols = total_cols


def collect_features(collector: FeatureMatrixCollector) -> Callable:
    """装饰器：被装饰函数需返回长度为 collector.n_features 的可迭代对象。

    装饰器会把返回值追加到 collector 并返回原始返回值。
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            # 尝试把返回值视为特征序列
            try:
                collector.append(res)
            except Exception as e:
                # 如果追加失败，抛出更有信息的错误
                raise RuntimeError(f"collect_features: 无法追加特征: {e}") from e
            return res

        # 保留原函数名和文档（简单做法）
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
