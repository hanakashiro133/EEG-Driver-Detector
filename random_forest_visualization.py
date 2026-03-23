#随机森林二分类训练与可视化脚本


#用法示例（在项目根目录运行）：
#python RF\random_forest_visualization.py --features RF/features.csv --out RF/rf_vis.png

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from feature_matrix import FeatureMatrixCollector
except Exception:
    FeatureMatrixCollector = None


def load_features_csv(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # 跳过表头（f1,...）
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data


def main(args):
    if args.features and os.path.exists(args.features):
        X = load_features_csv(args.features)
    else:        
        raise FileNotFoundError(f"特征文件未找到: {args.features}")

    # 标签加载
    if args.labels and os.path.exists(args.labels):
        y = np.loadtxt(args.labels, delimiter=',')
    else:
        raise AttributeError(f"标签加载失败")

    # 标准差化
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # PCA 到 2 维用于可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=args.test_size, random_state=args.random_state)

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 在 PCA 平面上绘制决策面：在 pca 空间上构建网格，逆变换到 scaled 特征，再预测
    x_min, x_max = X_pca[:, 0].min() - 1.0, X_pca[:, 0].max() + 1.0
    y_min, y_max = X_pca[:, 1].min() - 1.0, X_pca[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, args.resolution), np.linspace(y_min, y_max, args.resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = pca.inverse_transform(grid)
    Z = clf.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.RdYlBu
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=40, cmap=cmap, edgecolor='k')
    plt.xlabel('β/(α+θ)')
    plt.ylabel('θ/β')
    plt.title(f'RandomForest (acc={acc:.3f}) PCA-Visual')
    plt.colorbar(scatter, ticks=[0, 1])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"可视化已保存到 {args.out}")
    else:
        plt.show()
