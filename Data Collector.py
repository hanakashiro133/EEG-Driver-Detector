from cushy_serial import CushySerial
from BrainLinkParser import BrainLinkParser
import numpy as np
from feature_matrix import FeatureMatrixCollector, collect_features
from scipy.stats import entropy
from datetime import datetime

# 每隔多少个样本保存一次完整 CSV（同时仍然进行增量追加）
SAVE_EVERY = 100

def calculate_psd(data):
    PSD_b = (data.lowBeta + data.highBeta)  # Beta波的PSD
    PSD_a = (data.lowAlpha + data.highAlpha)  # Alpha波的PSD
    attation_b = (PSD_a + data.theta)  # 分母，Alpha波与Theta波的PSD
    addall = (data.delta + attation_b + PSD_b)  # 所有波的PSD之和
    features1 = PSD_b / attation_b if attation_b != 0 else 0  # Beta/(Alpha+Theta)
    features2 = PSD_b / data.theta if data.theta != 0 else 0  # Beta/Theta
    features3 = PSD_a / addall if addall != 0 else 0  # Alpha/总PSD
    features4 = data.delta / addall if addall != 0 else 0  # Delta/总PSD
    features5 = data.delta
    # 新增第6个特征：基于各波段 PSD 的谱熵（delta, alpha, beta, theta）
    p = np.array([data.delta, PSD_a, PSD_b, data.theta], dtype=float)
    s = p.sum()
    if s > 0:
        p_norm = p / s
        features6 = float(entropy(p_norm))
    else:
        features6 = 0.0

    # 返回六个特征的序列，便于装饰器或外部函数收集
    return [features1, features2, features3, features4, features5, features6]

# 创建特征矩阵收集器并包装 calculate_psd，使每次解析到的数据的 6 个特征被自动追加到矩阵
collector = FeatureMatrixCollector(n_features=6, as_columns=True)
wrapped_calculate_psd = collect_features(collector)(calculate_psd)
parser = BrainLinkParser(wrapped_calculate_psd)

serial = CushySerial("COM4", 115200)

@serial.on_message()
def handle_serial_message(msg: bytes):
    parser.parse(msg)
    # print(f"[serial] rec msg: {msg}")
    # 每次解析后把新增特征增量追加到 CSV（文件名：features.csv）
    try:
        collector.save_incremental('features.csv')
    except Exception:
        # 避免在串口回调中抛出异常中断流程，记录或忽略即可
        pass
    # 周期性保存完整快照（按样本计数）
    try:
        total = collector.matrix.shape[1] if collector.matrix.size else 0
        if total and (total % SAVE_EVERY) == 0:
            fname = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            collector.save_csv(fname)
    except Exception:
        pass