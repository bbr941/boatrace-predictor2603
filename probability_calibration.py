"""
probability_calibration.py
確率キャリブレーションおよびBenterモデル（Damping Factor）による3連単確率展開モジュール
"""

import numpy as np
import itertools
import os
import json
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'probability_config.json')
CALIBRATOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'calibrator.joblib')

class BoatRaceCalibrator:
    """
    LightGBM LambdaRankスコアまたは生Softmax確率を、現実の1着確率へ補正するキャリブレーター
    サポート手法:
      - 'platt': ロジスティック回帰 (Platt Scaling)
      - 'isotonic': 等張回帰 (Isotonic Regression)
      - 'softmax': 単純Softmax (ベースライン)
    """
    def __init__(self, method='platt'):
        self.method = method
        self.model = None
        self.is_fitted = False
        
        # デフォルトのPlatt Scalingパラメータ (未学習時のフォールバック用: 近似シグモイド係数)
        # s_raw -> z = 0.85 * s_raw - 0.05
        self.fallback_coef = 0.85
        self.fallback_intercept = -0.05

    def fit(self, scores_or_probs, y_is_1st, is_scores=True):
        """
        過去データからキャリブレーションモデルを学習
        scores_or_probs: 1次元配列 (各艇のスコアまたはSoftmax確率)
        y_is_1st: 1次元配列 (1着なら1, それ以外は0)
        """
        X = np.array(scores_or_probs).reshape(-1, 1)
        y = np.array(y_is_1st).astype(int)
        
        if self.method == 'platt':
            self.model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            self.model.fit(X, y)
            self.is_fitted = True
        elif self.method == 'isotonic':
            self.model = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
            self.model.fit(X.ravel(), y)
            self.is_fitted = True
        elif self.method == 'softmax':
            self.is_fitted = True
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def calibrate_scores(self, scores_dict):
        """
        6艇のスコア辞書 {boat_num: score} を入力とし、
        キャリブレーション済みの1着確率辞書 {boat_num: p1} (合計=1.0) を返す
        """
        boats = list(scores_dict.keys())
        scores = np.array([scores_dict[b] for b in boats], dtype=float)
        
        # 0. direct モード (入力が既に正規化済み確率の場合)
        if self.method in ('direct', 'identity') or (np.isclose(np.sum(scores), 1.0, atol=1e-3) and np.all(scores >= 0) and self.method in ('none', None)):
            p_norm = scores / np.maximum(np.sum(scores), 1e-9)
            return {boats[i]: float(p_norm[i]) for i in range(len(boats))}

        # 1. Softmax計算 (生確率)
        max_s = np.max(scores)
        exp_s = np.exp(scores - max_s)
        p_raw = exp_s / np.sum(exp_s)
        
        if self.method == 'softmax' or self.method is None or self.method == 'none':
            return {boats[i]: float(p_raw[i]) for i in range(len(boats))}
        
        # 2. キャリブレーション適用
        if self.is_fitted and self.model is not None:
            if self.method == 'platt':
                # ロジスティック回帰 (高速ベクトル演算)
                w = self.model.coef_[0][0]
                b = self.model.intercept_[0]
                z = w * scores + b
                p_unnorm = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            elif self.method == 'isotonic':
                # 等張回帰
                p_unnorm = self.model.predict(scores)
            else:
                p_unnorm = p_raw
        else:
            # フォールバック (パラメータ近似によるPlatt Scaling)
            if self.method == 'platt':
                z = self.fallback_coef * (scores - np.mean(scores)) + self.fallback_intercept
                p_unnorm = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            else:
                p_unnorm = p_raw
                
        # 3. 6艇の合計が1.0となるよう正規化
        p_unnorm = np.maximum(p_unnorm, 1e-6)
        sum_p = np.sum(p_unnorm)
        if sum_p <= 0:
            p_norm = np.ones(len(boats)) / len(boats)
        else:
            p_norm = p_unnorm / sum_p
            
        return {boats[i]: float(p_norm[i]) for i in range(len(boats))}

    def save(self, filepath=CALIBRATOR_MODEL_PATH):
        """モデルをディスクに保存"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'method': self.method,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'fallback_coef': self.fallback_coef,
            'fallback_intercept': self.fallback_intercept
        }, filepath)

    @classmethod
    def load(cls, filepath=CALIBRATOR_MODEL_PATH):
        """保存されたモデルを読み込み"""
        if not os.path.exists(filepath):
            return cls(method='platt')
        try:
            data = joblib.load(filepath)
            calibrator = cls(method=data.get('method', 'platt'))
            calibrator.model = data.get('model')
            calibrator.is_fitted = data.get('is_fitted', False)
            calibrator.fallback_coef = data.get('fallback_coef', 0.85)
            calibrator.fallback_intercept = data.get('fallback_intercept', -0.05)
            return calibrator
        except Exception:
            return cls(method='platt')


# シングルトンまたはデフォルトキャリブレーターの管理
_DEFAULT_CALIBRATOR = None

def get_default_calibrator(method='platt'):
    global _DEFAULT_CALIBRATOR
    if _DEFAULT_CALIBRATOR is None or _DEFAULT_CALIBRATOR.method != method:
        if os.path.exists(CALIBRATOR_MODEL_PATH):
            _DEFAULT_CALIBRATOR = BoatRaceCalibrator.load(CALIBRATOR_MODEL_PATH)
        else:
            _DEFAULT_CALIBRATOR = BoatRaceCalibrator(method=method)
    return _DEFAULT_CALIBRATOR

def load_probability_config():
    """最適化された設定 (d2, d3, calibration_method 等) を読み込む"""
    default_config = {
        'calibration_method': 'platt',
        'd2': 0.85,
        'd3': 0.65,
        'p1_th': 0.49,
        'gap_th': 0.010
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception:
            pass
    return default_config

def save_probability_config(config_dict):
    """最適化された設定を保存"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


BENTER_CLUSTER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'benter_cluster_config.json')
_DEFAULT_CLUSTER_CONFIG = None

def load_benter_cluster_config(filepath=BENTER_CLUSTER_CONFIG_PATH):
    """クラスタ別Benter最適化設定 (app_data/benter_cluster_config.json) を読み込む"""
    global _DEFAULT_CLUSTER_CONFIG
    default_config = {
        'clusters': {
            'cluster_0': {'name': 'イン超強水面', 'venues': [18, 21, 23, 24], 'd2': 0.50, 'd3': 0.75},
            'cluster_1': {'name': '難水面・イン受難', 'venues': [2, 3, 4, 14, 22], 'd2': 0.20, 'd3': 0.20},
            'cluster_2': {'name': '標準水面', 'venues': 'others', 'd2': 0.10, 'd3': 0.25}
        }
    }
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception:
            pass
    _DEFAULT_CLUSTER_CONFIG = default_config
    return default_config

def get_cluster_benter_params(venue_code, config=None):
    """
    会場コード (1〜24) から所属クラスタ (0, 1, 2) と最適化された (d2, d3, cluster_id, cluster_name) を取得する。
    """
    global _DEFAULT_CLUSTER_CONFIG
    if config is None:
        if _DEFAULT_CLUSTER_CONFIG is None:
            config = load_benter_cluster_config()
        else:
            config = _DEFAULT_CLUSTER_CONFIG

    c0_venues = set(config.get('clusters', {}).get('cluster_0', {}).get('venues', [18, 21, 23, 24]))
    c1_venues = set(config.get('clusters', {}).get('cluster_1', {}).get('venues', [2, 3, 4, 14, 22]))

    try:
        v_int = int(venue_code)
    except (ValueError, TypeError):
        v_int = 1

    if v_int in c0_venues:
        c_info = config.get('clusters', {}).get('cluster_0', {})
        return float(c_info.get('d2', 0.50)), float(c_info.get('d3', 0.75)), 0, c_info.get('name', 'イン超強水面')
    elif v_int in c1_venues:
        c_info = config.get('clusters', {}).get('cluster_1', {})
        return float(c_info.get('d2', 0.20)), float(c_info.get('d3', 0.20)), 1, c_info.get('name', '難水面・イン受難')
    else:
        c_info = config.get('clusters', {}).get('cluster_2', {})
        return float(c_info.get('d2', 0.10)), float(c_info.get('d3', 0.25)), 2, c_info.get('name', '標準水面')


def calculate_benter_probs(honmei_scores_dict, d2=1.0, d3=1.0, calibration_method='platt', calibrator=None):
    """
    本命スコアからキャリブレーションおよびBenterモデル (Damping Factor: d2, d3) を用いて
    全120通りの3連単確率を算出する。

    数式:
      P(i, j, k) = P(i) * [P(j)^d2 / sum_{m!=i} P(m)^d2] * [P(k)^d3 / sum_{m!=i,j} P(m)^d3]
    
    引数:
      honmei_scores_dict: {boat_number: score} 辞書
      d2: 2着の減衰パラメーター (デフォルト: 1.0 = ハルビル公式)
      d3: 3着の減衰パラメーター (デフォルト: 1.0 = ハルビル公式)
      calibration_method: 'platt', 'isotonic', 'softmax', 'direct'
      calibrator: BoatRaceCalibrator インスタンス (省略時はデフォルト)

    戻り値:
      pl_probs: [{'combo': '1-2-3', 'prob': 0.085}, ...] (降順ソート済みリスト)
      max_p1: 1着最高確率 (float)
      prob_gap: 1位と2位の3連単確率差 (float)
    """
    if not honmei_scores_dict:
        return [], 0.0, 0.0

    boats = list(honmei_scores_dict.keys())
    
    # 1. 確率キャリブレーション (1着確率 P(i) の算出)
    if calibration_method in ('direct', 'identity'):
        scores_arr = np.array([honmei_scores_dict[b] for b in boats], dtype=float)
        p_norm = scores_arr / np.maximum(np.sum(scores_arr), 1e-9)
        p1_dict = {boats[i]: float(p_norm[i]) for i in range(len(boats))}
    else:
        if calibrator is None:
            if calibration_method is not None:
                calibrator = get_default_calibrator(method=calibration_method)
            else:
                calibrator = get_default_calibrator()
        p1_dict = calibrator.calibrate_scores(honmei_scores_dict)
    
    # 2. Benterモデル用のべき乗確率辞書を事前計算
    # 浮動小数点数アンダーフロー防止のためクリップ
    p1_d2 = {b: max(p1_dict[b], 1e-9) ** d2 for b in boats}
    p1_d3 = {b: max(p1_dict[b], 1e-9) ** d3 for b in boats}
    
    # 3. 全120通りの3連単確率計算
    combos = list(itertools.permutations(boats, 3))
    benter_probs = []
    
    for c in combos:
        b1, b2, b3 = c
        prob1 = p1_dict[b1]
        
        # 2着条件付き確率 (Benter Damping)
        denom2 = sum(p1_d2[b] for b in boats if b != b1)
        prob2 = p1_d2[b2] / denom2 if denom2 > 0 else 1e-9
        
        # 3着条件付き確率 (Benter Damping)
        denom3 = sum(p1_d3[b] for b in boats if b != b1 and b != b2)
        prob3 = p1_d3[b3] / denom3 if denom3 > 0 else 1e-9
        
        total_prob = prob1 * prob2 * prob3
        combo_str = f"{b1}-{b2}-{b3}"
        benter_probs.append({'combo': combo_str, 'prob': float(total_prob)})
        
    # 確率降順にソート
    benter_probs.sort(key=lambda x: x['prob'], reverse=True)
    
    # フィルター指標用
    p1_sorted = sorted(p1_dict.items(), key=lambda x: x[1], reverse=True)
    max_p1 = float(p1_sorted[0][1]) if p1_sorted else 0.0
    prob_gap = float(benter_probs[0]['prob'] - benter_probs[1]['prob']) if len(benter_probs) >= 2 else 0.0
    
    return benter_probs, max_p1, prob_gap

def calculate_plackett_luce_probs(honmei_scores_dict, d2=1.0, d3=1.0, calibration_method='platt', calibrator=None):
    """
    既存コードとの後方互換用関数。
    内部で calculate_benter_probs を呼び出す。
    """
    return calculate_benter_probs(honmei_scores_dict, d2=d2, d3=d3, calibration_method=calibration_method, calibrator=calibrator)

