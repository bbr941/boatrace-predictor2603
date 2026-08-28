"""
odds_normalizer.py
オッズデータの控除率排除・正規化確率変換およびLightGBM用ベースマージン (init_score) ロジット生成モジュール
"""

from typing import Union, List, Dict, Optional, Sequence
import numpy as np
import pandas as pd

CLIP_EPS_DEFAULT = 1e-5
DEFAULT_PROB_EQUAL = 1.0 / 6.0
DEFAULT_LOGIT_EQUAL = float(np.log(DEFAULT_PROB_EQUAL / (1.0 - DEFAULT_PROB_EQUAL)))  # ln(1/5) ≈ -1.609438


def odds_to_normalized_probs(
    odds: Union[Sequence[float], np.ndarray, pd.Series, Dict[int, float]],
    n_boats: int = 6,
    default_prob: float = DEFAULT_PROB_EQUAL
) -> np.ndarray:
    """
    6艇の単勝オッズ配列から控除率（Overround）を排除した正規化勝率 P_norm を算出する。
    総和は厳密に 1.0 となる。

    引数:
      odds: 単勝オッズの配列 (長さ6のリスト、ndarray、Series、または {艇番: オッズ} 辞書)
      n_boats: レース出走艇数 (デフォルト: 6)
      default_prob: 欠損時のデフォルト確率 (デフォルト: 1/6)

    戻り値:
      np.ndarray (shape=(n_boats,), float64): 合計 1.0 の正規化確率
    """
    # 辞書型・Seriesからの配列変換
    if isinstance(odds, dict):
        # 1-indexed (1~6) または 0-indexed (0~5) に対応
        keys = sorted(odds.keys())
        if all(k in odds for k in range(1, n_boats + 1)):
            raw_odds = [odds[k] for k in range(1, n_boats + 1)]
        elif all(k in odds for k in range(n_boats)):
            raw_odds = [odds[k] for k in range(n_boats)]
        else:
            raw_odds = [odds.get(k, np.nan) for k in range(1, n_boats + 1)]
    elif isinstance(odds, pd.Series):
        raw_odds = odds.values
    else:
        raw_odds = list(odds)

    # 長さチェックとNumPy配列化
    arr = np.array(raw_odds, dtype=np.float64)
    if len(arr) != n_boats:
        # 長さが異なる場合の安全パディング/スライス
        padded = np.full(n_boats, np.nan, dtype=np.float64)
        m = min(len(arr), n_boats)
        padded[:m] = arr[:m]
        arr = padded

    # 有効なオッズ (> 0 かつ 有限数) の判定
    valid_mask = np.isfinite(arr) & (arr > 0.0)

    # 全艇欠損の場合 -> 均等確率 (1/6)
    if not np.any(valid_mask):
        return np.full(n_boats, 1.0 / n_boats, dtype=np.float64)

    # 逆数 q = 1 / O の計算
    raw_implied = np.zeros(n_boats, dtype=np.float64)
    raw_implied[valid_mask] = 1.0 / arr[valid_mask]

    # 一部欠損艇がある場合: 有効艇の平均逆数または均等確率で安全に補完
    if not np.all(valid_mask):
        mean_valid_implied = np.mean(raw_implied[valid_mask])
        fallback_val = mean_valid_implied if mean_valid_implied > 0 else default_prob
        raw_implied[~valid_mask] = fallback_val

    # 和で除算して正規化 (P_norm = q / sum(q))
    total_implied = np.sum(raw_implied)
    if total_implied <= 0 or not np.isfinite(total_implied):
        return np.full(n_boats, 1.0 / n_boats, dtype=np.float64)

    p_norm = raw_implied / total_implied

    # 数値誤差補正 (厳密に合計 1.0)
    p_norm = p_norm / np.sum(p_norm)
    return p_norm


def probs_to_init_scores(
    probs: Union[Sequence[float], np.ndarray],
    clip_eps: float = CLIP_EPS_DEFAULT
) -> np.ndarray:
    """
    確率配列 P_norm を [eps, 1-eps] でクリップし、ロジット変換 z = log(P / (1 - P)) を行う。
    LightGBMの init_score (ベースマージン) として利用可能。

    引数:
      probs: 確率配列 (各要素 0 <= P <= 1)
      clip_eps: ゼロ除算・無限大防止用のクリッピング閾値 (デフォルト: 1e-5)

    戻り値:
      np.ndarray (float64): ロジット値配列
    """
    arr = np.array(probs, dtype=np.float64)
    # NaN/Inf 対策
    arr = np.nan_to_num(arr, nan=DEFAULT_PROB_EQUAL, posinf=1.0 - clip_eps, neginf=clip_eps)

    # [eps, 1 - eps] でクリップ
    p_clipped = np.clip(arr, clip_eps, 1.0 - clip_eps)

    # ロジット変換: log(p / (1 - p))
    logits = np.log(p_clipped / (1.0 - p_clipped))
    return logits


def odds_to_init_scores(
    odds: Union[Sequence[float], np.ndarray, pd.Series, Dict[int, float]],
    n_boats: int = 6,
    clip_eps: float = CLIP_EPS_DEFAULT
) -> np.ndarray:
    """
    オッズ配列から直接、正規化確率を経由してLightGBM用ベースマージン (init_score) ロジット配列を算出する。

    引数:
      odds: 単勝オッズ (配列 / 辞書)
      n_boats: 出走艇数 (デフォルト: 6)
      clip_eps: クリップ閾値 (デフォルト: 1e-5)

    戻り値:
      np.ndarray (shape=(n_boats,), float64): ロジット配列
    """
    p_norm = odds_to_normalized_probs(odds, n_boats=n_boats)
    return probs_to_init_scores(p_norm, clip_eps=clip_eps)


class OddsNormalizer:
    """
    オッズ正規化およびベースマージン変換ユーティリティクラス
    """

    def __init__(self, clip_eps: float = CLIP_EPS_DEFAULT, n_boats: int = 6):
        self.clip_eps = clip_eps
        self.n_boats = n_boats

    def normalize_odds(self, odds: Union[Sequence[float], np.ndarray, Dict[int, float]]) -> np.ndarray:
        return odds_to_normalized_probs(odds, n_boats=self.n_boats)

    def to_init_score(self, odds: Union[Sequence[float], np.ndarray, Dict[int, float]]) -> np.ndarray:
        return odds_to_init_scores(odds, n_boats=self.n_boats, clip_eps=self.clip_eps)

    def batch_to_init_scores(self, odds_matrix: np.ndarray) -> np.ndarray:
        """
        複数レースのオッズ行列 (shape: [N_races, 6]) を一括で init_score 行列 (shape: [N_races, 6]) に変換する。
        """
        odds_arr = np.asarray(odds_matrix, dtype=np.float64)
        n_races, n_b = odds_arr.shape
        
        # 逆数計算 (0や負値、NaN対策)
        valid = np.isfinite(odds_arr) & (odds_arr > 0.0)
        raw_implied = np.zeros_like(odds_arr)
        raw_implied[valid] = 1.0 / odds_arr[valid]
        
        # 行ごとの和
        row_sums = np.sum(raw_implied, axis=1, keepdims=True)
        # 全艇欠損行のケア
        zero_rows = (row_sums.squeeze() <= 0) | (~np.isfinite(row_sums.squeeze()))
        
        # 安全な正規化
        safe_sums = np.where(zero_rows[:, None], 1.0, row_sums)
        p_norm = np.where(zero_rows[:, None], 1.0 / n_b, raw_implied / safe_sums)
        
        # クリップ & ロジット
        p_clipped = np.clip(p_norm, self.clip_eps, 1.0 - self.clip_eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        return logits
