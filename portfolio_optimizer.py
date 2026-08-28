"""
portfolio_optimizer.py
Markowitzの平均分散アプローチ (Mean-Variance Optimization / Simultaneous Kelly) に基づく
3連単120通りの最適資金配分モジュール
"""

import os
import itertools
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy.optimize import minimize

CORRELATION_MASK_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'correlation_mask.npy')

# 全120通りの3連単買い目順序 (generate_correlation_matrix.py と同一順序)
ALL_COMBINATIONS_TUPLES = list(itertools.permutations(range(1, 7), 3))
ALL_COMBINATIONS = [f"{c[0]}-{c[1]}-{c[2]}" for c in ALL_COMBINATIONS_TUPLES]
COMBO_TO_INDEX: Dict[str, int] = {c: idx for idx, c in enumerate(ALL_COMBINATIONS)}


def _normalize_combo_key(key: Any) -> str:
    """
    買い目キーの正規化 (例: '123' -> '1-2-3', (1, 2, 3) -> '1-2-3')
    """
    if isinstance(key, (list, tuple)) and len(key) == 3:
        return f"{key[0]}-{key[1]}-{key[2]}"
    key_str = str(key).strip()
    if len(key_str) == 3 and '-' not in key_str and key_str.isdigit():
        return f"{key_str[0]}-{key_str[1]}-{key_str[2]}"
    return key_str


def load_correlation_mask(filepath: str = CORRELATION_MASK_PATH) -> np.ndarray:
    """
    静的相関行列 (120x120) を読み込む。存在しない場合は自動生成する。
    """
    if os.path.exists(filepath):
        try:
            mask = np.load(filepath)
            if mask.shape == (120, 120):
                return mask
        except Exception:
            pass
    
    # 存在しない場合はオンデマンド生成
    try:
        from generate_correlation_matrix import generate_matrix
        mask, _ = generate_matrix()
        return mask
    except Exception:
        # フォールバック (単位行列)
        return np.eye(120, dtype=np.float64)


class PortfolioOptimizer:
    """
    Markowitz平均分散モデルによる単一レース内最適資金配分エンジン
    """

    def __init__(self, correlation_mask: Optional[np.ndarray] = None):
        if correlation_mask is not None and correlation_mask.shape == (120, 120):
            self.correlation_mask = correlation_mask
        else:
            self.correlation_mask = load_correlation_mask()

    def optimize_funds(
        self,
        probabilities: Dict[str, float],
        odds: Dict[str, float],
        bankroll: float = 100000.0,
        risk_aversion: float = 1.0,
        max_exposure: float = 0.05,
        max_concentration: float = 0.02,
        min_ev: float = 1.25,
        max_odds: float = 30.0,
        kelly_fraction: Optional[float] = 0.25,
    ) -> Dict[str, int]:
        """
        1レース内の最適資金配分を計算し、100円単位の購入金額辞書を返す。

        引数:
          probabilities: {買い目文字列 ('1-2-3'): Benter確率 (float)}
          odds: {買い目文字列 ('1-2-3'): 実オッズ (float)}
          bankroll: 総資金 (デフォルト: 100,000円)
          risk_aversion: リスク回避度 λ (デフォルト: 1.0)
          max_exposure: 1レースの最大投資ウェイト上限 (デフォルト: 0.05 = 5%)
          max_concentration: 1買い目あたりの最大投資ウェイト上限 (デフォルト: 0.02 = 2%)
          min_ev: 事前絞り込みの最小期待値閾値 (デフォルト: 1.25)
          max_odds: 事前絞り込みの最大オッズ上限 (デフォルト: 30.0)
          kelly_fraction: Fractional Kelly係数 (デフォルト: 0.25 = クォーター・ケリー)

        戻り値:
          {買い目: 投資金額(円)} (例: {'1-2-3': 1200, '1-2-4': 800})
        """
        # 1. キー正規化
        norm_probs = {_normalize_combo_key(k): float(v) for k, v in probabilities.items()}
        norm_odds = {_normalize_combo_key(k): float(v) for k, v in odds.items()}

        # 2. 【事前絞り込み】 EV >= min_ev (1.25) かつ odds <= max_odds (30.0) の候補を抽出
        candidate_combos: List[str] = []
        candidate_indices: List[int] = []
        p_list: List[float] = []
        o_list: List[float] = []
        ev_list: List[float] = []
        kelly_list: List[float] = []

        for combo in ALL_COMBINATIONS:
            if combo in norm_probs and combo in norm_odds:
                p = norm_probs[combo]
                o = norm_odds[combo]
                ev = p * o
                if ev >= min_ev and o <= max_odds and o > 1.0 and p > 0:
                    candidate_combos.append(combo)
                    candidate_indices.append(COMBO_TO_INDEX[combo])
                    p_list.append(p)
                    o_list.append(o)
                    ev_list.append(ev)
                    # ケリー基準: f_i = (EV_i - 1) / (Odds_i - 1)
                    f_i = (ev - 1.0) / max(o - 1.0, 1e-6)
                    kelly_list.append(max(f_i, 0.0))

        k = len(candidate_combos)
        if k == 0:
            return {}

        # 【Fractional Kelly による動的総ウェイト上限の算出】
        if kelly_fraction is not None and kelly_fraction > 0:
            f_total = sum(kelly_list)
            # f_total * kelly_fraction を計算し、上限0.10 (10%) でクリップ
            effective_max_exposure = min(f_total * kelly_fraction, 0.10)
            if effective_max_exposure <= 1e-4:
                return {}
        else:
            effective_max_exposure = max_exposure

        # 3. 【共分散行列の動的構築】
        # P_i, σ_i = sqrt(P_i * (1 - P_i))
        p_arr = np.array(p_list, dtype=np.float64)
        o_arr = np.array(o_list, dtype=np.float64)
        ev_arr = np.array(ev_list, dtype=np.float64)
        sigma_arr = np.sqrt(np.maximum(p_arr * (1.0 - p_arr), 1e-9))

        # 静的相関行列から候補の部分行列を抽出
        idx_arr = np.array(candidate_indices, dtype=int)
        corr_sub = self.correlation_mask[np.ix_(idx_arr, idx_arr)]

        # Σ_ij = σ_i * σ_j * C_ij
        cov_matrix = (sigma_arr[:, None] * sigma_arr[None, :]) * corr_sub
        # 数値安定化用の微小正則化 (Jitter)
        cov_matrix += np.eye(k) * 1e-8

        # 4. 【最適化 (SLSQP)】
        # 目的関数: f(w) = - (w^T * EV - 0.5 * λ * w^T * Σ * w)
        # 勾配: ∇f(w) = - EV + λ * Σ * w
        def objective(w: np.ndarray) -> float:
            port_ev = np.dot(w, ev_arr)
            port_var = np.dot(w, np.dot(cov_matrix, w))
            return float(-(port_ev - 0.5 * risk_aversion * port_var))

        def gradient(w: np.ndarray) -> np.ndarray:
            return -ev_arr + risk_aversion * np.dot(cov_matrix, w)

        # 制約条件: sum(w) <= effective_max_exposure
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda w: effective_max_exposure - np.sum(w),
                'jac': lambda w: -np.ones_like(w)
            }
        ]

        # 境界条件: 0 <= w_i <= effective_max_exposure (単一買い目上限)
        bound_cap = min(max_concentration, effective_max_exposure) if max_concentration is not None else effective_max_exposure
        bounds = [(0.0, bound_cap) for _ in range(k)]

        # 初期値: 均等割り
        initial_w = np.full(k, min(bound_cap, effective_max_exposure / max(k, 1)), dtype=np.float64)

        opt_result = minimize(
            fun=objective,
            x0=initial_w,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-7}
        )

        weights = np.maximum(opt_result.x, 0.0) if opt_result.success or opt_result.x is not None else initial_w

        # 5. 【離散化とトリガミ排除】
        # 100円単位で切り捨て
        raw_amounts = weights * bankroll
        bets: Dict[str, int] = {}
        for i in range(k):
            amt = int(raw_amounts[i] // 100) * 100
            if amt >= 100:
                bets[candidate_combos[i]] = amt

        if not bets:
            return {}

        # トリガミ排除ループ (的中時の払戻金 < 総投資額 となる買い目を排除)
        while bets:
            total_invest = sum(bets.values())
            trigami_found = False
            for combo, amt in list(bets.items()):
                expected_payout = amt * norm_odds[combo]
                if expected_payout < total_invest:
                    del bets[combo]
                    trigami_found = True

            if not trigami_found or not bets:
                break

        return bets



# モジュールレベルの関数インターフェース
_DEFAULT_OPTIMIZER: Optional[PortfolioOptimizer] = None


def optimize_portfolio(
    probabilities: Dict[str, float],
    odds: Dict[str, float],
    correlation_mask: Optional[np.ndarray] = None,
    bankroll: float = 100000.0,
    risk_aversion: float = 1.0,
    max_exposure: float = 0.05,
    max_concentration: float = 0.02,
    min_ev: float = 1.25,
    max_odds: float = 30.0,
    kelly_fraction: Optional[float] = 0.25,
) -> Dict[str, int]:
    """
    Markowitz平均分散アプローチによる1レース内の最適資金配分関数。

    引数:
      probabilities: dict (買い目 -> Benter確率)
      odds: dict (買い目 -> 実オッズ)
      correlation_mask: 120x120の静的相関行列 (NumPy array, 省略時は自動ロード)
      bankroll: 総資金 (デフォルト: 100,000)
      risk_aversion: リスク回避度 λ (デフォルト: 1.0)
      max_exposure: 1レースの最大投資ウェイト上限 (デフォルト: 0.05)
      max_concentration: 1買い目あたりの最大投資ウェイト上限 (デフォルト: 0.02)
      min_ev: 期待値閾値 (デフォルト: 1.25)
      max_odds: 最大オッズ上限 (デフォルト: 30.0)
      kelly_fraction: Fractional Kelly係数 (デフォルト: 0.25)

    戻り値:
      dict: {買い目: 投資金額(円)} (トリガミ排除・100円単位離散化済み)
    """
    global _DEFAULT_OPTIMIZER
    if correlation_mask is not None:
        optimizer = PortfolioOptimizer(correlation_mask=correlation_mask)
    else:
        if _DEFAULT_OPTIMIZER is None:
            _DEFAULT_OPTIMIZER = PortfolioOptimizer()
        optimizer = _DEFAULT_OPTIMIZER

    return optimizer.optimize_funds(
        probabilities=probabilities,
        odds=odds,
        bankroll=bankroll,
        risk_aversion=risk_aversion,
        max_exposure=max_exposure,
        max_concentration=max_concentration,
        min_ev=min_ev,
        max_odds=max_odds,
        kelly_fraction=kelly_fraction,
    )



if __name__ == "__main__":
    import time
    print("=== portfolio_optimizer.py 単体動作テスト ===")

    # テスト用ダミーデータ生成
    dummy_probs: Dict[str, float] = {}
    dummy_odds: Dict[str, float] = {}

    # サンプルとしていくつかの買い目に有意な確率・オッズを設定
    test_cases = [
        ("1-2-3", 0.12, 12.5),  # EV = 1.50
        ("1-2-4", 0.09, 14.0),  # EV = 1.26
        ("1-3-2", 0.08, 15.0),  # EV = 1.20
        ("2-1-3", 0.06, 22.0),  # EV = 1.32
        ("3-1-2", 0.04, 35.0),  # EV = 1.40
        ("1-4-5", 0.02, 40.0),  # EV = 0.80 (EV < 1.0 除外対象)
        ("4-5-6", 0.01, 80.0),  # EV = 0.80 (除外対象)
    ]
    for c, p, o in test_cases:
        dummy_probs[c] = p
        dummy_odds[c] = o

    optimizer = PortfolioOptimizer()

    t0 = time.time()
    bets = optimizer.optimize_funds(
        probabilities=dummy_probs,
        odds=dummy_odds,
        bankroll=100000.0,
        risk_aversion=1.0,
        max_exposure=0.05,      # 最大投資: 5,000円
        max_concentration=0.02, # 1点最大: 2,000円
        min_ev=1.0
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    print(f"計算所要時間: {elapsed_ms:.2f} ms")
    print(f"最適配分結果: {bets}")
    total_bet = sum(bets.values())
    print(f"総投資額: {total_bet} 円")
    
    for c, amt in bets.items():
        payout = amt * dummy_odds[c]
        profit = payout - total_bet
        print(f"  買い目 {c}: 投資 {amt}円 | オッズ {dummy_odds[c]:.1f}倍 | 払戻 {int(payout)}円 (利益: {int(profit):+d}円)")

    print("\nテスト完了！")
