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


def calculate_dynamic_min_ev(
    odds: float,
    base_min_ev: float = 1.25,
    use_dynamic_ev: bool = True,
    low_odds_threshold: float = 10.0,
    low_ev_min: float = 1.10,
    high_odds_threshold: float = 20.0,
    high_ev_slope: float = 0.015,
    max_ev_cap: float = 1.40
) -> float:
    """
    オッズ連動型動的EV閾値（分散連動型カットオフ）の算出。
    - 低オッズ (odds < 10.0): 分散リスクが低いため閾値を緩和 (例: 1.10 〜 1.25)
    - 中オッズ (10.0 <= odds < 20.0): 基準EV (例: 1.25)
    - 高オッズ (odds >= 20.0): 外れ続ける分散リスクが高いため閾値を厳格化 (例: 1.25 〜 1.40)
    """
    if not use_dynamic_ev or odds <= 0:
        return base_min_ev

    if odds < low_odds_threshold:
        # odds: 1.0 -> low_ev_min (1.10), odds: 10.0 -> base_min_ev (1.25)
        ratio = max(0.0, (odds - 1.0) / max(low_odds_threshold - 1.0, 1e-6))
        return low_ev_min + (base_min_ev - low_ev_min) * ratio
    elif odds < high_odds_threshold:
        return base_min_ev
    else:
        # odds: 20.0 -> 1.25, odds: 30.0 -> 1.40
        additional_ev = (odds - high_odds_threshold) * high_ev_slope
        return min(base_min_ev + additional_ev, max_ev_cap)


class FractionalRoundingPool:
    """
    100円未満の端数切り捨てロスを合算・プールし、100円に達するごとにEV最上位へ再配分するマネージャー
    """
    def __init__(self, carryover: bool = False):
        self.carryover = carryover
        self.pool_balance: float = 0.0

    def allocate_with_pool(
        self,
        raw_amounts: np.ndarray,
        candidate_combos: List[str],
        ev_arr: np.ndarray,
        p_arr: np.ndarray,
        max_single_bet_cap: float = float('inf'),
        min_base_amount: int = 100
    ) -> Dict[str, int]:
        """
        端数プールを合算し、100円単位で高EV買い目へ+100円を繰り上げ配分
        """
        if not self.carryover:
            self.pool_balance = 0.0

        k = len(candidate_combos)
        if k == 0:
            return {}

        base_amts = np.floor(raw_amounts / 100.0) * 100.0
        remainders = raw_amounts - base_amts
        
        # 端数をプールに加算
        self.pool_balance += float(np.sum(remainders))
        
        # 基礎配分額
        bets: Dict[str, int] = {}
        for i in range(k):
            if base_amts[i] >= min_base_amount:
                bets[candidate_combos[i]] = int(base_amts[i])

        # EV 降順（第2キー: 確率降順）で候補インデックスをソート
        sort_indices = sorted(
            range(k),
            key=lambda i: (ev_arr[i], p_arr[i]),
            reverse=True
        )

        # プール残高が100円以上ある限り、EV上位の買い目に+100円ずつ再投下
        idx_ptr = 0
        allocated_rounds = 0
        max_rounds = len(sort_indices) * 10  # 無限ループ防止
        
        while self.pool_balance >= 100.0 and len(sort_indices) > 0 and allocated_rounds < max_rounds:
            target_idx = sort_indices[idx_ptr % len(sort_indices)]
            target_combo = candidate_combos[target_idx]
            current_amt = bets.get(target_combo, 0)
            
            if current_amt + 100 <= max_single_bet_cap:
                bets[target_combo] = current_amt + 100
                self.pool_balance -= 100.0
            
            idx_ptr += 1
            allocated_rounds += 1

        return bets


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

    def __init__(self, correlation_mask: Optional[np.ndarray] = None, carryover_pool: bool = False):
        if correlation_mask is not None and correlation_mask.shape == (120, 120):
            self.correlation_mask = correlation_mask
        else:
            self.correlation_mask = load_correlation_mask()
        self.fractional_pool = FractionalRoundingPool(carryover=carryover_pool)


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
        use_dynamic_ev: bool = True,
        use_fractional_pool: bool = True,
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
          min_ev: 事前絞り込みの基準期待値閾値 (デフォルト: 1.25)
          max_odds: 事前絞り込みの最大オッズ上限 (デフォルト: 30.0)
          kelly_fraction: Fractional Kelly係数 (デフォルト: 0.25 = クォーター・ケリー)
          use_dynamic_ev: オッズ連動型動的EV閾値の適用有無 (デフォルト: True)
          use_fractional_pool: 100円未満端数プール再配分の適用有無 (デフォルト: True)

        戻り値:
          {買い目: 投資金額(円)} (例: {'1-2-3': 1200, '1-2-4': 800})
        """
        # 1. キー正規化
        norm_probs = {_normalize_combo_key(k): float(v) for k, v in probabilities.items()}
        norm_odds = {_normalize_combo_key(k): float(v) for k, v in odds.items()}

        # 2. 【事前絞り込み】 オッズ連動型動的EV (または一律 min_ev) かつ odds <= max_odds の候補を抽出
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
                required_min_ev = calculate_dynamic_min_ev(o, base_min_ev=min_ev, use_dynamic_ev=use_dynamic_ev)
                if ev >= required_min_ev and o <= max_odds and o > 1.0 and p > 0:
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

        # 5. 【離散化と端数プール再分配 & トリガミ排除】
        raw_amounts = weights * bankroll
        max_single_cap = bound_cap * bankroll

        if use_fractional_pool:
            bets = self.fractional_pool.allocate_with_pool(
                raw_amounts=raw_amounts,
                candidate_combos=candidate_combos,
                ev_arr=ev_arr,
                p_arr=p_arr,
                max_single_bet_cap=max_single_cap
            )
        else:
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
    use_dynamic_ev: bool = True,
    use_fractional_pool: bool = True,
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
      use_dynamic_ev: オッズ連動型動的EV閾値の適用有無 (デフォルト: True)
      use_fractional_pool: 100円未満端数プール再配分の適用有無 (デフォルト: True)

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
        use_dynamic_ev=use_dynamic_ev,
        use_fractional_pool=use_fractional_pool,
    )


def calculate_dutching_bets(
    benter_probs: Dict[str, float],
    odds_dict: Dict[str, float],
    budget: int = 1000,
    target_cum_prob: float = 0.50,
    max_combos: int = 8,
    min_combos: int = 2,
    min_synthetic_odds: float = 2.5
) -> Dict[str, int]:
    """
    的中特化: 累積確率50%（最大8点）を勝率順に抽出し、
    合成オッズ >= min_synthetic_odds (デフォルト2.5倍) の制約下で
    オッズ逆数比によるダッチング資金配分とトリガミ回避ループを実行
    """
    valid_combos = []
    for combo, prob in sorted(benter_probs.items(), key=lambda x: x[1], reverse=True):
        o = odds_dict.get(combo, 0.0)
        if o > 1.0 and prob > 0:
            valid_combos.append((combo, prob, o))
            
    if not valid_combos:
        return {}
        
    selected = []
    cum_p = 0.0
    inv_sum = 0.0
    max_inv_sum = (1.0 / min_synthetic_odds) if min_synthetic_odds > 0 else 0.95
    
    for item in valid_combos:
        c_name, c_prob, c_odds = item
        next_inv_sum = inv_sum + (1.0 / c_odds)
        
        # 既に1点以上あり、追加すると合成オッズが min_synthetic_odds を下回る場合は追加をストップ
        if selected and next_inv_sum > max_inv_sum:
            if len(selected) >= min_combos:
                break
            elif len(selected) + 1 >= min_combos and next_inv_sum <= 0.90:
                selected.append(item)
                cum_p += c_prob
                inv_sum = next_inv_sum
                break
            else:
                break
                
        selected.append(item)
        cum_p += c_prob
        inv_sum = next_inv_sum
        
        if (cum_p >= target_cum_prob or len(selected) >= max_combos) and len(selected) >= min_combos:
            break
            
    if len(selected) < min_combos:
        selected = valid_combos[:min(len(valid_combos), min_combos)]
        
    while len(selected) > min_combos:
        s_val = sum(1.0 / o for _, _, o in selected)
        if s_val <= max_inv_sum:
            break
        selected.pop()
        
    def allocate(combos_list, target_budget):
        s = sum(1.0 / o for _, _, o in combos_list)
        if s <= 0:
            return {c: 100 for c, _, _ in combos_list}
        res = {}
        for c, _, o in combos_list:
            raw = target_budget * ((1.0 / o) / s)
            amt = max(100, int(round(raw / 100.0)) * 100)
            res[c] = amt
        return res
        
    bets = allocate(selected, budget)
    
    for _ in range(5):
        tot = sum(bets.values())
        trigami_found = False
        for c, _, o in list(selected):
            payout = bets.get(c, 0) * o
            if payout <= tot:
                trigami_found = True
                if (bets[c] + 100) * o > (tot + 100):
                    bets[c] += 100
                elif len(selected) > min_combos:
                    selected.pop()
                    bets = allocate(selected, budget)
                    break
        if not trigami_found:
            break
            
    return bets


if __name__ == "__main__":
    import time
    print("=== portfolio_optimizer.py 単体動作テスト (動的EV & 端数プール) ===")

    # テスト用ダミーデータ生成
    dummy_probs: Dict[str, float] = {}
    dummy_odds: Dict[str, float] = {}

    # サンプルとしていくつかの買い目に確率・オッズを設定
    test_cases = [
        ("1-2-3", 0.15, 8.0),   # EV = 1.20 (低オッズ8.0倍: 動的EV 1.217 -> 除外または通過境界)
        ("1-2-4", 0.18, 6.5),   # EV = 1.17 (低オッズ6.5倍: 動的EV 1.192 -> 1.17 < 1.192)
        ("1-3-2", 0.20, 6.0),   # EV = 1.20 (低オッズ6.0倍: 動的EV 1.183 -> 1.20 >= 1.183 通過！)
        ("2-1-3", 0.08, 16.0),  # EV = 1.28 (中オッズ16.0倍: 基準EV 1.25 -> 通過！)
        ("3-1-2", 0.05, 25.0),  # EV = 1.25 (高オッズ25.0倍: 厳格EV 1.325 -> 1.25 < 1.325 除外！)
        ("1-4-5", 0.06, 26.0),  # EV = 1.56 (高オッズ26.0倍: 厳格EV 1.340 -> 1.56 >= 1.340 通過！)
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
        min_ev=1.25,
        use_dynamic_ev=True,
        use_fractional_pool=True
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    print(f"計算所要時間: {elapsed_ms:.2f} ms")
    print(f"最適配分結果: {bets}")
    total_bet = sum(bets.values())
    print(f"総投資額: {total_bet} 円 (端数余剰プール残高: {optimizer.fractional_pool.pool_balance:.1f}円)")
    
    for c, amt in bets.items():
        payout = amt * dummy_odds[c]
        profit = payout - total_bet
        ev = dummy_probs[c] * dummy_odds[c]
        req_ev = calculate_dynamic_min_ev(dummy_odds[c], base_min_ev=1.25)
        print(f"  買い目 {c}: 投資 {amt}円 | オッズ {dummy_odds[c]:.1f}倍 | EV: {ev:.2f} (要求: {req_ev:.2f}) | 払戻 {int(payout)}円 (利益: {int(profit):+d}円)")

    print("\nテスト完了！")

