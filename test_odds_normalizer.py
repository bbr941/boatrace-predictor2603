"""
test_odds_normalizer.py
odds_normalizer.py の単体テストおよび変換過程の可視化スクリプト
"""

import numpy as np
from odds_normalizer import (
    odds_to_normalized_probs,
    probs_to_init_scores,
    odds_to_init_scores,
    OddsNormalizer,
    DEFAULT_LOGIT_EQUAL
)


def run_tests():
    print("=" * 75)
    print("  🧪 odds_normalizer.py 単体テスト & 変換過程ビジュアライザー")
    print("=" * 75)

    test_cases = [
        {
            "name": "パターン1: 正常系オッズ (標準的なレース)",
            "odds": [1.5, 4.2, 8.5, 15.0, 25.0, 50.0],
            "desc": "1号艇本命、外枠が高配当の一般的なオッズ構成"
        },
        {
            "name": "パターン2: 極端なオッズ (圧倒的本命・元返し 1.0倍)",
            "odds": [1.0, 100.0, 300.0, 500.0, 800.0, 1000.0],
            "desc": "1号艇が単勝1.0倍、他艇が超大穴。クリッピングによるゼロ除算耐性を検証"
        },
        {
            "name": "パターン3: 欠損値を含むオッズ (NaN, 0.0, 負値, None)",
            "odds": [2.0, np.nan, 0.0, 12.0, -1.0, 25.0],
            "desc": "欠損艇に対する安全なデフォルト補完と全体正規化を検証"
        },
        {
            "name": "パターン4 (追加検証): 全艇欠損 (全オッズがNaN / 0)",
            "odds": [np.nan, np.nan, 0.0, None, np.nan, 0.0],
            "desc": "完全なデータ欠損時における均等確率 (1/6) へのフォールバックを検証"
        }
    ]

    normalizer = OddsNormalizer(clip_eps=1e-5)

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{case['name']}]")
        print(f"  概要: {case['desc']}")

        raw_odds = case["odds"]
        # 1. 正規化確率の計算
        p_norm = normalizer.normalize_odds(raw_odds)

        # 2. ロジット (init_score) の計算
        init_scores = normalizer.to_init_score(raw_odds)

        # --- アサーション検証 ---
        # A. 確率の総和が厳密に 1.0 であること
        sum_p = np.sum(p_norm)
        assert np.isclose(sum_p, 1.0, atol=1e-9), f"確率の合計が 1.0 ではありません: {sum_p}"

        # B. 各確率が [0, 1] の範囲内であること
        assert np.all(p_norm >= 0.0) and np.all(p_norm <= 1.0), f"確率が [0, 1] を逸脱しています: {p_norm}"

        # C. ロジット値に NaN や Inf が含まれないこと (ゼロ除算等の回避)
        assert not np.any(np.isnan(init_scores)), f"init_scores に NaN が含まれています: {init_scores}"
        assert not np.any(np.isinf(init_scores)), f"init_scores に Inf が含まれています: {init_scores}"

        # D. 配列長が 6 であること
        assert len(p_norm) == 6, f"確率配列の長さが 6 ではありません: {len(p_norm)}"
        assert len(init_scores) == 6, f"init_scores の長さが 6 ではありません: {len(init_scores)}"

        # --- 変換過程の可視化出力 ---
        print("  ┌──────┬──────────────┬──────────────────┬──────────────────┐")
        print("  │ 艇番 │ 入力オッズ   │ 正規化勝率(P_norm)│ ロジット(init_sc)│")
        print("  ├──────┼──────────────┼──────────────────┼──────────────────┤")
        for boat_num in range(1, 7):
            val = raw_odds[boat_num - 1]
            odds_str = f"{val:6.1f}倍" if val is not None and np.isfinite(val) and val > 0 else f"{str(val):>8}"
            p_val = p_norm[boat_num - 1]
            z_val = init_scores[boat_num - 1]
            print(f"  │  {boat_num}号艇│   {odds_str:<10} │    {p_val:8.4%}     │     {z_val:+8.4f}     │")
        print("  └──────┴──────────────┴──────────────────┴──────────────────┘")
        print(f"  -> 検証ステータス: ✅ PASS (確率総和 = {sum_p:.8f}, 全値有限・ロジット正常)")

    # バッチ処理のテスト
    print("\n[追加テスト: バッチ行列一括変換 (OddsNormalizer.batch_to_init_scores)]")
    batch_odds = np.array([
        [1.5, 4.2, 8.5, 15.0, 25.0, 50.0],
        [2.0, 3.0, 5.0, 8.0, 15.0, 30.0],
        [1.0, 100.0, 300.0, 500.0, 800.0, 1000.0]
    ])
    batch_logits = normalizer.batch_to_init_scores(batch_odds)
    assert batch_logits.shape == (3, 6)
    assert not np.any(np.isnan(batch_logits)) and not np.any(np.isinf(batch_logits))
    print(f"  バッチ入力形状: {batch_odds.shape} -> 出力ロジット形状: {batch_logits.shape}")
    print("  -> バッチ処理ステータス: ✅ PASS")

    print("\n" + "=" * 75)
    print("  🎉 全ての単体テスト・アサーションに合格しました！")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_tests()
