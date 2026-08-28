# 実装計画: 定量的投資システム（Quant Dual Engine）完全統合

本計画書は、オッズ非依存の本命モデル（Gatekeeper）とオッズ残差モデル（Extractor）、会場クラスタ別Benter展開、および数理ポートフォリオ最適化（Markowitz / Fractional Kelly）を結合した投資パイプラインの設計・実装方針をまとめたものです。

---

## 1. システムアーキテクチャ設計

### パイプライン構成
1. **Gatekeeper スクリーニング**:
   - `model_honmei.txt` の予測スコアを Platt Scaling で確率化。
   - 上位15%分位点（85th percentile: $P_1 \ge 0.7385$）を満たす本命レースを抽出。
2. **Extractor オッズ残差推論**:
   - `model_residual.txt` による残差ロジット $\Delta z$ と、直前オッズから算出したベースマージン `init_score` を加算して $P_{residual}$ を算出。
3. **会場クラスタ別 Benter 展開**:
   - 会場コードから水面クラスタ（Cluster 0: イン超強, Cluster 1: 難水面, Cluster 2: 標準水面）を特定し、最適 $(d_2, d_3)$ で全120通りの確率を展開。
4. **ポートフォリオ最適化**:
   - 制約（$\text{EV} \ge 1.25, \text{Odds} \le 30.0$）を通過した買い目に対し、クォーター・ケリー動的上限（$\le 10\%$）と Markowitz SLSQP 最適化で投資配分を算出。

---

## 2. 実装対象ファイル

1. [portfolio_optimizer.py](file:///d:/BOAT2512_AntiGravity_2_ana/portfolio_optimizer.py): Fractional Kelly 動的ウェイト計算と SLSQP 最適化。
2. [probability_calibration.py](file:///d:/BOAT2512_AntiGravity_2_ana/probability_calibration.py): 会場クラスタ設定ローダーと Benter 展開。
3. [odds_normalizer.py](file:///d:/BOAT2512_AntiGravity_2_ana/odds_normalizer.py): 合成勝率およびロジット変換。
4. [simulate_betting.py](file:///d:/BOAT2512_AntiGravity_2_ana/simulate_betting.py): 2段階パイプライン 高速バックテスト実行。
5. [app_boatrace.py](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [boatrace-v3-predictor/app_v3.py](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py): Streamlit アプリ UI の全面刷新。

---

## 3. 検証計画

- **バックテスト**: 過去5,000レースを対象に、参戦数、的中率、ROI、MDD、処理速度を測定。
- **UI検証**: Streamlit Cloud およびローカル環境でリアルタイムオッズ取得・推論・最適化が正常に完走することを確認。
