# 実装計画: portfolio_optimizer.py の高度化改修（動的EV閾値 & 端数プール再分配）

## 目的と概要
`portfolio_optimizer.py` に以下の 2 つのクオンツ高度化ロジックを実装します：
1. **オッズ連動型動的EV閾値（分散連動型カットオフ）**: 一律 `EV >= 1.25` を廃止し、本命サイド（オッズ < 10.0）は閾値を緩和（`EV >= 1.10`）、穴サイド（オッズ >= 20.0）は分散リスク対策として閾値を厳格化（`EV >= 1.30 ~ 1.40`）する動的関数を導入。
2. **端数プール再分配ロジック（Fractional Rounding Pool）**: 100円未満の切り捨てロス（例: 480円 $\rightarrow$ 400円で生じる80円）を合算プールし、100円に達するごとに最も EV（期待値）が高い買い目へ +100円 として繰り上げ再投下する。

改修後、`simulate_betting.py` および全システム（`auto_trader.py`, `app_boatrace.py`, `app_v3.py`）へ統合し、バックテストを実行して ROI・最終損益・ドローダウンへの影響を検証します。

---

## 1. 改修コンポーネント詳細

### ① オッズ連動型動的EV閾値 (`calculate_dynamic_min_ev`)
- **数理仕様**:
  - $O < 10.0$ (本命サイド): $\text{min\_ev} = 1.10 + (1.25 - 1.10) \times \frac{O - 1.0}{9.0}$ (1.10 〜 1.25 へ線形接続)
  - $10.0 \le O < 20.0$ (中配当サイド): $\text{min\_ev} = 1.25$ (基準閾値)
  - $O \ge 20.0$ (大穴サイド): $\text{min\_ev} = \min(1.40, 1.25 + (O - 20.0) \times 0.015)$ (最大 1.40 まで厳格化)
- **効果**: 的中率が高く分散の小さい本命の微バリュー（EV 1.10〜1.24）を拾い上げつつ、外れ続けるドローダウンリスクの高い大穴を厳選。

### ② 端数プール再分配クラス (`FractionalRoundingPool`)
- **アルゴリズム**:
  1. SLSQP 最適解の理論投資額 $W_i \times \text{Bankroll}$ に対し、基礎100円切り捨て額 $\lfloor \text{amt}_i / 100 \rfloor \times 100$ と端数余剰 $R_i = \text{amt}_i - \text{base\_amt}_i$ を算出。
  2. レース内の端数を合計: $\text{Pool} = \sum R_i$ (+ 前レース繰越残高)。
  3. $\text{Pool} \ge 100$ の間、候補買い目リストを **EV 降順**（第2キー: 確率降順）にソートし、最上位の買い目へ +100円 を繰り上げ配分。
  4. 集中投資上限制約（`max_concentration`）を遵守しつつ、端数残高が 100円 未満になるまでループ。
- **効果**: 100円切り捨てによる資金投下効率の目減りをゼロにし、最も期待値の高い買い目に余剰資金を集中投下。

### ③ バックテスト検証 (`simulate_betting.py`)
- 引数に `--use_dynamic_ev`（デフォルト: True）および `--use_fractional_pool`（デフォルト: True）を追加。
- 以下の 2 条件でバックテスト比較を実行：
  - **ベースライン**: 一律 EV >= 1.25, 難水面除外, 通常切り捨て
  - **高度化後**: 動的EV閾値 + 端数プール再分配, 難水面除外

---

## 2. 変更対象ファイル一覧

| ファイルパス | 変更区分 | 内容 |
|---|---|---|
| [`portfolio_optimizer.py`](file:///d:/BOAT2512_AntiGravity_2_ana/portfolio_optimizer.py) | [MODIFY] | `calculate_dynamic_min_ev`, `FractionalRoundingPool` クラス追加、`optimize_funds` の改修 |
| [`simulate_betting.py`](file:///d:/BOAT2512_AntiGravity_2_ana/simulate_betting.py) | [MODIFY] | 動的EV・端数プール対応オプション追加、バックテスト実行 |
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | リアルタイム自動推論への動的EV & 端数プール統合 |
| [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py) | [MODIFY] | Streamlit アプリへの動的EV & 端数プール統合 |

---

## 3. 検証計画

1. **単体テスト**:
   - `portfolio_optimizer.py` 内のダミーデータ（オッズ 5.0倍/EV 1.15、オッズ 25.0倍/EV 1.28、端数余剰 180円）で動的EV通過判定および端数プールの +100円 再配分を検証。
2. **バックテスト比較実行**:
   - `simulate_betting.py` を 5,000 レース規模で実行し、ROI、総利益、的中率、最大ドローダウンを比較出力。
