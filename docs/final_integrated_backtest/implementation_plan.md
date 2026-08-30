# 実装計画: 最終統合バックテスト (5,000レース規模)

## 概要と目的
最新の再学習済み本命モデル [`model_honmei.txt`](file:///d:/BOAT2512_AntiGravity_2_ana/model_honmei.txt)（Gatekeeper: 72特徴量）および残差モデル [`model_residual.txt`](file:///d:/BOAT2512_AntiGravity_2_ana/model_residual.txt)（Extractor: 71特徴量）を用いて、直近5,000レースに対する最終統合バックテストを実行します。
オッズ連動型動的EV閾値、端数プール再分配、難水面（Cluster 1）除外を適用し、前回ベースライン（ROI 101.55%, 純損益 +9,700円, MDD 112,700円, シャープレシオ 0.0041）との比較検証を実施します。

---

## 1. 実施手順

### ① シミュレーション設定
* **対象データ**: 直近 5,000 レース（`races` テーブルおよび `train_data_full.csv` の最新レース）
* **適用オプション**:
  - 難水面除外 (`--exclude_cluster1`): 戸田02, 江戸川03, 平和島04, 鳴門14, 福岡22 を Gatekeeper 前に除外
  - 動的EV閾値 (`--use_dynamic_ev`): オッズ連動型カットオフ（低オッズ 1.10〜 / 高オッズ 〜1.40）
  - 端数プール再分配 (`--use_fractional_pool`): 100円未満の端数をプールしEV最上位へ再配分
  - Gatekeeper: 相対評価 85th percentile（上位15%のレースのみ通過）
  - 資金配分: Fractional Kelly / 厳格化ポートフォリオ最適化

### ② 評価とパフォーマンス算出
* 以下の主要メトリクスを算出：
  1. 投資実行レース数（参戦数・参戦率）
  2. 的中レース数・的中率
  3. 総投資額、総払戻額、純損益
  4. 最終回収率 (ROI: Return on Investment)
  5. 最大ドローダウン (MDD: Max Drawdown)
  6. シャープレシオ (Sharpe Ratio)

### ③ 前回収支との比較検証
* 前回のベースライン（ROI 101.55%、純損益 +9,700円、MDD 112,700円、シャープレシオ 0.0041）と対比し、新特徴量（環境クロス・代替モメンタム）導入による投資効率の改善度を分析。

---

## 2. 変更・実行対象ファイル

| ファイルパス | 区分 | 内容 |
|---|---|---|
| [`simulate_betting.py`](file:///d:/BOAT2512_AntiGravity_2_ana/simulate_betting.py) | [MODIFY] | データ読み込み最適化・データセット指定の更新 |
| `docs/final_integrated_backtest/` | [NEW] | バックテスト結果・詳細レポートの保存フォルダ |

---

## 3. 検証計画
* コマンド実行:
  `python simulate_betting.py --races 5000 --exclude_cluster1 --use_dynamic_ev --use_fractional_pool`
* 結果を分析し、`docs/final_integrated_backtest/walkthrough.md` にまとめる。
