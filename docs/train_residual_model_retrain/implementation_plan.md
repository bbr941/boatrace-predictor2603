# 実装計画: train_residual.py の高度化と残差モデル（model_residual.txt）の再学習

## 概要と目的
拡張された75カラムのフルデータセット `train_data_full.csv`（2024年〜2026年 88.7万行）を用いて、オッズ残差学習モデル（Extractor: `model_residual.txt`）の学習スクリプト `train_residual.py` を高度化します。
オッズや配当金関連カラムの厳密なリーク防止、時系列分割（Out-of-Time: 2026年〜 37,298レース）、深さ制限・正則化による過学習防止を適用し、未見データに対する LogLoss・Brier Score・ROC-AUC および Feature Importance を検証の上、本番モデル `model_residual.txt` としてデプロイします。

---

## 1. 実施手順

### ① データソース & 特徴量リーク防止の厳密化
* 読み込みデータセットを `train_data_full.csv` に設定。
* 目的関数がオッズ残差（ベースマージン `init_score` に対する残差補正）であるため、特徴量空間 $X$ にオッズ・配当・確定結果が混入しないよう除外リストを徹底：
  - **オッズ・配当関連（リーク禁止）**: `syn_win_rate`, `odds`, `odds_1min`, `prediction_odds`, `popularity`, `vote_count`, `win_share`, `init_score`, `market_p_norm`, `has_valid_odds`, `target_binary`, `payout`, `payoff`, `profit`, `actual_result`, `is_resolved`
  - **識別子・ターゲット**: `race_id`, `boat_number`, `racer_id`, `rank`, `relevance`, `race_date`, `venue_name`, `prior_results` 等
* 75カラムから新規環境クロス・代替モメンタム特徴量を抽出して特徴量空間を構成。

### ② 正則化とハイパーパラメータの最適化
* 特徴量増加に伴う過学習を防止するため、正則化パラメータを強化：
  - `objective`: `'binary'`
  - `metric`: `'binary_logloss'`
  - `learning_rate`: `0.03`
  - `max_depth`: `6` (過度に深い分岐を抑制)
  - `num_leaves`: `31`
  - `min_data_in_leaf`: `50` (ノイズの拾いすぎを防止)
  - `feature_fraction`: `0.8` (特徴量サブサンプリング)
  - `bagging_fraction`: `0.8`, `bagging_freq`: `1`
  - `lambda_l1`: `1.0`, `lambda_l2`: `2.0` (オッズ残差の微小ノイズに対するL1/L2正則化)

### ③ 時系列分割（Out-of-Time: 2026-01-01）と評価指標の拡充
* **データ分割**:
  - Train: 2024年1月1日 〜 2025年12月31日 (663,240 行 / 110,540 レース)
  - Test: 2026年1月1日 〜 2026年8月29日 (223,788 行 / 37,298 レース)
* **評価指標**:
  - **Binary LogLoss** (オッズ単体 Base vs 旧モデル vs 新モデル)
  - **Brier Score** ($\frac{1}{N} \sum (p_i - y_i)^2$)
  - **ROC-AUC**
  - **Top-1 的中率** (レース内最高予測確率艇の1着的中精度)
  - **Feature Importance** (Gain 寄与度 Top 20、環境クロス/モメンタムの貢献度算出)

### ④ バックアップ・本番デプロイと推論互換性確認
* 既存の `model_residual.txt` を `model_residual_backup.txt` に退避。
* 新モデルを `model_residual.txt` として保存。
* `auto_trader.py` および `simulate_betting.py` から `model_residual.txt` を読み込み、推論動作の完全な互換性を確認。

---

## 2. 変更対象ファイル

| ファイルパス | 区分 | 内容 |
|---|---|---|
| [`train_residual.py`](file:///d:/BOAT2512_AntiGravity_2_ana/train_residual.py) | [MODIFY] | データ読み込み、リーク防止、時系列分割、正則化、Brier Score/LogLoss評価・Feature Importance出力の実装 |
| [`model_residual.txt`](file:///d:/BOAT2512_AntiGravity_2_ana/model_residual.txt) | [MODIFY] | 新規学習済み LightGBM 残差モデルファイル |
| `docs/train_residual_model_retrain/` | [NEW] | 設計書・タスク・検証レポートの保存フォルダ |

---

## 3. 検証計画

1. **学習・評価スクリプトの実行**:
   - `python train_residual.py --data_path train_data_full.csv --split_date 2026-01-01` を実行。
   - 2026年 Out-of-Time テストデータに対する LogLoss, Brier Score, ROC-AUC, Top-1 的中率を出力。
2. **Feature Importance 確認**:
   - オッズ残差抽出において環境クロス特徴量・代替モメンタムがどのように寄与しているか確認。
3. **推論結合テスト**:
   - `auto_trader.py` および `simulate_betting.py` で新 `model_residual.txt` がエラーなくロードされ、正常に確率補正が動作することをテスト。
