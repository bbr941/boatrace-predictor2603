# 実装計画: 新データセット生成と本命モデル（model_honmei.txt）の再学習

## 概要と目的
`make_data_set.py` に移植された75カラムのデータパイプラインを用いて、2024年〜2026年の実データ（約14.7万レース／約88万レコード）から本番学習用フルデータセット `train_data_full.csv` を生成します。
その後、`train_model.py` を改修して新規環境クロス特徴量・代替モメンタム特徴量を組み込んだ本命モデル（LambdaRank）を再学習させ、Out-of-Time テストデータ（2026年〜）に対する精度評価（NDCG@1〜3、Top-1的中率、3連単的中率）と Feature Importance を検証の上、本番モデル `model_honmei.txt` として保存・デプロイします。

---

## 1. 実施手順

### ① フルデータセットの生成 (`make_data_set.py`)
* `python make_data_set.py --start_date 2024-01-01 --output train_data_full.csv`
* 2024年以降の全レース（約14.7万レース）から75カラムの完全な学習用データを生成。

### ② `train_model.py` の改修と特徴量リスト拡充
* 以下の新規特徴量を `get_features(df, mode='honmei')` の学習対象に明示的に登録：
  - **風速クロス**: `wind_makuri_cross`, `strong_wind_makuri`, `wind_makurizashi_cross`, `strong_wind_outer_adv`, `wind_nige_vulnerability`
  - **波高クロス**: `wave_weight_prod`, `wave_weight_ratio`, `high_wave_heavy_penalty`, `high_wave_inner_risk`
  - **代替モメンタム**: `ex_diff_from_race_min`, `ex_diff_from_race_mean`, `ex_rank_in_race`, `ex_momentum_diff`, `ex_momentum_deviation`
  - **決まり手出現率**: `makuri_rate`, `makurizashi_rate`, `sashi_rate`, `nige_rate`
* 時系列分割（Out-of-Time）:
  - Train: 2024年1月1日 〜 2025年12月31日
  - Test: 2026年1月1日 〜 現在（約8ヶ月分の未見データで厳密検証）
* 評価指標の拡張:
  - NDCG@1, NDCG@2, NDCG@3
  - レース単位 Top-1 予想的中率（1着的中精度）
  - 3連単 Top-1 予想的中率（1-2-3着完全的中精度）
  - Feature Importance（Gain 寄与度ランキング Top 20）の出力

### ③ 本番推論エンジン (`auto_trader.py`) への特徴量反映
* `auto_trader.py` の `evaluate_race` 内の特徴量生成処理に、新規環境クロス・波高クロス・展示偏差の計算ロジックを追加し、実運用時にも欠損なく特徴量が渡るよう整合性を確保。

### ④ 精度比較・検証と本番デプロイ
* 旧 `model_honmei.txt` を `model_honmei_backup.txt` に退避。
* 新モデルを評価し、精度向上を確認後 `model_honmei.txt` に上書き保存。
* `walkthrough.md` に新旧精度の比較表と Feature Importance を記録し、GitHub へプッシュ。

---

## 2. 変更対象ファイル

| ファイルパス | 区分 | 内容 |
|---|---|---|
| [`train_model.py`](file:///d:/BOAT2512_AntiGravity_2_ana/train_model.py) | [MODIFY] | 新規特徴量対応、時系列スプリット、NDCG/Top-1評価・Feature Importance出力の実装 |
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | リアルタイム推論時の新規クロス特徴量・モメンタム生成ロジックの追加 |
| [`model_honmei.txt`](file:///d:/BOAT2512_AntiGravity_2_ana/model_honmei.txt) | [MODIFY] | 新規学習済み LightGBM LambdaRank モデルファイル |
| `docs/train_honmei_model_retrain/` | [NEW] | 設計書・タスク・検証レポートの保存フォルダ |

---

## 3. 検証計画

1. **データセット生成完了の確認**:
   - `train_data_full.csv` の行数、カラム数、欠損値 0 件を確認。
2. **モデル学習・評価の実行**:
   - `python train_model.py` を実行。
   - Out-of-Time（2026年）テストデータに対する NDCG@1〜3、Top-1的中率、Feature Importance を確認。
3. **推論パイプラインの結合テスト**:
   - `auto_trader.py` および `simulate_betting.py` から新 `model_honmei.txt` を読み込み、推論が正常に動作することを確認。
