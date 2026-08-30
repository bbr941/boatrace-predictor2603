# タスクリスト: train_residual.py の高度化と残差モデル再学習

- [x] **1. train_residual.py の改修** <!-- id: 0 -->
  - [x] 読み込み対象を `train_data_full.csv` に設定 <!-- id: 1 -->
  - [x] オッズ・配当・確定情報のリーク防止除外リストの徹底 <!-- id: 2 -->
  - [x] Out-of-Time（2026-01-01基準）時系列分割の実装 <!-- id: 3 -->
  - [x] 正則化ハイパーパラメータ（max_depth, min_data_in_leaf, L1/L2正則化）のチューニング <!-- id: 4 -->
  - [x] 評価指標（LogLoss, Brier Score, ROC-AUC, Top-1的中率）およびFeature Importance出力の実装 <!-- id: 5 -->
- [x] **2. 残差モデルの再学習と Out-of-Time 評価** <!-- id: 6 -->
  - [x] 旧モデルのバックアップ（`model_residual_backup.txt`） <!-- id: 7 -->
  - [x] `train_residual.py` の実行と学習 <!-- id: 8 -->
  - [x] 2026年テストデータにおける精度評価とベースライン比較 <!-- id: 9 -->
  - [x] Feature Importance（Gain 寄与度 Top 20）の分析 <!-- id: 10 -->
- [x] **3. 本番デプロイと推論パイプライン結合確認** <!-- id: 11 -->
  - [x] `model_residual.txt` の更新保存 <!-- id: 12 -->
  - [x] `auto_trader.py` / `simulate_betting.py` からの推論動作テスト <!-- id: 13 -->
- [x] **4. ドキュメント作成 & Git コミット** <!-- id: 14 -->
  - [x] `walkthrough.md` の作成 <!-- id: 15 -->
  - [x] Git コミット & プッシュ <!-- id: 16 -->
