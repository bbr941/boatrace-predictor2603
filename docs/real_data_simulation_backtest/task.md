# 実データ・シミュレーションバックテスト 実装タスク

- [/] 1. プロジェクト構成の準備
    - [x] ドキュメント用フォルダの作成 (`docs/real_data_simulation_backtest`)
    - [x] `task.md` の作成（brain/project）
    - [/] `implementation_plan.md` の作成（設計とレビュー）
- [x] 2. 実データバックテスト本体の実装 (`monte_carlo_backtest.py`)
    - [x] `boatrace.db` からの特徴量・結果・配当データ取得 (SQL)
    - [x] 特徴量の正規化パイプライン (MinMaxScaler)
    - [x] 時系列 Train/Test Split (例: 前半2ヶ月をTrain、後半1ヶ月をTest)
    - [x] OptunaによるTrain期間での重み最適化
    - [x] Test期間でのバックテスト実行とROI/的中頻度の算出
- [x] 3. 動作検証と結果の確認
    - [x] 探索された重みの妥当性確認
    - [x] 未知データに対するROIとベット機会のバランスチェック
- [x] 4. 完了報告
    - [x] `walkthrough.md` の作成
    - [x] ユーザーへの最終報告
    - [x] プロジェクト内の `docs` フォルダへのドキュメント同期
