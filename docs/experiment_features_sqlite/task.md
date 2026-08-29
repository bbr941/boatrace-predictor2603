# タスクリスト: ローカルSQLiteを用いた環境クロス特徴量の検証

- [x] **1. experiment_features_sqlite.py の設計と作成** <!-- id: 0 -->
  - [x] SQLite (`boatrace.db`) からの高速 SQL 抽出関数の実装 <!-- id: 1 -->
  - [x] 新規特徴量（風速クロス、波高クロス、代替モメンタム）の算出ロジック実装 <!-- id: 2 -->
  - [x] ベースライン vs 新規特徴量追加版の LightGBM 学習・評価ループ実装 <!-- id: 3 -->
  - [x] Feature Importance 出力 & 指標比較機能の実装 <!-- id: 4 -->
- [x] **2. 実験の実行と検証** <!-- id: 5 -->
  - [x] スクリプト実行による AUC / LogLoss / Brier Score の測定 <!-- id: 6 -->
  - [x] Feature Importance の分析 <!-- id: 7 -->
- [x] **3. ドキュメント作成 & Git コミット** <!-- id: 8 -->
  - [x] `walkthrough.md` の作成 <!-- id: 9 -->
  - [x] Git コミット & プッシュ <!-- id: 10 -->
