# タスクリスト: make_data_set.py への新規特徴量パイプライン移植

- [x] **1. make_data_set.py の改修** <!-- id: 0 -->
  - [x] `load_base_data` SQLクエリの拡張（`makurizashi_count`, `wintech_races_run` 等の抽出） <!-- id: 1 -->
  - [x] `process_features` への風速クロス・波高クロス・代替モメンタム特徴量ロジック統合 <!-- id: 2 -->
  - [x] 欠損値（NaN）の安全な補完・ゼロ除算ガードの実装 <!-- id: 3 -->
  - [x] CLI 引数（`--limit`, `--output`, `--test`）の追加 <!-- id: 4 -->
- [x] **2. テスト実行とデータセット整合性検証** <!-- id: 5 -->
  - [x] テストデータセット生成スクリプト実行 <!-- id: 6 -->
  - [x] 新規特徴量カラム一覧・欠損値有無・統計値の確認 <!-- id: 7 -->
- [x] **3. ドキュメント作成 & Git コミット** <!-- id: 8 -->
  - [x] `walkthrough.md` の作成 <!-- id: 9 -->
  - [x] Git コミット & プッシュ <!-- id: 10 -->
