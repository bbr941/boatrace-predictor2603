# タスクリスト: Supabase (PostgreSQL) への完全移行

- [x] **1. パッケージ追加 & 接続設定の更新** <!-- id: 0 -->
  - [x] psycopg2-binary のインストール <!-- id: 1 -->
  - [x] 
equirements.txt への追加 <!-- id: 2 -->
  - [x] .env の接続 URL 最適化 (Pooler 経由) <!-- id: 3 -->
- [x] **2. 共通データベースマネージャー (db_manager.py) の実装** <!-- id: 4 -->
  - [x] PostgreSQL / SQLite ハイブリッド接続管理 (get_connection) <!-- id: 5 -->
  - [x] 特殊文字入りパスワード対応パーサー <!-- id: 6 -->
  - [x] テーブル自動初期化・マイグレーション (init_database) <!-- id: 7 -->
  - [x] レース推論・オッズ・買い目・通知ログの保存 & 取得関数 <!-- id: 8 -->
- [x] **3. 各スクリプトの PostgreSQL 対応** <!-- id: 9 -->
  - [x] uto_trader.py への Supabase 自動保存ロジック統合 <!-- id: 10 -->
  - [x] simulate_betting.py 等のプレースホルダー・クエリ改修 <!-- id: 11 -->
- [x] **4. Supabase テーブル構築 & 接続・CRUD 検証** <!-- id: 12 -->
  - [x] Supabase へのテーブルマイグレーション実行 <!-- id: 13 -->
  - [x] データ挿入・読み込みの単体テスト <!-- id: 14 -->
  - [x] uto_trader.py --mock --dry-run での E2E 永続化テスト <!-- id: 15 -->
- [x] **5. ドキュメント作成 & Git コミット** <!-- id: 16 -->
  - [x] walkthrough.md の作成 <!-- id: 17 -->
  - [x] Git コミット & プッシュ <!-- id: 18 -->
