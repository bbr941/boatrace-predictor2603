# 実装計画: Supabase (PostgreSQL) への完全移行

## 目的と概要
本計画は、ボートレース投資システム全体のデータベース接続先を、ローカルの SQLite (`boatrace.db`) からクラウド型 PostgreSQL データベース（**Supabase**）へ完全移行するものです。

これにより、クラウド常駐ワーカー（`auto_trader.py`）、Streamlit アプリケーション、シミュレーション・分析スクリプト間で推論結果・オッズ・投資履歴・通知ログを一元管理・共有できる強固なデータ基盤を構築します。

---

## 1. 移行対象コンポーネントと方針

### A. 依存関係の更新 (`requirements.txt`)
- PostgreSQL 接続用ドライバ `psycopg2-binary` を追加。

### B. 共通データベース接続・マイグレーションマネージャー (`db_manager.py`)
- `.env` から `DATABASE_URL` を読み込み。
  - 特殊文字（パスワード内の `@` や `%`）を含む接続文字列に対応した堅牢なパーサーおよびコネクション生成ロジックを実装。
  - Supabase Connection Pooler（東京リージョン `aws-0-ap-northeast-1.pooler.supabase.com:5432` / `6543`）に対応。
  - `DATABASE_URL` 未設定時はローカル SQLite への自動フォールバックを維持（下位互換性）。
- **SQL 構文の PostgreSQL 対応**:
  - プレースホルダー: `?` $\to$ `%s`
  - 主キー: `INTEGER PRIMARY KEY AUTOINCREMENT` $\to$ `SERIAL PRIMARY KEY`
  - 重複回避: `INSERT OR IGNORE INTO` $\to$ `INSERT INTO ... ON CONFLICT (...) DO NOTHING`

### C. Supabase 側 テーブルスキーマ定義 (`init_database`)
1. **`race_predictions`**:
   - `id SERIAL PRIMARY KEY`, `race_id VARCHAR(50) UNIQUE`, `race_date VARCHAR(10)`, `venue_code INT`, `venue_name VARCHAR(20)`, `race_no INT`, `deadline_time VARCHAR(20)`, `top_boat INT`, `max_p1 FLOAT`, `prob_gap FLOAT`, `gatekeeper_passed BOOLEAN`, `cluster_id INT`, `status VARCHAR(50)`, `created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP`
2. **`recommended_bets`**:
   - `id SERIAL PRIMARY KEY`, `race_id VARCHAR(50)`, `combination VARCHAR(10)`, `bet_amount INT`, `prob FLOAT`, `odds FLOAT`, `ev FLOAT`, `expected_return INT`, `created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP`, `UNIQUE(race_id, combination)`
3. **`odds_data`**:
   - `id SERIAL PRIMARY KEY`, `race_id VARCHAR(50)`, `combination VARCHAR(10)`, `odds_1min FLOAT`, `created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP`, `UNIQUE(race_id, combination)`
4. **`notification_logs`**:
   - `id SERIAL PRIMARY KEY`, `race_id VARCHAR(50)`, `channel VARCHAR(20)`, `title VARCHAR(200)`, `message_payload JSONB`, `status VARCHAR(20)`, `created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP`

### D. 各種スクリプトの改修
- **`auto_trader.py`**: 推論実行時、Gatekeeper 結果・Extractor Benter 確率・選出買い目・Discord 通知結果を Supabase へ自動永続化。
- **`simulate_betting.py` / `train_v3.py`**: PostgreSQL / SQLite 両対応のデータ取得クエリに対応。
- **`.env`**: Supabase Pooler 接続 URL を最適化・更新。

---

## 2. 変更予定ファイル一覧

| ファイルパス | 変更区分 | 内容 |
|---|---|---|
| [`requirements.txt`](file:///d:/BOAT2512_AntiGravity_2_ana/requirements.txt) | [MODIFY] | `psycopg2-binary` を追加 |
| [`.env`](file:///d:/BOAT2512_AntiGravity_2_ana/.env) | [MODIFY] | Supabase 接続 URL を最適化 |
| [`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py) | [NEW] | 共通 DB 接続・マイグレーション・CRUD ヘルパー |
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | Supabase への推論結果・オッズ・買い目・通知ログ保存を統合 |
| [`simulate_betting.py`](file:///d:/BOAT2512_AntiGravity_2_ana/simulate_betting.py) | [MODIFY] | `db_manager.py` 経由での PostgreSQL / SQLite クエリ対応 |

---

## 3. 検証計画

1. **Supabase 接続 & マイグレーション検証**:
   - `python -c "import db_manager; db_manager.init_database()"` を実行し、Supabase 上に全テーブルが正常に作成されるか確認。
2. **データの書き込み・読み込みドライラン**:
   - `db_manager.py` の単体テストを実行し、テスト用レースデータ・オッズ・買い目データの `INSERT ... ON CONFLICT DO NOTHING` および `SELECT` がエラーなく動作することを検証。
3. **`auto_trader.py` 結合テスト**:
   - `python auto_trader.py --mock --dry-run` を実行し、推論結果が Supabase に自動保存されることを確認。
