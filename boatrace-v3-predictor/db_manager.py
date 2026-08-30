"""
db_manager.py
🚤 BOATRACE AI - Supabase (PostgreSQL) データベースマネージャー
- .env の DATABASE_URL (Supabase Connection Pooler) を自動読み込み & 接続
- PostgreSQL / SQLite 自動判定 & フォールバック
- テーブル自動マイグレーション (init_database)
- レース推論結果、推奨買い目、オッズ、通知ログの CRUD ヘルパー
- プレースホルダー (%s / ?) および ON CONFLICT (DO NOTHING / UPDATE) の自動対応
"""

import os
import re
import json
import logging
import datetime
from datetime import timedelta
import urllib.parse
from typing import Dict, List, Tuple, Optional, Any, Union


# 環境変数の安全なロード (.env があれば読み込む)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("DBManager")

def get_database_url() -> str:
    """
    環境変数または Streamlit Secrets から DATABASE_URL を取得
    """
    url = os.getenv('DATABASE_URL', '')
    if not url:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
                url = st.secrets['DATABASE_URL']
        except Exception:
            pass
    return url

# PostgreSQL ドライバの安全なインポート
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    HAS_PSYCOPG2 = False

import sqlite3

DATABASE_URL = get_database_url()
SQLITE_DB_PATH = 'boatrace.db'


def parse_database_url(url: str) -> Dict[str, Any]:
    """
    URLエンコードされた特殊文字（@, %等）を含む PostgreSQL URL を安全にパース
    """
    if not url:
        return {}
        
    try:
        # standard parsed url
        parsed = urllib.parse.urlparse(url)
        user = urllib.parse.unquote(parsed.username) if parsed.username else ''
        password = urllib.parse.unquote(parsed.password) if parsed.password else ''
        host = parsed.hostname or ''
        port = parsed.port or 5432
        dbname = parsed.path.lstrip('/') or 'postgres'
        return {
            'user': user,
            'password': password,
            'host': host,
            'port': port,
            'dbname': dbname
        }
    except Exception as e:
        logger.warning(f"URLパース例外: {e}, 正規表現フォールバックを試行します...")
        
    # フォールバック正規表現: postgresql://<user>:<password>@<host>:<port>/<dbname>
    m = re.match(r'^(?:postgresql|postgres)://([^:]+):(.*)@([^:/]+)(?::(\d+))?/(.+)$', url)
    if m:
        user_raw, pass_raw, host, port_raw, dbname = m.groups()
        return {
            'user': urllib.parse.unquote(user_raw),
            'password': urllib.parse.unquote(pass_raw),
            'host': host,
            'port': int(port_raw) if port_raw else 5432,
            'dbname': dbname
        }
    return {}


class DBConnection:
    """
    PostgreSQL と SQLite を透過的に扱うコネクションラッパー
    """
    def __init__(self, raw_conn, is_postgres: bool):
        self.conn = raw_conn
        self.is_postgres = is_postgres
        
    def cursor(self):
        return self.conn.cursor()
        
    def commit(self):
        self.conn.commit()
        
    def rollback(self):
        self.conn.rollback()
        
    def close(self):
        self.conn.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


def get_db_connection() -> DBConnection:
    """
    PostgreSQL (Supabase) への接続を優先確立し、未設定時は SQLite にフォールバック
    """
    db_url = get_database_url()
    
    if db_url and HAS_PSYCOPG2:
        params = parse_database_url(db_url)
        if params and params.get('host'):
            try:
                pg_conn = psycopg2.connect(
                    user=params['user'],
                    password=params['password'],
                    host=params['host'],
                    port=params['port'],
                    dbname=params['dbname'],
                    sslmode='require',
                    connect_timeout=8
                )
                return DBConnection(pg_conn, is_postgres=True)
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL接続失敗 ({params['host']}): {e} -> SQLiteへフォールバック")
                
    # SQLite フォールバック
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    return DBConnection(sqlite_conn, is_postgres=False)


# =====================================================================
# テーブル初期化 & マイグレーション
# =====================================================================

def init_database() -> bool:
    """
    Supabase (PostgreSQL) または SQLite 上に必要なテーブルとインデックスを作成
    """
    with get_db_connection() as db:
        cur = db.cursor()
        
        if db.is_postgres:
            logger.info("🐘 [Supabase / PostgreSQL] テーブルスキーマのマイグレーションを開始します...")
            
            # 1. レース推論・Gatekeeper 結果テーブル
            cur.execute("""
            CREATE TABLE IF NOT EXISTS race_predictions (
                id SERIAL PRIMARY KEY,
                race_id VARCHAR(50) UNIQUE NOT NULL,
                race_date VARCHAR(10) NOT NULL,
                venue_code INT NOT NULL,
                venue_name VARCHAR(20) NOT NULL,
                race_no INT NOT NULL,
                deadline_time VARCHAR(20),
                top_boat INT,
                max_p1 FLOAT,
                prob_gap FLOAT,
                gatekeeper_passed BOOLEAN DEFAULT FALSE,
                cluster_id INT,
                cluster_name VARCHAR(50),
                status VARCHAR(50) NOT NULL,
                source VARCHAR(20) DEFAULT 'auto',
                actual_result VARCHAR(10),
                payout INT DEFAULT 0,
                profit INT DEFAULT 0,
                is_resolved BOOLEAN DEFAULT FALSE,
                hit_status VARCHAR(20),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # カラムのマイグレーション（既存テーブルへの追加）
            for col_ddl in [
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'auto';",
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS actual_result VARCHAR(10);",
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS payout INT DEFAULT 0;",
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS profit INT DEFAULT 0;",
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS is_resolved BOOLEAN DEFAULT FALSE;",
                "ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS hit_status VARCHAR(20);"
            ]:
                try: cur.execute(col_ddl)
                except Exception: pass
            
            # 2. 最適化選出買い目テーブル
            cur.execute("""
            CREATE TABLE IF NOT EXISTS recommended_bets (
                id SERIAL PRIMARY KEY,
                race_id VARCHAR(50) NOT NULL,
                combination VARCHAR(10) NOT NULL,
                bet_amount INT NOT NULL,
                prob FLOAT NOT NULL,
                odds FLOAT NOT NULL,
                ev FLOAT NOT NULL,
                expected_return INT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_race_bet UNIQUE(race_id, combination)
            );
            """)
            
            # 3. 3連単直前オッズキャッシュテーブル
            cur.execute("""
            CREATE TABLE IF NOT EXISTS odds_data (
                id SERIAL PRIMARY KEY,
                race_id VARCHAR(50) NOT NULL,
                combination VARCHAR(10) NOT NULL,
                odds_1min FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_race_odds UNIQUE(race_id, combination)
            );
            """)
            
            # 4. Discord / 外部通知ログテーブル
            cur.execute("""
            CREATE TABLE IF NOT EXISTS notification_logs (
                id SERIAL PRIMARY KEY,
                race_id VARCHAR(50) NOT NULL,
                channel VARCHAR(20) DEFAULT 'discord',
                title VARCHAR(200),
                message_payload JSONB,
                status VARCHAR(20),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 5. 直近節間展示タイム同期テーブル (クラウド用)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS recent_exhibitions (
                id SERIAL PRIMARY KEY,
                race_date VARCHAR(10) NOT NULL,
                venue_code INT NOT NULL,
                race_no INT NOT NULL,
                boat_number INT NOT NULL,
                racer_id INT NOT NULL,
                exhibition_time FLOAT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_recent_exhibition UNIQUE (race_date, venue_code, race_no, racer_id)
            );
            """)
            
            # インデックスの作成
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_date ON race_predictions(race_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_status ON race_predictions(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_source ON race_predictions(source);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_resolved ON race_predictions(is_resolved);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rec_bets_race ON recommended_bets(race_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_odds_data_race ON odds_data(race_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_ex_venue_racer ON recent_exhibitions(venue_code, racer_id, race_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_ex_date ON recent_exhibitions(race_date);")
            
            logger.info("✅ [Supabase / PostgreSQL] 全5テーブルおよびインデックスのマイグレーションが完了しました！")
            
        else:
            logger.info("📁 [SQLite] テーブルスキーマのマイグレーションを開始します...")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS race_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE NOT NULL,
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                venue_name TEXT NOT NULL,
                race_no INTEGER NOT NULL,
                deadline_time TEXT,
                top_boat INTEGER,
                max_p1 REAL,
                prob_gap REAL,
                gatekeeper_passed INTEGER DEFAULT 0,
                cluster_id INTEGER,
                cluster_name TEXT,
                status TEXT NOT NULL,
                source TEXT DEFAULT 'auto',
                actual_result TEXT,
                payout INTEGER DEFAULT 0,
                profit INTEGER DEFAULT 0,
                is_resolved INTEGER DEFAULT 0,
                hit_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # SQLite カラムマイグレーション
            for col_ddl in [
                "ALTER TABLE race_predictions ADD COLUMN source TEXT DEFAULT 'auto';",
                "ALTER TABLE race_predictions ADD COLUMN actual_result TEXT;",
                "ALTER TABLE race_predictions ADD COLUMN payout INTEGER DEFAULT 0;",
                "ALTER TABLE race_predictions ADD COLUMN profit INTEGER DEFAULT 0;",
                "ALTER TABLE race_predictions ADD COLUMN is_resolved INTEGER DEFAULT 0;",
                "ALTER TABLE race_predictions ADD COLUMN hit_status TEXT;"
            ]:
                try: cur.execute(col_ddl)
                except Exception: pass
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS recommended_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                combination TEXT NOT NULL,
                bet_amount INTEGER NOT NULL,
                prob REAL NOT NULL,
                odds REAL NOT NULL,
                ev REAL NOT NULL,
                expected_return INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(race_id, combination)
            );
            """)
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS odds_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                combination TEXT NOT NULL,
                odds_1min REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(race_id, combination)
            );
            """)
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS notification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                channel TEXT DEFAULT 'discord',
                title TEXT,
                message_payload TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS recent_exhibitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                boat_number INTEGER NOT NULL,
                racer_id INTEGER NOT NULL,
                exhibition_time REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(race_date, venue_code, race_no, racer_id)
            );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_date ON race_predictions(race_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_source ON race_predictions(source);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_resolved ON race_predictions(is_resolved);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_ex_venue_racer ON recent_exhibitions(venue_code, racer_id, race_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_ex_date ON recent_exhibitions(race_date);")
            
            logger.info("✅ [SQLite] 全5テーブルのマイグレーションが完了しました！")


            
    return True


# =====================================================================
# CRUD ヘルパー関数
# =====================================================================

def save_race_prediction(
    race_id: str,
    race_date: str,
    venue_code: int,
    venue_name: str,
    race_no: int,
    deadline_time: str,
    top_boat: Optional[int],
    max_p1: Optional[float],
    prob_gap: Optional[float],
    gatekeeper_passed: bool,
    cluster_id: Optional[int],
    cluster_name: Optional[str],
    status: str,
    source: str = 'auto'
) -> bool:
    """
    推論結果・Gatekeeper 判定結果を保存 (UPSERT)
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        
        if db.is_postgres:
            query = f"""
            INSERT INTO race_predictions (
                race_id, race_date, venue_code, venue_name, race_no,
                deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
                cluster_id, cluster_name, status, source
            ) VALUES ({', '.join([ph]*14)})
            ON CONFLICT (race_id) DO UPDATE SET
                top_boat = EXCLUDED.top_boat,
                max_p1 = EXCLUDED.max_p1,
                prob_gap = EXCLUDED.prob_gap,
                gatekeeper_passed = EXCLUDED.gatekeeper_passed,
                cluster_id = EXCLUDED.cluster_id,
                cluster_name = EXCLUDED.cluster_name,
                status = EXCLUDED.status,
                source = EXCLUDED.source,
                created_at = CURRENT_TIMESTAMP;
            """
        else:
            query = f"""
            INSERT INTO race_predictions (
                race_id, race_date, venue_code, venue_name, race_no,
                deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
                cluster_id, cluster_name, status, source
            ) VALUES ({', '.join([ph]*14)})
            ON CONFLICT(race_id) DO UPDATE SET
                top_boat = excluded.top_boat,
                max_p1 = excluded.max_p1,
                prob_gap = excluded.prob_gap,
                gatekeeper_passed = excluded.gatekeeper_passed,
                cluster_id = excluded.cluster_id,
                cluster_name = excluded.cluster_name,
                status = excluded.status,
                source = excluded.source,
                created_at = CURRENT_TIMESTAMP;
            """
            
        params = (
            race_id, race_date, venue_code, venue_name, race_no,
            deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
            cluster_id, cluster_name, status, source
        )
        cur.execute(query, params)
        logger.debug(f"Saved prediction for {race_id} (status: {status}, source: {source})")
        return True



def save_recommended_bets(
    race_id: str,
    bets: Dict[str, int],
    benter_probs: Dict[str, float],
    all_odds: Dict[str, float]
) -> int:
    """
    選出された最適化買い目を保存 (UPSERT)
    """
    if not bets:
        return 0
        
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        count = 0
        
        for combo, amt in bets.items():
            p = float(benter_probs.get(combo, 0.0))
            o = float(all_odds.get(combo, 0.0))
            ev = float(p * o)
            exp_ret = int(amt * o)
            
            if db.is_postgres:
                query = f"""
                INSERT INTO recommended_bets (
                    race_id, combination, bet_amount, prob, odds, ev, expected_return
                ) VALUES ({', '.join([ph]*7)})
                ON CONFLICT (race_id, combination) DO UPDATE SET
                    bet_amount = EXCLUDED.bet_amount,
                    prob = EXCLUDED.prob,
                    odds = EXCLUDED.odds,
                    ev = EXCLUDED.ev,
                    expected_return = EXCLUDED.expected_return,
                    created_at = CURRENT_TIMESTAMP;
                """
            else:
                query = f"""
                INSERT INTO recommended_bets (
                    race_id, combination, bet_amount, prob, odds, ev, expected_return
                ) VALUES ({', '.join([ph]*7)})
                ON CONFLICT(race_id, combination) DO UPDATE SET
                    bet_amount = excluded.bet_amount,
                    prob = excluded.prob,
                    odds = excluded.odds,
                    ev = excluded.ev,
                    expected_return = excluded.expected_return,
                    created_at = CURRENT_TIMESTAMP;
                """
            cur.execute(query, (race_id, combo, int(amt), p, o, ev, exp_ret))
            count += 1
            
        logger.info(f"💾 [{race_id}] {count} 件の推奨買い目をデータベースに保存しました。")
        return count


def save_odds_batch(race_id: str, odds_dict: Dict[str, float]) -> int:
    """
    全120通りの直前オッズデータを保存 (UPSERT)
    """
    if not odds_dict:
        return 0
        
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        count = 0
        
        for combo, o_val in odds_dict.items():
            if db.is_postgres:
                query = f"""
                INSERT INTO odds_data (race_id, combination, odds_1min)
                VALUES ({ph}, {ph}, {ph})
                ON CONFLICT (race_id, combination) DO UPDATE SET
                    odds_1min = EXCLUDED.odds_1min,
                    created_at = CURRENT_TIMESTAMP;
                """
            else:
                query = f"""
                INSERT INTO odds_data (race_id, combination, odds_1min)
                VALUES ({ph}, {ph}, {ph})
                ON CONFLICT(race_id, combination) DO UPDATE SET
                    odds_1min = excluded.odds_1min,
                    created_at = CURRENT_TIMESTAMP;
                """
            cur.execute(query, (race_id, str(combo), float(o_val)))
            count += 1
            
        return count


def save_notification_log(
    race_id: str,
    channel: str,
    title: str,
    payload: Dict[str, Any],
    status: str
) -> bool:
    """
    通知送信ログを保存
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        
        if db.is_postgres:
            payload_data = Json(payload) if HAS_PSYCOPG2 else json.dumps(payload, ensure_ascii=False)
        else:
            payload_data = json.dumps(payload, ensure_ascii=False)
            
        query = f"""
        INSERT INTO notification_logs (race_id, channel, title, message_payload, status)
        VALUES ({', '.join([ph]*5)});
        """
        cur.execute(query, (race_id, channel, title, payload_data, status))
        return True


def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    """
    直近のレース推論結果を取得
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        query = f"""
        SELECT race_id, race_date, venue_name, race_no, deadline_time,
               top_boat, max_p1, gatekeeper_passed, cluster_name, status, created_at
        FROM race_predictions
        ORDER BY created_at DESC
        LIMIT {ph};
        """
        cur.execute(query, (limit,))
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


def get_recommended_bets(race_id: str) -> List[Dict[str, Any]]:
    """
    指定レースの推奨買い目を取得
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        query = f"""
        SELECT combination, bet_amount, prob, odds, ev, expected_return
        FROM recommended_bets
        WHERE race_id = {ph}
        ORDER BY bet_amount DESC;
        """
        cur.execute(query, (race_id,))
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


def update_race_result(
    race_id: str,
    actual_result: str,
    payout: int,
    profit: int,
    hit_status: str
) -> bool:
    """
    レース確定結果と払戻金・確定損益を更新 (is_resolved = TRUE)
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        resolved_val = "TRUE" if db.is_postgres else "1"
        
        query = f"""
        UPDATE race_predictions
        SET actual_result = {ph},
            payout = {ph},
            profit = {ph},
            hit_status = {ph},
            is_resolved = {resolved_val}
        WHERE race_id = {ph};
        """
        cur.execute(query, (actual_result, int(payout), int(profit), hit_status, race_id))
        logger.info(f"🏁 [{race_id}] 結果確定更新: {actual_result} | Status: {hit_status} | Payout: {payout:,}円 | Profit: {profit:+,}円")
        return True


def get_unresolved_predictions(
    date_str: Optional[str] = None,
    source: Optional[str] = 'auto'
) -> List[Dict[str, Any]]:
    """
    結果未確定の推論レコードを取得
    """
    with get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"
        
        conditions = []
        params = []
        
        if db.is_postgres:
            conditions.append("(is_resolved IS NULL OR is_resolved = FALSE)")
        else:
            conditions.append("(is_resolved IS NULL OR is_resolved = 0)")
            
        if date_str:
            conditions.append(f"race_date = {ph}")
            params.append(date_str)
        if source:
            conditions.append(f"source = {ph}")
            params.append(source)
            
        where_sql = f"WHERE {' AND '.join(conditions)}"
        query = f"""
        SELECT race_id, race_date, venue_code, venue_name, race_no, deadline_time, status, source
        FROM race_predictions
        {where_sql}
        ORDER BY race_date ASC, deadline_time ASC;
        """
        cur.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_dashboard_stats(date_str: Optional[str] = None, source: Optional[str] = 'auto') -> Dict[str, Any]:
    """
    ダッシュボード用の KPI 統計集計 (確定損益・実回収率・的中率を含む)
    """
    default_stats = {
        'total_evaluated': 0,
        'gatekeeper_passed': 0,
        'gatekeeper_rate': 0.0,
        'investment_go': 0,
        'total_recommended_bet': 0,
        'resolved_races': 0,
        'hit_count': 0,
        'miss_count': 0,
        'hit_rate': 0.0,
        'total_payout': 0,
        'net_profit': 0,
        'resolved_bet': 0,
        'recovery_rate': 0.0
    }
    try:
        with get_db_connection() as db:
            cur = db.cursor()
            ph = "%s" if db.is_postgres else "?"
            
            conditions = []
            params = []
            if date_str:
                conditions.append(f"race_date = {ph}")
                params.append(date_str)
            if source:
                conditions.append(f"source = {ph}")
                params.append(source)
                
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                
            # 1. 評価総数
            cur.execute(f"SELECT COUNT(*) FROM race_predictions {where_clause};", params)
            total_eval = cur.fetchone()[0] or 0
            
            # 2. Gatekeeper 通過数
            gk_condition = "gatekeeper_passed = TRUE" if db.is_postgres else "gatekeeper_passed = 1"
            cur.execute(f"SELECT COUNT(*) FROM race_predictions {where_clause + (' AND ' if where_clause else 'WHERE ')} {gk_condition};", params)
            gk_passed = cur.fetchone()[0] or 0
            
            # 3. 投資GOサイン数 (ガチ投資・エンタメ・的中特化を含む)
            go_status_cond = "status IN ('investment_go', 'mock_investment_go', 'entertainment_go', 'hit_focused_go')"
            cur.execute(f"SELECT COUNT(*) FROM race_predictions {where_clause + (' AND ' if where_clause else 'WHERE ')} {go_status_cond};", params)
            go_count = cur.fetchone()[0] or 0
            
            # 4. 推奨投資総額
            join_conds = []
            if date_str:
                join_conds.append(f"p.race_date = {ph}")
            if source:
                join_conds.append(f"p.source = {ph}")
            join_where = f"WHERE {' AND '.join(join_conds)}" if join_conds else ""
            join_clause = f"JOIN race_predictions p ON b.race_id = p.race_id {join_where}" if join_where else ""
            
            cur.execute(f"SELECT COALESCE(SUM(b.bet_amount), 0) FROM recommended_bets b {join_clause};", params)
            total_bet = cur.fetchone()[0] or 0
            
            # 5. 確定損益・的中数集計 (投資GOサイン対象レースのみ)
            resolved_cond = "is_resolved = TRUE" if db.is_postgres else "is_resolved = 1"
            pnl_where = f"{where_clause + (' AND ' if where_clause else 'WHERE ')} {resolved_cond} AND {go_status_cond}"
            
            cur.execute(f"""
                SELECT 
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN hit_status = 'hit' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN hit_status = 'miss' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(payout), 0),
                    COALESCE(SUM(profit), 0)
                FROM race_predictions
                {pnl_where};
            """, params)
            pnl_row = cur.fetchone()
            resolved_races = pnl_row[0] or 0
            hit_count = pnl_row[1] or 0
            miss_count = pnl_row[2] or 0
            total_payout = int(pnl_row[3] or 0)
            net_profit = int(pnl_row[4] or 0)
            
            # 確定レースの投資総額
            cur.execute(f"""
                SELECT COALESCE(SUM(b.bet_amount), 0)
                FROM recommended_bets b
                JOIN race_predictions p ON b.race_id = p.race_id
                {join_where + (' AND ' if join_where else 'WHERE ')} {resolved_cond.replace('is_resolved', 'p.is_resolved')} AND {go_status_cond.replace('status', 'p.status')};
            """, params)
            resolved_bet = cur.fetchone()[0] or 0
            
            hit_rate = (hit_count / (hit_count + miss_count)) if (hit_count + miss_count) > 0 else 0.0
            recovery_rate = (total_payout / resolved_bet * 100.0) if resolved_bet > 0 else 0.0
            
            return {
                'total_evaluated': total_eval,
                'gatekeeper_passed': gk_passed,
                'gatekeeper_rate': (gk_passed / total_eval) if total_eval > 0 else 0.0,
                'investment_go': go_count,
                'total_recommended_bet': int(total_bet),
                'resolved_races': resolved_races,
                'hit_count': hit_count,
                'miss_count': miss_count,
                'hit_rate': hit_rate,
                'total_payout': total_payout,
                'net_profit': net_profit,
                'resolved_bet': int(resolved_bet),
                'recovery_rate': recovery_rate
            }
    except Exception as e:
        logger.warning(f"get_dashboard_stats例外: {e} -> テーブル自動初期化を試行します...")
        try:
            init_database()
        except Exception:
            pass
        return default_stats


def get_all_predictions_with_bets(
    date_str: Optional[str] = None,
    status_filter: Optional[str] = None,
    venue_filter: Optional[str] = None,
    source: Optional[str] = 'auto',
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    推論結果と対応する推奨買い目を結合して取得 (確定結果フィールドを含む)
    """
    try:
        with get_db_connection() as db:
            cur = db.cursor()
            ph = "%s" if db.is_postgres else "?"
            
            conditions = []
            params = []
            
            if date_str:
                conditions.append(f"p.race_date = {ph}")
                params.append(date_str)
            if source:
                conditions.append(f"p.source = {ph}")
                params.append(source)
            if status_filter and status_filter != 'all':
                if status_filter == 'investment_go':
                    conditions.append("p.status IN ('investment_go', 'mock_investment_go', 'entertainment_go', 'hit_focused_go')")
                elif status_filter == 'gatekeeper_passed':
                    conditions.append("p.gatekeeper_passed = TRUE" if db.is_postgres else "p.gatekeeper_passed = 1")
                elif status_filter == 'hit':
                    conditions.append("p.hit_status = 'hit'")
                elif status_filter == 'miss':
                    conditions.append("p.hit_status = 'miss'")
                else:
                    conditions.append(f"p.status = {ph}")
                    params.append(status_filter)
            if venue_filter and venue_filter != 'all':
                conditions.append(f"p.venue_name = {ph}")
                params.append(venue_filter)
                
            where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            
            query = f"""
            SELECT p.race_id, p.race_date, p.venue_code, p.venue_name, p.race_no,
                   p.deadline_time, p.top_boat, p.max_p1, p.prob_gap, p.gatekeeper_passed,
                   p.cluster_id, p.cluster_name, p.status, p.source,
                   p.actual_result, p.payout, p.profit, p.is_resolved, p.hit_status,
                   p.created_at
            FROM race_predictions p
            {where_sql}
            ORDER BY p.created_at DESC
            LIMIT {ph};
            """
            cur.execute(query, params)
            cols = [desc[0] for desc in cur.description]
            races = [dict(zip(cols, row)) for row in cur.fetchall()]

            
            # 買い目を紐付け
            if races:
                race_ids = [r['race_id'] for r in races]
                in_ph = ','.join([ph] * len(race_ids))
                cur.execute(f"""
                    SELECT race_id, combination, bet_amount, prob, odds, ev, expected_return
                    FROM recommended_bets
                    WHERE race_id IN ({in_ph})
                    ORDER BY bet_amount DESC;
                """, race_ids)
                bet_cols = [desc[0] for desc in cur.description]
                all_bets = [dict(zip(bet_cols, row)) for row in cur.fetchall()]
                
                bets_by_race = {}
                for b in all_bets:
                    rid = b['race_id']
                    if rid not in bets_by_race:
                        bets_by_race[rid] = []
                    bets_by_race[rid].append(b)
                    
                for r in races:
                    r['bets'] = bets_by_race.get(r['race_id'], [])
                    r['total_bet'] = sum(b['bet_amount'] for b in r['bets'])
                    r['max_return'] = max([b['expected_return'] for b in r['bets']]) if r['bets'] else 0
            return races
    except Exception as e:
        logger.warning(f"get_all_predictions_with_bets例外: {e} -> テーブル自動初期化を試行します...")
        try:
            init_database()
        except Exception:
            pass
        return []



def get_notification_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    通知ログを取得 (セーフフォールバック付き)
    """
    try:
        with get_db_connection() as db:
            cur = db.cursor()
            ph = "%s" if db.is_postgres else "?"
            query = f"""
            SELECT id, race_id, channel, title, message_payload, status, created_at
            FROM notification_logs
            ORDER BY created_at DESC
            LIMIT {ph};
            """
            cur.execute(query, (limit,))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            result = []
            for row in rows:
                d = dict(zip(cols, row))
                if isinstance(d.get('message_payload'), str):
                    try: d['message_payload'] = json.loads(d['message_payload'])
                    except Exception: pass
                result.append(d)
            return result
    except Exception as e:
        logger.warning(f"get_notification_logs例外: {e}")
        try:
            init_database()
        except Exception:
            pass
        return []


# =====================================================================
# クラウド連携モメンタム (recent_exhibitions) 同期 & 取得
# =====================================================================

def sync_recent_exhibitions(days: int = 10, sqlite_db_path: str = SQLITE_DB_PATH, batch_size: int = 500) -> int:
    """
    自宅PCの boatrace.db から直近N日分の展示タイムを抽出し、
    Supabase (PostgreSQL) の recent_exhibitions テーブルへ一括同期 (Upsert) する。
    """
    target_path = sqlite_db_path
    if not os.path.exists(target_path):
        alt_path = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
        if os.path.exists(alt_path):
            target_path = alt_path
        else:
            logger.warning(f"⚠️ SQLite DB ({sqlite_db_path}) が見つからないため、展示タイム同期をスキップします。")
            return 0

    logger.info(f"🔄 [Sync Exhibitions] SQLite ({target_path}) から直近 {days} 日分の展示タイムを抽出中...")
    
    try:
        s_conn = sqlite3.connect(target_path)
        s_cur = s_conn.cursor()
        query = f"""
        SELECT 
            r.race_date, 
            CAST(r.venue_code AS INTEGER) as venue_code, 
            CAST(r.race_number AS INTEGER) as race_no, 
            CAST(re.boat_number AS INTEGER) as boat_number,
            CAST(re.racer_id AS INTEGER) as racer_id, 
            CAST(bi.exhibition_time AS REAL) as exhibition_time
        FROM before_info bi
        JOIN races r ON bi.race_id = r.race_id
        JOIN race_entries re ON bi.race_id = re.race_id AND bi.boat_number = re.boat_number
        WHERE r.race_date >= date('now', '-{days} days')
          AND bi.exhibition_time > 0
        ORDER BY r.race_date ASC, r.race_number ASC
        """
        s_cur.execute(query)
        rows = s_cur.fetchall()
        s_conn.close()
        
        if not rows:
            logger.info("ℹ️ 同期対象の展示タイムレコードがありませんでした (0件)。")
            return 0
            
        logger.info(f"📦 抽出完了: {len(rows)} 件の展示タイムを Supabase (recent_exhibitions) へ同期します...")
        
        # Supabase (PostgreSQL) または SQLite に Upsert
        with get_db_connection() as db:
            cur = db.cursor()
            
            # テーブル存在確認 (未作成なら作成)
            init_database()
            
            synced_count = 0
            if db.is_postgres and HAS_PSYCOPG2:
                from psycopg2.extras import execute_values
                upsert_sql = """
                INSERT INTO recent_exhibitions (
                    race_date, venue_code, race_no, boat_number, racer_id, exhibition_time
                ) VALUES %s
                ON CONFLICT (race_date, venue_code, race_no, racer_id)
                DO UPDATE SET
                    boat_number = EXCLUDED.boat_number,
                    exhibition_time = EXCLUDED.exhibition_time,
                    created_at = CURRENT_TIMESTAMP;
                """
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    execute_values(cur, upsert_sql, batch)
                    synced_count += len(batch)
                    
                # 30日以上前の古いレコードを自動クリーンアップ
                try:
                    cur.execute("DELETE FROM recent_exhibitions WHERE race_date < (CURRENT_DATE - INTERVAL '30 days')::text;")
                except Exception:
                    pass
            else:
                ph = "?"
                insert_sql = f"""
                INSERT INTO recent_exhibitions (
                    race_date, venue_code, race_no, boat_number, racer_id, exhibition_time
                ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                ON CONFLICT(race_date, venue_code, race_no, racer_id) DO UPDATE SET
                    boat_number = excluded.boat_number,
                    exhibition_time = excluded.exhibition_time,
                    created_at = CURRENT_TIMESTAMP;
                """
                for r in rows:
                    cur.execute(insert_sql, r)
                    synced_count += 1
                    
            logger.info(f"✅ [Sync Exhibitions] Supabase への展示タイム同期が完了しました ({synced_count} 件)")
            return synced_count
            
    except Exception as e:
        logger.error(f"❌ [Sync Exhibitions] 同期エラー: {e}")
        return 0


def get_cloud_series_momentum(venue_code: int, racer_ids: List[int], race_date: Optional[str] = None) -> Dict[int, List[float]]:
    """
    Supabase の recent_exhibitions テーブルから、指定会場・指定選手の過去7日分の展示タイムを取得。
    返り値: {racer_id: [ex_time_1, ex_time_2, ...]} (時系列昇順)
    """
    res_dict = {int(rid): [] for rid in racer_ids}
    if not racer_ids:
        return res_dict
        
    # 日付フォーマットの正規化 (YYYY-MM-DD)
    if race_date:
        r_str = str(race_date)
        if len(r_str) == 8 and r_str.isdigit():
            formatted_date = f"{r_str[:4]}-{r_str[4:6]}-{r_str[6:]}"
        else:
            formatted_date = r_str
    else:
        formatted_date = datetime.date.today().strftime('%Y-%m-%d')
        
    try:
        target_dt = datetime.datetime.strptime(formatted_date, '%Y-%m-%d').date()
    except Exception:
        target_dt = datetime.date.today()
        formatted_date = target_dt.strftime('%Y-%m-%d')
        
    start_date_str = (target_dt - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        with get_db_connection() as db:
            cur = db.cursor()
            ph = "%s" if db.is_postgres else "?"
            r_placeholders = ','.join([ph] * len(racer_ids))
            
            query = f"""
            SELECT racer_id, exhibition_time
            FROM recent_exhibitions
            WHERE venue_code = {ph}
              AND racer_id IN ({r_placeholders})
              AND race_date >= {ph}
              AND race_date <= {ph}
              AND exhibition_time > 0
            ORDER BY race_date ASC, race_no ASC;
            """
            params = [int(venue_code)] + [int(r) for r in racer_ids] + [start_date_str, formatted_date]
            cur.execute(query, params)
            rows = cur.fetchall()
            
            for row in rows:
                rid = int(row[0])
                ex_time = float(row[1])
                if rid in res_dict:
                    res_dict[rid].append(ex_time)
                    
    except Exception as e:
        logger.warning(f"⚠️ [Cloud Momentum] Supabase からの展示履歴取得例外: {e}")
        
    return res_dict


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    print("=" * 65)
    print("🚤 BOATRACE AI データベースマネージャー 初期化 & 接続テスト")
    print("=" * 65)
    
    # 1. 接続確認
    with get_db_connection() as db:
        cur = db.cursor()
        if db.is_postgres:
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
            print(f"✅ Supabase (PostgreSQL) 接続成功！\n   Version: {ver[:60]}...")
        else:
            print("📁 SQLite 接続成功！")
            
    # 2. スキーママイグレーション
    init_database()
    
    # 3. テストデータの挿入 & 取得
    test_rid = "20260829_18_10_TEST"
    save_race_prediction(
        race_id=test_rid,
        race_date="20260829",
        venue_code=18,
        venue_name="徳山",
        race_no=10,
        deadline_time="15:25",
        top_boat=1,
        max_p1=0.796,
        prob_gap=0.650,
        gatekeeper_passed=True,
        cluster_id=0,
        cluster_name="イン超強水面",
        status="investment_go"
    )
    
    save_recommended_bets(
        race_id=test_rid,
        bets={"1-2-5": 2000},
        benter_probs={"1-2-5": 0.0511},
        all_odds={"1-2-5": 26.0}
    )
    
    recent = get_recent_predictions(5)
    print(f"\n📊 直近の推論データ (件数: {len(recent)}):")
    for r in recent:
        print(f"  ・[{r['race_id']}] {r['venue_name']} {r['race_no']}R | P1={r['max_p1']:.1%} | 判定: {r['status']}")
        
    bets = get_recommended_bets(test_rid)
    print(f"\n🎯 テスト買い目取得 ({test_rid}):")
    for b in bets:
        print(f"  ・{b['combination']}: {b['bet_amount']:,}円 (オッズ {b['odds']}倍, EV {b['ev']:.2f})")
        
    print("\n" + "=" * 65)
    print("✅ 全てのマイグレーションおよび CRUD テストが完了しました！")
    print("=" * 65)
