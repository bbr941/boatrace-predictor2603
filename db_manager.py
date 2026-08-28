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
import urllib.parse
from typing import Dict, List, Tuple, Optional, Any, Union

from dotenv import load_dotenv

# 環境変数ロード
load_dotenv()

logger = logging.getLogger("DBManager")

# PostgreSQL ドライバの安全なインポート
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    HAS_PSYCOPG2 = False

import sqlite3

DATABASE_URL = os.getenv('DATABASE_URL', '')
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
    db_url = os.getenv('DATABASE_URL', '')
    
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
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
            
            # インデックスの作成
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_date ON race_predictions(race_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_race_pred_status ON race_predictions(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rec_bets_race ON recommended_bets(race_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_odds_data_race ON odds_data(race_id);")
            
            logger.info("✅ [Supabase / PostgreSQL] 全4テーブルおよびインデックスのマイグレーションが完了しました！")
            
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
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
            
            logger.info("✅ [SQLite] 全4テーブルのマイグレーションが完了しました！")
            
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
    status: str
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
                cluster_id, cluster_name, status
            ) VALUES ({', '.join([ph]*13)})
            ON CONFLICT (race_id) DO UPDATE SET
                top_boat = EXCLUDED.top_boat,
                max_p1 = EXCLUDED.max_p1,
                prob_gap = EXCLUDED.prob_gap,
                gatekeeper_passed = EXCLUDED.gatekeeper_passed,
                cluster_id = EXCLUDED.cluster_id,
                cluster_name = EXCLUDED.cluster_name,
                status = EXCLUDED.status,
                created_at = CURRENT_TIMESTAMP;
            """
        else:
            query = f"""
            INSERT INTO race_predictions (
                race_id, race_date, venue_code, venue_name, race_no,
                deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
                cluster_id, cluster_name, status
            ) VALUES ({', '.join([ph]*13)})
            ON CONFLICT(race_id) DO UPDATE SET
                top_boat = excluded.top_boat,
                max_p1 = excluded.max_p1,
                prob_gap = excluded.prob_gap,
                gatekeeper_passed = excluded.gatekeeper_passed,
                cluster_id = excluded.cluster_id,
                cluster_name = excluded.cluster_name,
                status = excluded.status,
                created_at = CURRENT_TIMESTAMP;
            """
            
        params = (
            race_id, race_date, venue_code, venue_name, race_no,
            deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
            cluster_id, cluster_name, status
        )
        cur.execute(query, params)
        logger.debug(f"Saved prediction for {race_id} (status: {status})")
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
