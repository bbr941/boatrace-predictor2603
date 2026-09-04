"""
fix_missing_records.py
hit_focused_predictions から 9月3日・4日の見送りレコードを抽出し、
race_predictions テーブルへ安全にバックフィル（データ復旧）するスクリプト
"""

import os
import sys
from typing import Dict, List, Any

# プロジェクトルート設定
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db_manager


def fix_missing_records(target_dates=("20260903", "20260904")):
    print("=" * 80)
    print(" 🛠️ [DATA RECOVERY] race_predictions 見送りレコード復旧スクリプト開始")
    print(f" 📅 対象日付: {target_dates}")
    print("=" * 80)

    total_inserted = 0

    with db_manager.get_db_connection() as db:
        cur = db.cursor()
        ph = "%s" if db.is_postgres else "?"

        for d in target_dates:
            print(f"\n--- 📆 対象日: {d} ---")

            # 1. 既存の race_predictions レコードを取得
            cur.execute(f"SELECT race_id, venue_name, race_no FROM race_predictions WHERE race_date = {ph};", (d,))
            existing_rows = cur.fetchall()
            existing_race_ids = {r[0] for r in existing_rows}
            existing_venue_rno = {(r[1], r[2]) for r in existing_rows}
            print(f"  ・現在の race_predictions 件数: {len(existing_race_ids)} 件")

            # 2. hit_focused_predictions から全レコードを取得
            cur.execute(f"""
                SELECT race_id, race_date, venue_code, venue_name, race_no,
                       deadline_time, top_boat, max_p1, prob_gap, cluster_id,
                       cluster_name, status, actual_result, is_resolved
                FROM hit_focused_predictions
                WHERE race_date = {ph}
                ORDER BY venue_code ASC, race_no ASC;
            """, (d,))
            cols = [desc[0] for desc in cur.description]
            hit_races = [dict(zip(cols, row)) for row in cur.fetchall()]
            print(f"  ・hit_focused_predictions 蓄積件数: {len(hit_races)} 件")

            # 3. 欠損している見送りレースを抽出
            missing_races = []
            for hr in hit_races:
                rid = hr['race_id']
                vname = hr['venue_name']
                rno = hr['race_no']
                if rid not in existing_race_ids and (vname, rno) not in existing_venue_rno:
                    missing_races.append(hr)

            print(f"  ・未登録（見送り）レース件数: {len(missing_races)} 件")

            # 4. クレンジング & race_predictions へ INSERT
            inserted_count = 0
            for hr in missing_races:
                status = hr['status']
                max_p1 = hr.get('max_p1') or 0.0

                # Gatekeeper 通過判定の補正
                if status == 'gatekeeper_skipped':
                    gk_passed = False
                elif status in ('sniper_skipped', 'no_value_bets', 'investment_go'):
                    gk_passed = True
                else:
                    gk_passed = (max_p1 >= 0.7438)

                # 結果確定状態の反映 (見送りレースのため、投資0/払戻0/損益0)
                is_resolved = bool(hr.get('is_resolved') or hr.get('actual_result'))
                actual_result = hr.get('actual_result')
                payout = 0
                profit = 0
                hit_status = "no_bet" if is_resolved else None

                if db.is_postgres:
                    insert_sql = f"""
                    INSERT INTO race_predictions (
                        race_id, race_date, venue_code, venue_name, race_no,
                        deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
                        cluster_id, cluster_name, status, source,
                        actual_result, payout, profit, is_resolved, hit_status
                    ) VALUES ({', '.join([ph]*19)})
                    ON CONFLICT (race_id) DO NOTHING;
                    """
                else:
                    insert_sql = f"""
                    INSERT INTO race_predictions (
                        race_id, race_date, venue_code, venue_name, race_no,
                        deadline_time, top_boat, max_p1, prob_gap, gatekeeper_passed,
                        cluster_id, cluster_name, status, source,
                        actual_result, payout, profit, is_resolved, hit_status
                    ) VALUES ({', '.join([ph]*19)})
                    ON CONFLICT (race_id) DO NOTHING;
                    """

                params = (
                    hr['race_id'], hr['race_date'], hr['venue_code'], hr['venue_name'], hr['race_no'],
                    hr.get('deadline_time'), hr.get('top_boat'), hr.get('max_p1'), hr.get('prob_gap'),
                    gk_passed, hr.get('cluster_id'), hr.get('cluster_name'), status, "auto",
                    actual_result, payout, profit, is_resolved, hit_status
                )
                cur.execute(insert_sql, params)
                inserted_count += 1

            total_inserted += inserted_count
            print(f"  ✅ {d} の見送りレコード {inserted_count} 件を race_predictions へ復旧しました。")

    print("\n" + "=" * 80)
    print(f" ✨ データ復旧完了: 合計 {total_inserted} 件のレースを race_predictions へバックフィルしました。")
    print("=" * 80)

    # 5. 集計結果の確認
    print("\n📊 【ダッシュボード集計値の確認】")
    for d in target_dates:
        stats = db_manager.get_dashboard_stats(date_str=d, source='auto')
        print(f"  [{d}] 総評価レース数: {stats['total_evaluated']} 件 | GK通過数: {stats['gatekeeper_passed']} 件 ({stats['gatekeeper_rate']:.1%}) | 投資GO: {stats['investment_go']} 件 | 確定損益: {stats['net_profit']:+,} 円")


if __name__ == '__main__':
    fix_missing_records()
