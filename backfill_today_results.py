"""
backfill_today_results.py
本日分の終了レース結果を公式から高速並列取得し、hit_focused_predictions テーブルに遡及反映（バックフィル精算）するスクリプト
"""

import os
import sys
import time
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# プロジェクトルート
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db_manager
from auto_trader import BoatRaceScraper, VENUE_MAP


def process_single_race(r: Dict[str, Any], target_date: str) -> Optional[Dict[str, Any]]:
    race_id = r['race_id']
    race_date = r['race_date']
    venue_code = r['venue_code']
    venue_name = r.get('venue_name') or VENUE_MAP.get(venue_code, f"会場{venue_code}")
    race_no = r['race_no']
    deadline = r.get('deadline_time', '--:--')
    
    if "_MOCK" in race_id or "_TEST" in race_id:
        return None
        
    clean_d = race_date.replace("-", "") if race_date else target_date
    res = BoatRaceScraper.get_race_result(clean_d, venue_code, race_no)
    
    if not res:
        return {
            'race_id': race_id,
            'venue_name': venue_name,
            'race_no': race_no,
            'deadline': deadline,
            'status': 'skipped',
            'message': '⏳ 結果未確定（レース未終了または集計中）'
        }
        
    combo = res['combo']
    payout_per_100 = res['payout_per_100']
    
    detail = db_manager.get_hit_focused_race_detail(race_id)
    bets = detail.get('bets', []) if detail else []
    
    if not bets:
        db_manager.update_hit_focused_result(
            race_id=race_id,
            actual_result=combo,
            payout=0,
            profit=0,
            hit_status="no_bet"
        )
        return {
            'race_id': race_id,
            'venue_name': venue_name,
            'race_no': race_no,
            'deadline': deadline,
            'status': 'settled',
            'actual_result': combo,
            'hit_status': 'no_bet',
            'total_bet': 0,
            'payout': 0,
            'profit': 0,
            'message': f"☕ 買い目なし (出目: {combo})"
        }
        
    total_bet = sum(b['bet_amount'] for b in bets)
    hit_bet = next((b for b in bets if b['combination'] == combo), None)
    
    if hit_bet:
        bet_amt = hit_bet['bet_amount']
        actual_payout = int((bet_amt / 100.0) * payout_per_100)
        profit = actual_payout - total_bet
        hit_status = "hit"
        msg = f"💮 【的中!!】 出目: {combo} | 投資: {total_bet:,}円 -> 払戻: {actual_payout:,}円 (利益: {profit:+,}円)"
    else:
        actual_payout = 0
        profit = - total_bet
        hit_status = "miss"
        msg = f"💀 【ハズレ】 出目: {combo} | 投資: {total_bet:,}円 -> 払戻: 0円 (損失: {profit:,}円)"
        
    db_manager.update_hit_focused_result(
        race_id=race_id,
        actual_result=combo,
        payout=actual_payout,
        profit=profit,
        hit_status=hit_status
    )
    
    return {
        'race_id': race_id,
        'venue_name': venue_name,
        'race_no': race_no,
        'deadline': deadline,
        'status': 'settled',
        'actual_result': combo,
        'hit_status': hit_status,
        'total_bet': total_bet,
        'payout': actual_payout,
        'profit': profit,
        'message': msg
    }


def backfill_today_results(target_date: str = None, max_workers: int = 8):
    today = datetime.date.today()
    if not target_date:
        target_date = today.strftime('%Y%m%d')
        
    now_time_str = datetime.datetime.now().strftime('%H:%M')
    
    print("=" * 80, flush=True)
    print(f" 🏁 [BACKFILL] 本日分レース結果の遡及決済を開始します (対象日: {target_date}, 現在時刻: {now_time_str}, 並列数: {max_workers})", flush=True)
    print("=" * 80, flush=True)

    unresolved_races = db_manager.get_unresolved_hit_focused_predictions(target_date)
    print(f"📋 データベース内の未確定レース件数: {len(unresolved_races)} 件\n", flush=True)

    if not unresolved_races:
        print("💡 精算対象となる未確定レースはありませんでした。")
        return

    settled_list = []
    skipped_list = []
    
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_race = {
            executor.submit(process_single_race, r, target_date): r
            for r in unresolved_races
        }
        
        for future in as_completed(future_to_race):
            res_item = future.result()
            if not res_item:
                continue
                
            v_name = res_item['venue_name']
            r_no = res_item['race_no']
            dl = res_item['deadline']
            msg = res_item['message']
            
            print(f"[{v_name} {r_no:2d}R (締切 {dl})] {msg}", flush=True)
            
            if res_item['status'] == 'settled':
                settled_list.append(res_item)
            else:
                skipped_list.append(res_item)

    elapsed = time.time() - t0

    # 3. 集計サマリー表示
    print("\n" + "=" * 80, flush=True)
    print(" 📊 【本日確定成績 サマリー】", flush=True)
    print("=" * 80, flush=True)
    
    if settled_list:
        total_settled = len(settled_list)
        hit_count = sum(1 for s in settled_list if s['hit_status'] == 'hit')
        miss_count = sum(1 for s in settled_list if s['hit_status'] == 'miss')
        no_bet_count = sum(1 for s in settled_list if s['hit_status'] == 'no_bet')
        total_invest = sum(s['total_bet'] for s in settled_list)
        total_payout = sum(s['payout'] for s in settled_list)
        net_profit = total_payout - total_invest
        hit_rate = (hit_count / (hit_count + miss_count) * 100.0) if (hit_count + miss_count) > 0 else 0.0
        roi = (total_payout / total_invest * 100.0) if total_invest > 0 else 0.0
        
        print(f"  ・確定レース数 : {total_settled} レース (内 投資対象: {hit_count + miss_count} R)")
        print(f"  ・的中レース数 : {hit_count} レース (的中率: {hit_rate:.1f}%)")
        print(f"  ・不的中レース : {miss_count} レース")
        print(f"  ・総投資金額   : {total_invest:,} 円")
        print(f"  ・総払戻金額   : {total_payout:,} 円")
        print(f"  ・確定純損益   : {net_profit:+,} 円")
        print(f"  ・実効回収率   : {roi:.1f} %")
    else:
        print("本日確定できたレースはありませんでした。")
        
    print(f"  ・未確定（未発走）残レース数: {len(skipped_list)} 件")
    print(f"  ・精算所要時間: {elapsed:.2f} 秒")
    print("=" * 80, flush=True)
    print("✨ バックフィル精算処理が完了しました！\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="本日分レース結果のバックフィル精算")
    parser.add_argument('--date', type=str, default=None, help="対象日 (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument('--workers', type=int, default=8, help="並列ワーカ数")
    args = parser.parse_args()
    
    backfill_today_results(target_date=args.date, max_workers=args.workers)
