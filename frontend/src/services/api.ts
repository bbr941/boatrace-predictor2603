import { supabase, isSupabaseConfigured } from '../lib/supabase';
import { 
  HitFocusedPrediction, 
  RacePrediction, 
  BetItem, 
  PredictionMode 
} from '../types';

// Format helper: normalize "2026-09-07" to ["20260907", "2026-09-07"]
export const getDateVariants = (d: string): string[] => {
  const clean = d.replace(/-/g, '');
  if (clean.length === 8) {
    const formatted = `${clean.slice(0, 4)}-${clean.slice(4, 6)}-${clean.slice(6, 8)}`;
    return [clean, formatted];
  }
  return [d];
};

export const getTodayDateStr = (): string => {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
};

// In-memory cache for bets to ensure sub-millisecond response
const betsCache: Record<string, BetItem[]> = {};

/**
 * 的中特化 (hit_focused_predictions) の指定日データ取得
 */
export async function getHitFocusedPredictions(dateStr: string): Promise<HitFocusedPrediction[]> {
  if (!isSupabaseConfigured) {
    return getMockHitFocusedPredictions(dateStr);
  }

  try {
    const variants = getDateVariants(dateStr);
    const { data, error } = await supabase
      .from('hit_focused_predictions')
      .select('*')
      .in('race_date', variants)
      .order('venue_code', { ascending: true })
      .order('race_no', { ascending: true });

    if (error) {
      console.warn('Supabase fetch error, falling back to mock:', error.message);
      return getMockHitFocusedPredictions(dateStr);
    }

    if (!data || data.length === 0) {
      // If today has no records yet, try to fetch the most recent records
      const { data: latestData } = await supabase
        .from('hit_focused_predictions')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(36);
      
      if (latestData && latestData.length > 0) {
        return latestData.sort((a, b) => a.venue_code - b.venue_code || a.race_no - b.race_no);
      }
      return getMockHitFocusedPredictions(dateStr);
    }

    return data as HitFocusedPrediction[];
  } catch (err) {
    console.error('Fetch error:', err);
    return getMockHitFocusedPredictions(dateStr);
  }
}

/**
 * 黄金ベースライン (race_predictions) の指定日データ取得
 */
export async function getGoldenPredictions(dateStr: string): Promise<RacePrediction[]> {
  if (!isSupabaseConfigured) {
    return getMockGoldenPredictions(dateStr);
  }

  try {
    const variants = getDateVariants(dateStr);
    const { data, error } = await supabase
      .from('race_predictions')
      .select('*')
      .in('race_date', variants)
      .order('venue_code', { ascending: true })
      .order('race_no', { ascending: true });

    if (error) {
      console.warn('Supabase fetch error, falling back to mock:', error.message);
      return getMockGoldenPredictions(dateStr);
    }

    if (!data || data.length === 0) {
      const { data: latestData } = await supabase
        .from('race_predictions')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(36);
      
      if (latestData && latestData.length > 0) {
        return latestData.sort((a, b) => a.venue_code - b.venue_code || a.race_no - b.race_no);
      }
      return getMockGoldenPredictions(dateStr);
    }

    return data as RacePrediction[];
  } catch (err) {
    console.error('Fetch error:', err);
    return getMockGoldenPredictions(dateStr);
  }
}

/**
 * 買い目詳細の高速取得 (キャッシュ利用)
 */
export async function getRaceBets(raceId: string, mode: PredictionMode): Promise<BetItem[]> {
  const cacheKey = `${mode}:${raceId}`;
  if (betsCache[cacheKey]) {
    return betsCache[cacheKey];
  }

  if (!isSupabaseConfigured) {
    const mockBets = generateMockBets(raceId);
    betsCache[cacheKey] = mockBets;
    return mockBets;
  }

  try {
    const tableName = mode === 'hit_focused' ? 'hit_focused_bets' : 'recommended_bets';
    const { data, error } = await supabase
      .from(tableName)
      .select('*')
      .eq('race_id', raceId)
      .order('bet_amount', { ascending: false })
      .order('prob', { ascending: false });

    if (error || !data || data.length === 0) {
      const mockBets = generateMockBets(raceId);
      betsCache[cacheKey] = mockBets;
      return mockBets;
    }

    const bets: BetItem[] = data.map((b: any) => ({
      combination: b.combination,
      bet_amount: b.bet_amount,
      prob: b.prob,
      odds: b.odds,
      ev: b.ev,
      expected_return: b.expected_return || Math.round((b.bet_amount * b.odds)),
      profit: b.profit ?? (Math.round(b.bet_amount * b.odds) - 1000)
    }));

    betsCache[cacheKey] = bets;
    return bets;
  } catch (err) {
    console.error('Fetch bets error:', err);
    return generateMockBets(raceId);
  }
}

// ----------------------------------------------------
// モックデータジェネレーター（キー未設定時・オフライン用）
// ----------------------------------------------------
function getMockHitFocusedPredictions(dateStr: string): HitFocusedPrediction[] {
  const venues = [
    { code: 1, name: '桐生' },
    { code: 3, name: '江戸川' },
    { code: 6, name: '浜名湖' },
    { code: 12, name: '住之江' },
  ];

  const list: HitFocusedPrediction[] = [];
  const cleanDate = dateStr.replace(/-/g, '');

  venues.forEach((venue) => {
    for (let r = 1; r <= 12; r++) {
      const raceId = `${cleanDate}_${venue.code}_${r}`;
      
      // 状況に応じたシミュレーション
      let status = 'hit_focused_go';
      let isResolved = false;
      let actualResult: string | null = null;
      let payout = 0;
      let profit = 0;
      let hitStatus: 'hit' | 'miss' | null = null;

      if (r <= 4) {
        // 的中レース (緑)
        isResolved = true;
        hitStatus = 'hit';
        actualResult = '1-2-3';
        payout = 2320;
        profit = 1320;
      } else if (r <= 8) {
        // ハズレレース (ダークトーン)
        isResolved = true;
        hitStatus = 'miss';
        actualResult = '3-1-5';
        payout = 0;
        profit = -1000;
      } else if (r <= 10) {
        // レース前 (投資GO)
        isResolved = false;
        hitStatus = null;
        status = 'hit_focused_go';
      } else {
        // スキップ / 見送り
        isResolved = false;
        status = 'gatekeeper_skipped';
      }

      list.push({
        id: venue.code * 100 + r,
        race_id: raceId,
        race_date: cleanDate,
        venue_code: venue.code,
        venue_name: venue.name,
        race_no: r,
        deadline_time: `${14 + Math.floor(r / 2)}:${(r % 2) * 30 + 15}`,
        top_boat: 1,
        max_p1: 0.62,
        prob_gap: 0.28,
        cluster_id: 1,
        cluster_name: 'イン超強水面',
        status: status,
        total_bet: 1000,
        bets_count: 5,
        min_return: 2320,
        max_return: 3840,
        min_profit: 1320,
        target_cum_prob: 0.52,
        actual_result: actualResult,
        payout: payout,
        profit: profit,
        is_resolved: isResolved,
        hit_status: hitStatus,
        created_at: new Date().toISOString()
      });
    }
  });

  return list;
}

function getMockGoldenPredictions(dateStr: string): RacePrediction[] {
  const venues = [
    { code: 1, name: '桐生' },
    { code: 3, name: '江戸川' },
    { code: 12, name: '住之江' },
  ];

  const list: RacePrediction[] = [];
  const cleanDate = dateStr.replace(/-/g, '');

  venues.forEach((venue) => {
    for (let r = 1; r <= 12; r++) {
      const raceId = `${cleanDate}_${venue.code}_${r}`;
      let status = r % 3 === 0 ? 'investment_go' : 'sniper_skipped';
      let isResolved = r <= 6;
      let actualResult = isResolved ? (r === 3 ? '1-3-2' : '2-1-4') : null;
      let payout = r === 3 ? 3450 : 0;
      let profit = r === 3 ? 2450 : (isResolved && status === 'investment_go' ? -1000 : 0);
      let hitStatus = isResolved && status === 'investment_go' ? (r === 3 ? 'hit' : 'miss') : null;

      list.push({
        id: venue.code * 100 + r,
        race_id: raceId,
        race_date: cleanDate,
        venue_code: venue.code,
        venue_name: venue.name,
        race_no: r,
        deadline_time: `${15 + Math.floor(r / 2)}:${(r % 2) * 25 + 10}`,
        top_boat: 1,
        max_p1: 0.71,
        prob_gap: 0.35,
        gatekeeper_passed: status === 'investment_go',
        cluster_id: 1,
        cluster_name: 'イン超強水面',
        status: status,
        source: 'auto',
        actual_result: actualResult,
        payout: payout,
        profit: profit,
        is_resolved: isResolved,
        hit_status: hitStatus,
        created_at: new Date().toISOString(),
        total_bet: 1000,
        bets_count: 4,
        min_return: 2200,
        max_return: 4500
      });
    }
  });

  return list;
}

function generateMockBets(raceId: string): BetItem[] {
  return [
    { combination: '1-2-3', bet_amount: 300, prob: 0.182, odds: 7.8, ev: 1.42, expected_return: 2340, profit: 1340 },
    { combination: '1-2-4', bet_amount: 200, prob: 0.125, odds: 11.5, ev: 1.44, expected_return: 2300, profit: 1300 },
    { combination: '1-3-2', bet_amount: 200, prob: 0.114, odds: 12.2, ev: 1.39, expected_return: 2440, profit: 1440 },
    { combination: '1-3-4', bet_amount: 200, prob: 0.089, odds: 15.6, ev: 1.38, expected_return: 3120, profit: 2120 },
    { combination: '1-4-2', bet_amount: 100, prob: 0.061, odds: 24.0, ev: 1.46, expected_return: 2400, profit: 1400 },
  ];
}
