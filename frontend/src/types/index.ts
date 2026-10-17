export type PredictionMode = 'hit_focused' | 'golden_baseline';
export type ViewLayout = 'grid' | 'timeline';

export interface HitFocusedPrediction {
  id: number;
  race_id: string;
  race_date: string;
  venue_code: number;
  venue_name: string;
  race_no: number;
  deadline_time: string;
  top_boat?: number | null;
  max_p1?: number | null;
  prob_gap?: number | null;
  cluster_id?: number | null;
  cluster_name?: string | null;
  status: string;
  total_bet: number;
  bets_count: number;
  min_return: number;
  max_return: number;
  min_profit: number;
  target_cum_prob: number;
  actual_result?: string | null;
  payout: number;
  profit: number;
  is_resolved: boolean;
  hit_status?: 'hit' | 'miss' | null | string;
  is_gogo_lamp?: boolean | null;
  is_chance_lamp?: boolean | null;
  expected_roi?: number | null;
  motor_rate?: number | null;
  synthetic_odds?: number | null;
  created_at?: string;
}

export interface HitFocusedBet {
  id?: number;
  race_id: string;
  combination: string;
  bet_amount: number;
  prob: number;
  odds: number;
  ev: number;
  expected_return: number;
  profit?: number;
}

export interface RacePrediction {
  id: number;
  race_id: string;
  race_date: string;
  venue_code: number;
  venue_name: string;
  race_no: number;
  deadline_time: string;
  top_boat?: number | null;
  max_p1?: number | null;
  prob_gap?: number | null;
  gatekeeper_passed?: boolean;
  cluster_id?: number | null;
  cluster_name?: string | null;
  status: string;
  source?: string;
  actual_result?: string | null;
  payout: number;
  profit: number;
  is_resolved: boolean;
  hit_status?: 'hit' | 'miss' | null | string;
  created_at?: string;
  // Computed or joined
  total_bet?: number;
  bets_count?: number;
  min_return?: number;
  max_return?: number;
}

export interface RecommendedBet {
  id?: number;
  race_id: string;
  combination: string;
  bet_amount: number;
  prob: number;
  odds: number;
  ev: number;
  expected_return: number;
}

export interface BetItem {
  combination: string;
  bet_amount: number;
  prob: number;
  odds: number;
  ev: number;
  expected_return: number;
  profit?: number;
}

export interface VenueGroup<T> {
  venue_code: number;
  venue_name: string;
  races: T[];
}

export interface DashboardStats {
  totalRaces: number;
  investmentRaces: number;
  resolvedRaces: number;
  hitCount: number;
  missCount: number;
  hitRate: number;
  totalBet: number;
  totalPayout: number;
  netProfit: number;
  recoveryRate: number;
}

/**
 * 的中特化において黄金ベースライン緩和条件（回収率100%超）に達しているか判定
 * 
 * 🎰 【Tier 1: GOGO!ランプ (超激アツ)】
 * 条件: P1 >= 55.0% & モーター2連率 >= 40.0% & 合成期待回収率 >= 1.15
 * 実績: 的中率 53.9% / 回収率 110.6%
 * 
 * ⭐ 【Tier 2: CHANCEランプ (チャンス)】
 * 条件: P1 >= 55.0% & モーター2連率 >= 36.0% & 合成オッズ >= 2.0 & 期待回収率 >= 1.05
 * 実績: 的中率 39.6% / 回収率 101.3% (GOGO含む全体では回収率 101.9%)
 */
export function isGogoLampActive(race: HitFocusedPrediction | RacePrediction): boolean {
  if ('is_gogo_lamp' in race && (race as HitFocusedPrediction).is_gogo_lamp !== undefined && (race as HitFocusedPrediction).is_gogo_lamp !== null) {
    return Boolean((race as HitFocusedPrediction).is_gogo_lamp);
  }
  
  // フォールバック判定 (P1 >= 55% & Motor >= 40% & ExpROI >= 1.15)
  const p1 = race.max_p1 || 0;
  const mot = (race as HitFocusedPrediction).motor_rate || 0;
  const expRoi = (race as HitFocusedPrediction).expected_roi || 0;

  return p1 >= 0.55 && mot >= 40.0 && expRoi >= 1.15;
}

export function isChanceLampActive(race: HitFocusedPrediction | RacePrediction): boolean {
  // GOGOランプが点灯している場合はCHANCEランプは点灯しない (Tier 1 優先)
  if (isGogoLampActive(race)) return false;

  if ('is_chance_lamp' in race && (race as HitFocusedPrediction).is_chance_lamp !== undefined && (race as HitFocusedPrediction).is_chance_lamp !== null) {
    return Boolean((race as HitFocusedPrediction).is_chance_lamp);
  }

  // フォールバック判定 (P1 >= 55% & Motor >= 36% & 合成オッズ >= 2.0 & ExpROI >= 1.05)
  const p1 = race.max_p1 || 0;
  const mot = (race as HitFocusedPrediction).motor_rate || 0;
  const expRoi = (race as HitFocusedPrediction).expected_roi || 0;
  const synOdds = (race as HitFocusedPrediction).synthetic_odds || 
    ((race as HitFocusedPrediction).total_bet && (race as HitFocusedPrediction).min_return 
      ? (race as HitFocusedPrediction).min_return / (race as HitFocusedPrediction).total_bet 
      : 0);

  return p1 >= 0.55 && mot >= 36.0 && synOdds >= 2.0 && expRoi >= 1.05;
}

export function getLampTier(race: HitFocusedPrediction | RacePrediction): 'gogo' | 'chance' | 'none' {
  if (isGogoLampActive(race)) return 'gogo';
  if (isChanceLampActive(race)) return 'chance';
  return 'none';
}


