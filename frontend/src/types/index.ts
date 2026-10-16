export type PredictionMode = 'hit_focused' | 'golden_baseline';

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
 * 的中特化において回収率が100%を超える条件に達しているか判定 (GOGOランプ点灯判定)
 * - レース前: 最低払戻額が投資額を上回る（min_return > total_bet または min_profit > 0）、つまりトリガミなしで回収率100%超が確定している激熱状態
 * - レース後: 的中し回収率100%超を達成（profit > 0 または payout >= total_bet）
 */
export function isGogoLampActive(race: HitFocusedPrediction | RacePrediction): boolean {
  if (race.is_resolved) {
    return race.hit_status === 'hit' && (race.profit > 0 || (race.payout > 0 && race.payout >= (race.total_bet || 1000)));
  }
  const minReturn = (race as HitFocusedPrediction).min_return || 0;
  const totalBet = race.total_bet || 0;
  const minProfit = (race as HitFocusedPrediction).min_profit ?? 0;
  
  if (totalBet > 0 && minReturn > totalBet) return true;
  if (minProfit > 0) return true;
  return false;
}

