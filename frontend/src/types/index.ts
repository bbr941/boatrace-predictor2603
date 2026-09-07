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
