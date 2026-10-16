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
  is_gogo_lamp?: boolean | null;
  expected_roi?: number | null;
  motor_rate?: number | null;
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
 * 的中特化において黄金ベースライン緩和条件（回収率100%超）に達しているか判定 (GOGOランプ点灯判定)
 * 【バックテスト検証済基準】:
 * 1. 本命信頼度: P1 >= 55.0%
 * 2. 好調モーター: 1号艇モーター2連率 >= 40.0%
 * 3. 合成期待回収率: Expected ROI >= 1.15 (115%以上)
 * （2026年実績: 的中率 53.9% / 回収率 110.6% / 8ヶ月中7ヶ月プラス収支）
 */
export function isGogoLampActive(race: HitFocusedPrediction | RacePrediction): boolean {
  // DBに保存された is_gogo_lamp フラグがある場合はそれを優先
  if ('is_gogo_lamp' in race && (race as HitFocusedPrediction).is_gogo_lamp !== undefined && (race as HitFocusedPrediction).is_gogo_lamp !== null) {
    return Boolean((race as HitFocusedPrediction).is_gogo_lamp);
  }
  
  // フォールバック判定 (P1 >= 55% & Motor >= 40% & ExpROI >= 1.15)
  const p1 = race.max_p1 || 0;
  const mot = (race as HitFocusedPrediction).motor_rate || 0;
  const expRoi = (race as HitFocusedPrediction).expected_roi || 0;

  if (p1 >= 0.55 && mot >= 40.0 && expRoi >= 1.15) {
    return true;
  }

  return false;
}

