import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { PredictionMode, HitFocusedPrediction, RacePrediction, DashboardStats } from './types';
import { getHitFocusedPredictions, getGoldenPredictions, getTodayDateStr } from './services/api';
import { Header } from './components/Header';
import { StrategyGuide } from './components/StrategyGuide';
import { SummaryStats } from './components/SummaryStats';
import { VenueGrid } from './components/VenueGrid';
import { RaceDetailModal } from './components/RaceDetailModal';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const [mode, setMode] = useState<PredictionMode>('hit_focused');
  const [selectedDate, setSelectedDate] = useState<string>(getTodayDateStr());
  const [races, setRaces] = useState<(HitFocusedPrediction | RacePrediction)[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedRace, setSelectedRace] = useState<HitFocusedPrediction | RacePrediction | null>(null);

  // Fetch races function
  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      if (mode === 'hit_focused') {
        const data = await getHitFocusedPredictions(selectedDate);
        setRaces(data);
      } else {
        const data = await getGoldenPredictions(selectedDate);
        setRaces(data);
      }
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Error loading data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [mode, selectedDate]);

  // Initial load and periodic refresh (every 30s)
  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Compute stats for current mode & date
  const stats = useMemo<DashboardStats>(() => {
    const totalRaces = races.length;
    // Investment target races
    const targetRaces = races.filter((r) => r.status.includes('go') || (r as any).gatekeeper_passed);
    const investmentRaces = targetRaces.length;

    // Resolved among investment races
    const resolvedTargets = targetRaces.filter((r) => r.is_resolved);
    const resolvedRaces = resolvedTargets.length;

    const hitCount = resolvedTargets.filter((r) => r.hit_status === 'hit').length;
    const missCount = resolvedTargets.filter((r) => r.hit_status === 'miss').length;
    const hitRate = (hitCount + missCount) > 0 ? (hitCount / (hitCount + missCount)) * 100 : 0;

    const totalBet = resolvedTargets.reduce((sum, r) => sum + (r.total_bet || 1000), 0);
    const totalPayout = resolvedTargets.reduce((sum, r) => sum + (r.payout || 0), 0);
    const netProfit = resolvedTargets.reduce((sum, r) => sum + (r.profit || (r.hit_status === 'hit' ? (r.payout || 0) - (r.total_bet || 1000) : -(r.total_bet || 1000))), 0);
    const recoveryRate = totalBet > 0 ? (totalPayout / totalBet) * 100 : 0;

    return {
      totalRaces,
      investmentRaces,
      resolvedRaces,
      hitCount,
      missCount,
      hitRate,
      totalBet,
      totalPayout,
      netProfit,
      recoveryRate,
    };
  }, [races]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* 1. Header with Mode & Date & Refresh */}
      <Header
        mode={mode}
        onModeChange={setMode}
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
        onRefresh={loadData}
        isLoading={isLoading}
        lastUpdated={lastUpdated}
      />

      {/* 2. Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 space-y-4 sm:space-y-6">
        {/* Strategy Guide Banner (Accordion) */}
        <StrategyGuide />

        {/* Dashboard Summary Stats */}
        <SummaryStats stats={stats} />

        {/* Loading Indicator bar */}
        {isLoading && races.length === 0 && (
          <div className="py-20 flex flex-col items-center justify-center space-y-3">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <p className="text-xs text-slate-400 font-mono tracking-wider">
              Supabaseからデータを取得中...
            </p>
          </div>
        )}

        {/* Venue Grid Panels (4 columns PC / 2-3 columns mobile) */}
        <VenueGrid
          races={races}
          selectedRaceId={selectedRace?.race_id}
          onSelectRace={(race) => setSelectedRace(race)}
        />
      </main>

      {/* 3. Fast Detail Modal */}
      <RaceDetailModal
        race={selectedRace}
        mode={mode}
        onClose={() => setSelectedRace(null)}
      />

      {/* 4. Footer */}
      <footer className="bg-slate-950 border-t border-slate-900 py-4 px-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>BOATRACE AI Edge Prediction System</span>
          <span className="text-slate-600">
            Powered by Cloudflare Pages + Vite React + Supabase Direct REST
          </span>
        </div>
      </footer>
    </div>
  );
};
