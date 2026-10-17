import React, { useState, useEffect, useMemo } from 'react';
import { HitFocusedPrediction, RacePrediction, BetItem, PredictionMode, isGogoLampActive, isChanceLampActive } from '../types';
import { getBatchRaceBets } from '../services/api';
import { CombinationDisplay } from './CombinationDisplay';
import { Clock, Flame, CheckCircle, AlertCircle, Sparkles, Filter, ArrowUpDown, ChevronRight, Loader2, Star } from 'lucide-react';

interface TimelineFeedProps {
  races: (HitFocusedPrediction | RacePrediction)[];
  mode: PredictionMode;
  onSelectRace: (race: HitFocusedPrediction | RacePrediction) => void;
}

type SortType = 'latest' | 'upcoming';
type FilterType = 'all' | 'gogo' | 'chance' | 'invested' | 'unresolved' | 'hit';

export const TimelineFeed: React.FC<TimelineFeedProps> = ({ races, mode, onSelectRace }) => {
  const [betsMap, setBetsMap] = useState<Record<string, BetItem[]>>({});
  const [isLoadingBets, setIsLoadingBets] = useState<boolean>(true);
  const [sortType, setSortType] = useState<SortType>('latest');
  const [filterType, setFilterType] = useState<FilterType>('all');

  // Load all bets in batch for instant inline display without clicking
  useEffect(() => {
    let active = true;
    const raceIds = races.map((r) => r.race_id);

    if (raceIds.length === 0) {
      setBetsMap({});
      setIsLoadingBets(false);
      return;
    }

    setIsLoadingBets(true);
    getBatchRaceBets(raceIds, mode)
      .then((data) => {
        if (active) {
          setBetsMap(data);
          setIsLoadingBets(false);
        }
      })
      .catch((err) => {
        console.error('Error fetching batch bets:', err);
        if (active) setIsLoadingBets(false);
      });

    return () => {
      active = false;
    };
  }, [races, mode]);

  // Helper for sorting key
  const getSortScore = (r: HitFocusedPrediction | RacePrediction, type: SortType): number => {
    if (type === 'latest') {
      // 1. If created_at is available, use it (newest timestamp first)
      if (r.created_at) {
        return new Date(r.created_at).getTime();
      }
      // 2. Fallback to deadline_time (latest time first)
      if (r.deadline_time) {
        const cleaned = r.deadline_time.replace(/[^0-9:]/g, '');
        const [h, m] = cleaned.split(':').map(Number);
        if (!isNaN(h) && !isNaN(m)) {
          return h * 60 + m;
        }
      }
      return 0;
    } else {
      // 'upcoming': Closest upcoming deadline first
      // Put unresolved upcoming first, then past
      let timeVal = 9999;
      if (r.deadline_time) {
        const cleaned = r.deadline_time.replace(/[^0-9:]/g, '');
        const [h, m] = cleaned.split(':').map(Number);
        if (!isNaN(h) && !isNaN(m)) {
          timeVal = h * 60 + m;
        }
      }
      return (r.is_resolved ? 100000 : 0) + timeVal;
    }
  };

  // Filter & Sort
  const processedRaces = useMemo(() => {
    let list = [...races];

    // Filter
    if (filterType === 'gogo') {
      list = list.filter((r) => isGogoLampActive(r));
    } else if (filterType === 'chance') {
      list = list.filter((r) => isGogoLampActive(r) || isChanceLampActive(r));
    } else if (filterType === 'invested') {
      list = list.filter((r) => r.status.includes('go') || r.hit_status === 'hit' || r.hit_status === 'miss');
    } else if (filterType === 'unresolved') {
      list = list.filter((r) => !r.is_resolved);
    } else if (filterType === 'hit') {
      list = list.filter((r) => r.hit_status === 'hit');
    }

    // Sort
    list.sort((a, b) => {
      if (sortType === 'latest') {
        return getSortScore(b, 'latest') - getSortScore(a, 'latest');
      } else {
        return getSortScore(a, 'upcoming') - getSortScore(b, 'upcoming');
      }
    });

    return list;
  }, [races, sortType, filterType]);

  return (
    <div className="space-y-4">
      {/* Control Toolbar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 sm:p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3 shadow-md">
        {/* Sort selector */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 text-xs text-slate-400 font-semibold mr-1">
            <ArrowUpDown className="w-3.5 h-3.5 text-blue-400" />
            <span>並び順:</span>
          </div>
          <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-medium">
            <button
              onClick={() => setSortType('latest')}
              className={`px-2.5 py-1 rounded-md transition ${
                sortType === 'latest'
                  ? 'bg-blue-600 text-white font-bold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              ⚡ 最新公開・配信順 (最新が上)
            </button>
            <button
              onClick={() => setSortType('upcoming')}
              className={`px-2.5 py-1 rounded-md transition ${
                sortType === 'upcoming'
                  ? 'bg-blue-600 text-white font-bold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              ⏰ 締切が近い順
            </button>
          </div>
        </div>

        {/* Filter chips */}
        <div className="flex items-center flex-wrap gap-1.5 text-xs">
          <div className="flex items-center space-x-1 text-xs text-slate-400 font-semibold mr-1">
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <span>絞り込み:</span>
          </div>
          {[
            { id: 'all', label: 'すべて' },
            { id: 'gogo', label: '🎰 GOGO! (回収110%)', activeClass: 'bg-pink-600 text-white font-bold shadow-sm' },
            { id: 'chance', label: '⭐ CHANCE以上 (回収100%超)', activeClass: 'bg-amber-500 text-slate-950 font-black shadow-sm' },
            { id: 'invested', label: '🎯 参戦レース' },
            { id: 'unresolved', label: '⏳ 未確定' },
            { id: 'hit', label: '💮 的中のみ' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setFilterType(item.id as FilterType)}
              className={`px-2.5 py-1 rounded-lg transition font-medium ${
                filterType === item.id
                  ? (item.activeClass || 'bg-slate-700 text-white font-bold')
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800/80 hover:bg-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Race Count Summary */}
      <div className="flex items-center justify-between text-xs text-slate-400 px-1">
        <span>
          表示中: <strong className="text-white font-mono">{processedRaces.length}</strong> レース
        </span>
        <span className="text-slate-500 text-[11px]">
          ※ 各カード内にすべての買い目・オッズ・配分額を常時表示中
        </span>
      </div>

      {/* Empty State */}
      {processedRaces.length === 0 && (
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl py-16 text-center text-slate-400 space-y-2">
          <p className="text-sm font-semibold text-slate-300">該当するレースがありません</p>
          <p className="text-xs text-slate-500">絞り込み条件を変更するか、日付をご確認ください。</p>
        </div>
      )}

      {/* Timeline Cards Feed */}
      <div className="space-y-3.5">
        {processedRaces.map((race) => {
          const isGogo = isGogoLampActive(race);
          const isChance = isChanceLampActive(race);
          const isResolved = race.is_resolved;
          const hitStatus = race.hit_status;
          const isHit = hitStatus === 'hit';
          const isMiss = hitStatus === 'miss';
          const bets = betsMap[race.race_id] || [];
          const totalBet = race.total_bet || (bets.length > 0 ? bets.reduce((s, b) => s + b.bet_amount, 0) : 1000);
          const profit = race.profit || 0;
          const p1 = race.max_p1 ? (race.max_p1 * 100).toFixed(1) : null;
          const motor = (race as HitFocusedPrediction).motor_rate;
          const expRoi = (race as HitFocusedPrediction).expected_roi;
          const synOdds = (race as HitFocusedPrediction).synthetic_odds ||
            (totalBet > 0 && (race as HitFocusedPrediction).min_return 
              ? (race as HitFocusedPrediction).min_return / totalBet 
              : 0);

          // Card Border / Glow
          let cardBorder = "border-slate-800/90 hover:border-slate-700 bg-slate-900/70";
          if (isResolved) {
            if (isHit) {
              cardBorder = profit < 0
                ? "border-amber-500/50 bg-gradient-to-br from-amber-950/30 via-slate-900/90 to-slate-950 shadow-md shadow-amber-950/20"
                : "border-emerald-500/60 bg-gradient-to-br from-emerald-950/40 via-slate-900/90 to-slate-950 shadow-lg shadow-emerald-950/20";
            } else if (isMiss) {
              cardBorder = "border-slate-800/80 bg-slate-950/90";
            }
          } else if (isGogo) {
            cardBorder = "border-pink-500/70 bg-gradient-to-br from-purple-950/30 via-slate-900/90 to-pink-950/20 shadow-xl shadow-pink-950/35 ring-1 ring-pink-500/50";
          } else if (isChance) {
            cardBorder = "border-amber-400/70 bg-gradient-to-br from-amber-950/30 via-slate-900/90 to-yellow-950/20 shadow-lg shadow-amber-950/30 ring-1 ring-amber-400/50";
          } else if (race.status.includes('go')) {
            cardBorder = "border-blue-500/50 bg-gradient-to-br from-blue-950/30 via-slate-900/90 to-slate-950";
          }

          return (
            <div
              key={race.race_id}
              className={`rounded-xl border transition-all duration-200 overflow-hidden ${cardBorder}`}
            >
              {/* Card Header Bar */}
              <div className="p-3 sm:p-4 bg-slate-950/60 border-b border-white/5 flex flex-wrap items-center justify-between gap-2.5">
                {/* Left: Venue, Race No, Deadline */}
                <div className="flex items-center flex-wrap gap-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-blue-600 text-white shadow-sm">
                      {race.venue_name}
                    </span>
                    <span className="text-base sm:text-lg font-black font-mono text-white">
                      {race.race_no}R
                    </span>
                  </div>

                  <span className="flex items-center gap-1 text-xs font-mono text-slate-300 bg-slate-900/90 px-2 py-0.5 rounded border border-slate-800">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span>{race.deadline_time || '締切未定'} 締切</span>
                  </span>

                  <span className="text-[11px] text-slate-400 hidden sm:inline-block">
                    {race.cluster_name || '標準水面'}
                  </span>
                </div>

                {/* Right: GOGO / CHANCE Badge / Hit Result Badge */}
                <div className="flex items-center space-x-2">
                  {isGogo ? (
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-black italic bg-gradient-to-r from-pink-600 to-purple-600 text-white border border-pink-400/60 shadow-[0_0_10px_rgba(236,72,153,0.7)] animate-pulse flex items-center gap-1">
                      <span>🎰 GOGO!</span>
                      <span className="not-italic text-[10px] text-amber-200 font-bold hidden sm:inline">
                        超激アツ (回収110%)
                      </span>
                    </span>
                  ) : isChance ? (
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-black italic bg-gradient-to-r from-amber-500 to-orange-500 text-slate-950 border border-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.6)] animate-pulse flex items-center gap-1">
                      <Star className="w-3 h-3 fill-slate-950 text-slate-950" />
                      <span>CHANCE</span>
                      <span className="not-italic text-[10px] text-slate-950 font-bold hidden sm:inline">
                        好機 (回収101%)
                      </span>
                    </span>
                  ) : null}

                  {isResolved ? (
                    isHit ? (
                      profit < 0 ? (
                        <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1 font-mono">
                          <CheckCircle className="w-3.5 h-3.5 text-amber-400" />
                          <span>💮 的中(ガミ) -{Math.abs(profit).toLocaleString()}円</span>
                        </span>
                      ) : (
                        <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1 font-mono">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                          <span>💮 的中! {profit > 0 ? `+${profit.toLocaleString()}` : '±0'}円</span>
                        </span>
                      )
                    ) : isMiss ? (
                      <span className="px-2 py-1 rounded-lg text-xs font-medium bg-slate-900 text-slate-400 border border-slate-800 font-mono">
                        💀 収支: -{Math.abs(profit).toLocaleString()}円
                      </span>
                    ) : (
                      <span className="px-2 py-1 rounded-lg text-xs font-medium bg-slate-900 text-slate-500 border border-slate-800">
                        見送り終了
                      </span>
                    )
                  ) : race.status.includes('go') ? (
                    <span className="px-2.5 py-0.5 rounded-lg text-xs font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30 font-mono">
                      🎯 投資GO ({bets.length || race.bets_count || 5}点 / {totalBet.toLocaleString()}円)
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-900 text-slate-500 border border-slate-800">
                      見送り判定
                    </span>
                  )}

                  {/* Open Detail Modal Button */}
                  <button
                    onClick={() => onSelectRace(race)}
                    className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition"
                    title="詳細モーダルを開く"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Subheader Metrics Bar */}
              <div className="px-3 sm:px-4 py-1.5 bg-slate-950/30 border-b border-white/5 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
                <div className="flex items-center space-x-3 font-mono">
                  {p1 && (
                    <span>
                      本命信頼度(P1): <strong className="text-slate-200">{p1}%</strong>
                    </span>
                  )}
                  {motor !== undefined && motor !== null && motor > 0 && (
                    <span>
                      1号艇モーター: <strong className="text-slate-200">{motor.toFixed(1)}%</strong>
                    </span>
                  )}
                  {expRoi !== undefined && expRoi !== null && expRoi > 0 && (
                    <span>
                      合成期待回収: <strong className={expRoi >= 1.15 ? 'text-pink-300' : expRoi >= 1.05 ? 'text-amber-300' : 'text-slate-200'}>{(expRoi * 100).toFixed(0)}%</strong>
                    </span>
                  )}
                  {synOdds !== undefined && synOdds > 0 && (
                    <span>
                      合成オッズ: <strong className="text-slate-200">{synOdds.toFixed(1)}倍</strong>
                    </span>
                  )}
                </div>

                {isResolved && race.actual_result && (
                  <div className="flex items-center space-x-1.5 font-mono text-xs">
                    <span className="text-slate-400">確定結果:</span>
                    <CombinationDisplay combo={race.actual_result} size="sm" />
                    {race.payout > 0 && (
                      <span className="text-emerald-300 font-bold ml-1">
                        払戻: {race.payout.toLocaleString()}円
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* INLINE BETS AREA: クリックしなくても買い目が一目瞭然 */}
              <div className="p-3 sm:p-4">
                {isLoadingBets && bets.length === 0 ? (
                  <div className="py-6 flex items-center justify-center space-x-2 text-xs text-slate-500 font-mono">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                    <span>買い目データを読み込み中...</span>
                  </div>
                ) : bets.length === 0 ? (
                  <div className="py-3 px-4 bg-slate-950/60 rounded-lg border border-slate-800 text-xs text-slate-500 text-center font-mono">
                    買い目データなし（条件未達の見送りレース）
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-300 flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-cyan-400" />
                        <span>推奨買い目配分（全{bets.length}点）</span>
                      </span>
                      <span className="text-[11px] text-slate-400 font-mono">
                        総配分額: <strong className="text-slate-200">{totalBet.toLocaleString()}円</strong>
                      </span>
                    </div>

                    {/* Responsive Bets Grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {bets.map((bet, idx) => {
                        const isComboHit = isResolved && race.actual_result === bet.combination;
                        const expReturn = bet.expected_return || Math.round(bet.bet_amount * bet.odds);

                        return (
                          <div
                            key={idx}
                            className={`p-2 rounded-lg border transition ${
                              isComboHit
                                ? 'bg-emerald-950/70 border-emerald-400/80 ring-1 ring-emerald-400/60 shadow-md shadow-emerald-950/40'
                                : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <CombinationDisplay combo={bet.combination} size="sm" />
                              <span className="text-xs font-mono font-bold text-amber-300">
                                {bet.odds.toFixed(1)}倍
                              </span>
                            </div>

                            <div className="flex items-center justify-between text-[11px] mt-1.5 pt-1 border-t border-white/5 font-mono">
                              <span className="text-slate-300">
                                {bet.bet_amount.toLocaleString()}円
                              </span>
                              <span className={isComboHit ? 'text-emerald-300 font-bold' : 'text-slate-400'}>
                                {isComboHit ? '💮 的中!' : `➔ ${expReturn.toLocaleString()}円`}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
