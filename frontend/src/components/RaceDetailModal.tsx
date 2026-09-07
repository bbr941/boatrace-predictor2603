import React, { useEffect, useState } from 'react';
import { HitFocusedPrediction, RacePrediction, BetItem, PredictionMode } from '../types';
import { getRaceBets } from '../services/api';
import { X, CheckCircle, AlertCircle, Clock, Zap, DollarSign } from 'lucide-react';

interface RaceDetailModalProps {
  race: HitFocusedPrediction | RacePrediction | null;
  mode: PredictionMode;
  onClose: () => void;
}

// Boat color pill helper (1:白, 2:黒, 3:赤, 4:青, 5:黄, 6:緑)
const BoatTag: React.FC<{ num: string }> = ({ num }) => {
  const n = parseInt(num, 10);
  const colorMap: Record<number, string> = {
    1: 'bg-slate-100 text-slate-900 border-slate-300 font-bold',
    2: 'bg-slate-900 text-white border-slate-700 font-bold',
    3: 'bg-red-600 text-white border-red-500 font-bold',
    4: 'bg-blue-600 text-white border-blue-500 font-bold',
    5: 'bg-amber-400 text-slate-950 border-amber-300 font-bold',
    6: 'bg-emerald-600 text-white border-emerald-500 font-bold',
  };

  return (
    <span className={`inline-flex items-center justify-center w-5 h-5 rounded text-xs border ${colorMap[n] || 'bg-slate-800 text-white'}`}>
      {num}
    </span>
  );
};

const CombinationDisplay: React.FC<{ combo: string }> = ({ combo }) => {
  const boats = combo.split('-');
  return (
    <div className="flex items-center space-x-1">
      {boats.map((b, idx) => (
        <React.Fragment key={idx}>
          <BoatTag num={b} />
          {idx < boats.length - 1 && <span className="text-slate-500 text-xs font-mono">-</span>}
        </React.Fragment>
      ))}
    </div>
  );
};

export const RaceDetailModal: React.FC<RaceDetailModalProps> = ({ race, mode, onClose }) => {
  const [bets, setBets] = useState<BetItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!race) return;

    // Load bets
    let active = true;
    setLoading(true);
    getRaceBets(race.race_id, mode).then((data) => {
      if (active) {
        setBets(data);
        setLoading(false);
      }
    });

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      active = false;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [race, mode, onClose]);

  if (!race) return null;

  const isResolved = race.is_resolved;
  const isHit = race.hit_status === 'hit';
  const isMiss = race.hit_status === 'miss';
  const isInvested = race.status.includes('go') || isHit || isMiss;
  const totalBet = isInvested ? (race.total_bet || bets.reduce((s, b) => s + b.bet_amount, 0) || 1000) : 0;
  const payout = isInvested ? (race.payout || 0) : 0;
  const profit = isInvested
    ? (typeof race.profit === 'number' ? race.profit : (isResolved ? (isHit ? payout - totalBet : -totalBet) : 0))
    : 0;

  // Calculate synthetic odds: 1 / sum(1 / odds)
  const syntheticOdds = bets.length > 0
    ? (1 / bets.reduce((sum, b) => sum + (b.odds > 0 ? 1 / b.odds : 0), 0)).toFixed(2)
    : '2.50';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      {/* Modal Card */}
      <div 
        className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-4 py-3.5 sm:px-6 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <span className="text-xl">🚤</span>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base sm:text-lg font-bold text-white">
                  {race.venue_name} {race.race_no}R
                </h3>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 font-mono border border-blue-500/30">
                  {mode === 'hit_focused' ? '的中特化' : '黄金ベースライン'}
                </span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-slate-400 mt-0.5">
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3 h-3 text-slate-500" />
                  締切: {race.deadline_time}
                </span>
                <span>•</span>
                <span className="text-slate-300">{race.cluster_name || '標準水面'}</span>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-4">
          {/* Result Banner if Resolved */}
          {isResolved && (
            <div
              className={`p-3.5 rounded-xl border flex items-center justify-between ${
                isHit
                  ? 'bg-emerald-950/60 border-emerald-500/50 text-emerald-200'
                  : isMiss
                  ? 'bg-rose-950/40 border-rose-800/50 text-rose-300'
                  : 'bg-slate-950 border-slate-800 text-slate-400'
              }`}
            >
              <div className="flex items-center space-x-2.5">
                {isHit ? (
                  <div className="p-2 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                ) : isMiss ? (
                  <div className="p-2 rounded-full bg-rose-900/40 text-rose-400 border border-rose-700/50">
                    <AlertCircle className="w-5 h-5" />
                  </div>
                ) : (
                  <div className="p-2 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                    <AlertCircle className="w-5 h-5" />
                  </div>
                )}
                <div>
                  <div className="text-xs text-slate-400 font-semibold">
                    {isInvested ? '確定着順' : 'レース結果（見送り）'}
                  </div>
                  <div className="text-lg font-bold font-mono text-white flex items-center gap-2">
                    {race.actual_result ? (
                      <CombinationDisplay combo={race.actual_result} />
                    ) : (
                      '確定'
                    )}
                  </div>
                </div>
              </div>

              <div className="text-right">
                <div className="text-xs text-slate-400">確定損益</div>
                <div
                  className={`text-lg font-mono font-bold ${
                    profit > 0
                      ? 'text-emerald-400'
                      : profit < 0
                      ? 'text-rose-400'
                      : 'text-slate-400'
                  }`}
                >
                  {isInvested
                    ? (profit > 0 ? `+${profit.toLocaleString()}` : `${profit.toLocaleString()}円`)
                    : '0円 (見送り)'}
                </div>
                {payout > 0 && isInvested && (
                  <div className="text-[11px] text-slate-400 font-mono">
                    払戻: {payout.toLocaleString()}円
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Strategy / Prediction Info */}
          <div className="grid grid-cols-3 gap-2 text-center bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs">
            <div>
              <div className="text-slate-500 text-[10px]">総推奨配分</div>
              <div className="font-mono font-bold text-slate-200 text-sm mt-0.5">
                {totalBet.toLocaleString()}円
              </div>
            </div>
            <div className="border-x border-slate-800">
              <div className="text-slate-500 text-[10px]">目標合成オッズ</div>
              <div className="font-mono font-bold text-amber-300 text-sm mt-0.5">
                {syntheticOdds}倍
              </div>
            </div>
            <div>
              <div className="text-slate-500 text-[10px]">点数</div>
              <div className="font-mono font-bold text-blue-400 text-sm mt-0.5">
                {bets.length || race.bets_count || 5} 点
              </div>
            </div>
          </div>

          {/* Bets Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-blue-400" />
                <span>推奨買い目配分表（動的ダッチング）</span>
              </h4>
              <span className="text-[11px] text-slate-500 font-mono">全5点均等傾斜</span>
            </div>

            {loading ? (
              <div className="py-8 text-center text-xs text-slate-500">
                買い目データを読み込み中...
              </div>
            ) : bets.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-500 bg-slate-950 rounded-xl border border-slate-800">
                買い目データが登録されていません
              </div>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-xs text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-950 border-b border-slate-800 text-[11px] text-slate-400 font-mono">
                      <th className="py-2 px-3">組番</th>
                      <th className="py-2 px-3 text-right">配分額</th>
                      <th className="py-2 px-3 text-right">オッズ</th>
                      <th className="py-2 px-3 text-right">期待払戻</th>
                      <th className="py-2 px-3 text-center">結果</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                    {bets.map((bet, idx) => {
                      const isComboHit = isResolved && race.actual_result === bet.combination;
                      const expReturn = bet.expected_return || Math.round(bet.bet_amount * bet.odds);

                      return (
                        <tr
                          key={idx}
                          className={`transition ${
                            isComboHit
                              ? 'bg-emerald-950/60 font-semibold'
                              : 'hover:bg-slate-800/40'
                          }`}
                        >
                          <td className="py-2 px-3 font-mono">
                            <div className="flex items-center space-x-2">
                              <CombinationDisplay combo={bet.combination} />
                            </div>
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-slate-200">
                            {bet.bet_amount.toLocaleString()}円
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-amber-300 font-medium">
                            {bet.odds.toFixed(1)}
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-emerald-300">
                            {expReturn.toLocaleString()}円
                          </td>
                          <td className="py-2 px-3 text-center">
                            {isComboHit ? (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500 text-slate-950">
                                的中
                              </span>
                            ) : (
                              <span className="text-slate-600 text-[11px]">-</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-4 py-3 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1 font-mono text-[11px]">
            <DollarSign className="w-3.5 h-3.5 text-amber-400" />
            回収想定: 払戻金の均等化済み
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition font-medium"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};
