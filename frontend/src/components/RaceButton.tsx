import React from 'react';
import { HitFocusedPrediction, RacePrediction } from '../types';
import { Clock } from 'lucide-react';

interface RaceButtonProps {
  race: HitFocusedPrediction | RacePrediction;
  isSelected?: boolean;
  onClick: () => void;
}

export const RaceButton: React.FC<RaceButtonProps> = ({ race, isSelected, onClick }) => {
  const isResolved = race.is_resolved;
  const hitStatus = race.hit_status;
  const isGo = race.status.includes('go');
  const profit = race.profit || 0;
  const totalBet = race.total_bet || 1000;
  const betsCount = race.bets_count || 5;
  const minProfit = (race as HitFocusedPrediction).min_profit || 1320;

  // Visual Styles based on status
  let cardStyle = "bg-slate-900/40 border-slate-800/80 text-slate-500 hover:border-slate-700";
  let badgeStyle = "text-slate-500 bg-slate-800/50";
  let contentNode: React.ReactNode = null;

  if (isResolved) {
    if (hitStatus === 'hit') {
      // 💮 的中! (鮮やかな緑色ハイライト)
      cardStyle = "bg-gradient-to-b from-emerald-950/80 to-emerald-900/40 border-emerald-500/60 text-emerald-200 shadow-lg shadow-emerald-950/40 hover:border-emerald-400 ring-1 ring-emerald-500/30";
      badgeStyle = "text-emerald-300 bg-emerald-900/60 border border-emerald-500/40";
      contentNode = (
        <div className="flex flex-col items-center justify-center py-1">
          <div className="flex items-center space-x-1 text-xs font-bold text-emerald-300">
            <span>💮 的中!</span>
            <span className="font-mono">+{profit.toLocaleString()}円</span>
          </div>
          {race.actual_result && (
            <span className="text-[10px] text-emerald-400/80 font-mono tracking-wider">
              結果: {race.actual_result}
            </span>
          )}
        </div>
      );
    } else if (hitStatus === 'miss') {
      // 💀 ハズレ (落ち着いたダークトーン)
      cardStyle = "bg-slate-950/80 border-slate-800/90 text-slate-400 hover:border-slate-700 hover:bg-slate-900/60";
      badgeStyle = "text-slate-500 bg-slate-900 border border-slate-800";
      const lossAmount = Math.abs(profit) || totalBet;
      contentNode = (
        <div className="flex flex-col items-center justify-center py-1">
          <div className="flex items-center space-x-1 text-xs font-medium text-slate-400">
            <span>💀 収支:</span>
            <span className="font-mono font-semibold text-rose-400/90">
              -{lossAmount.toLocaleString()}円
            </span>
          </div>
          {race.actual_result && (
            <span className="text-[10px] text-slate-500 font-mono">
              結果: {race.actual_result}
            </span>
          )}
        </div>
      );
    } else {
      // 確定済みだが非参戦（見送り終了）
      cardStyle = "bg-slate-950/40 border-slate-900 text-slate-600 hover:border-slate-800";
      badgeStyle = "text-slate-600 bg-slate-950 border border-slate-900";
      contentNode = (
        <div className="flex flex-col items-center justify-center py-1">
          <span className="text-xs text-slate-500">見送り終了</span>
          {race.actual_result && (
            <span className="text-[10px] text-slate-600 font-mono">
              結果: {race.actual_result}
            </span>
          )}
        </div>
      );
    }
  } else {
    // レース前
    if (isGo) {
      // 🎯 レース前 (投資GO)
      cardStyle = "bg-gradient-to-b from-blue-950/70 to-slate-900 border-blue-500/50 text-blue-100 hover:border-blue-400 shadow-md shadow-blue-950/30 hover:scale-[1.01]";
      badgeStyle = "text-blue-300 bg-blue-900/60 border border-blue-500/40";
      contentNode = (
        <div className="flex flex-col items-center justify-center py-1">
          <div className="flex items-center space-x-1 text-[11px] sm:text-xs font-semibold text-blue-200">
            <span>🎯 {betsCount}点 ({totalBet.toLocaleString()}円)</span>
          </div>
          <div className="text-[11px] text-amber-300 font-mono font-bold">
            予想: +{minProfit.toLocaleString()}円
          </div>
        </div>
      );
    } else {
      // レース前・スキップ / 判定外
      cardStyle = "bg-slate-950/40 border-slate-900 text-slate-600 hover:border-slate-800 hover:text-slate-500";
      badgeStyle = "text-slate-600 bg-slate-950 border border-slate-900";
      contentNode = (
        <div className="text-center py-1.5">
          <span className="text-xs text-slate-600">
            {race.status === 'gatekeeper_skipped' ? 'GK見送り' : '見送り (Pass)'}
          </span>
        </div>
      );
    }
  }

  return (
    <button
      onClick={onClick}
      className={`relative w-full rounded-xl border p-2.5 flex flex-col justify-between transition duration-150 text-left min-h-[78px] ${cardStyle} ${
        isSelected ? 'ring-2 ring-cyan-400 ring-offset-2 ring-offset-slate-950' : ''
      }`}
    >
      {/* Card Header: Race No & Deadline */}
      <div className="flex items-center justify-between w-full border-b border-white/5 pb-1">
        <span className="font-bold font-mono text-xs text-slate-200">
          {race.race_no}R
        </span>

        <div className="flex items-center space-x-1 text-[10px] font-mono">
          <Clock className="w-3 h-3 text-slate-400" />
          <span className="text-slate-300">{race.deadline_time || '--:--'}</span>
        </div>
      </div>

      {/* Main Status Display */}
      <div className="my-auto w-full">
        {contentNode}
      </div>

      {/* Cluster / Status Pill */}
      <div className="w-full flex items-center justify-between pt-1 border-t border-white/5 text-[10px]">
        <span className="truncate max-w-[80px] text-slate-400">
          {race.cluster_name || '標準'}
        </span>
        <span className={`px-1.5 py-0.2 rounded text-[9px] font-medium ${badgeStyle}`}>
          {isResolved
            ? (hitStatus === 'hit' ? '的中' : (hitStatus === 'miss' ? 'ハズレ' : '見送り'))
            : (isGo ? '投資GO' : '見送り')}
        </span>
      </div>
    </button>
  );
};
