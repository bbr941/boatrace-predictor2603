import React from 'react';
import { HitFocusedPrediction, RacePrediction, isGogoLampActive, isChanceLampActive } from '../types';
import { Clock, Flame, Star } from 'lucide-react';

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
  const minProfit = (race as HitFocusedPrediction).min_profit || 0;
  const isGogo = isGogoLampActive(race);
  const isChance = isChanceLampActive(race);

  // Visual Styles based on status
  let cardStyle = "bg-slate-900/40 border-slate-800/80 text-slate-500 hover:border-slate-700";
  let contentNode: React.ReactNode = null;

  if (isResolved) {
    if (hitStatus === 'hit') {
      const isTrigami = profit < 0;
      cardStyle = isTrigami
        ? "bg-gradient-to-b from-amber-950/60 to-slate-900 border-amber-500/50 text-amber-200 shadow-md hover:border-amber-400 ring-1 ring-amber-500/30"
        : (isGogo
          ? "bg-gradient-to-b from-emerald-950/90 via-slate-900 to-pink-950/40 border-emerald-400/80 text-emerald-200 shadow-lg shadow-emerald-950/50 hover:border-emerald-300 ring-1 ring-pink-500/40"
          : isChance
          ? "bg-gradient-to-b from-emerald-950/90 via-slate-900 to-amber-950/40 border-emerald-400/80 text-emerald-200 shadow-lg shadow-emerald-950/50 hover:border-emerald-300 ring-1 ring-amber-400/40"
          : "bg-gradient-to-b from-emerald-950/80 to-emerald-900/40 border-emerald-500/60 text-emerald-200 shadow-lg shadow-emerald-950/40 hover:border-emerald-400 ring-1 ring-emerald-500/30");
      
      contentNode = (
        <div className="flex flex-col items-center justify-center py-1">
          <div className={`flex items-center space-x-1 text-xs font-bold ${isTrigami ? 'text-amber-300' : 'text-emerald-300'}`}>
            <span>{isTrigami ? '💮 的中(ガミ)' : '💮 的中!'}</span>
            <span className="font-mono">
              {profit > 0 ? `+${profit.toLocaleString()}` : profit < 0 ? `-${Math.abs(profit).toLocaleString()}` : '±0'}円
            </span>
          </div>
          {race.actual_result && (
            <span className={`text-[10px] ${isTrigami ? 'text-amber-400/80' : 'text-emerald-400/80'} font-mono tracking-wider`}>
              結果: {race.actual_result}
            </span>
          )}
        </div>
      );
    } else if (hitStatus === 'miss') {
      // 💀 ハズレ (落ち着いたダークトーン)
      cardStyle = "bg-slate-950/80 border-slate-800/90 text-slate-400 hover:border-slate-700 hover:bg-slate-900/60";
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
      if (isGogo) {
        // 🎰 GOGOランプ点灯 (Tier 1: 回収率110%超)
        cardStyle = "bg-gradient-to-b from-purple-950/80 via-slate-900 to-pink-950/50 border-pink-500/70 text-pink-100 hover:border-pink-400 shadow-xl shadow-pink-950/40 hover:scale-[1.02] ring-1 ring-pink-500/50";
        contentNode = (
          <div className="flex flex-col items-center justify-center py-1">
            <div className="flex items-center space-x-1 text-[11px] sm:text-xs font-semibold text-pink-200">
              <span>🎯 {betsCount}点 ({totalBet.toLocaleString()}円)</span>
            </div>
            <div className="text-[11px] text-amber-300 font-mono font-black drop-shadow-[0_0_4px_rgba(245,158,11,0.8)]">
              ★期待回収110%超 (激熱)
            </div>
          </div>
        );
      } else if (isChance) {
        // ⭐ CHANCEランプ点灯 (Tier 2: 回収率101%超)
        cardStyle = "bg-gradient-to-b from-amber-950/70 via-slate-900 to-yellow-950/40 border-amber-400/70 text-amber-100 hover:border-amber-300 shadow-lg shadow-amber-950/30 hover:scale-[1.02] ring-1 ring-amber-400/50";
        contentNode = (
          <div className="flex flex-col items-center justify-center py-1">
            <div className="flex items-center space-x-1 text-[11px] sm:text-xs font-semibold text-amber-200">
              <span>🎯 {betsCount}点 ({totalBet.toLocaleString()}円)</span>
            </div>
            <div className="text-[11px] text-amber-300 font-mono font-black drop-shadow-[0_0_4px_rgba(245,158,11,0.8)]">
              ★期待回収101%超 (好機)
            </div>
          </div>
        );
      } else {
        // 🎯 レース前 (通常投資GO)
        cardStyle = "bg-gradient-to-b from-blue-950/70 to-slate-900 border-blue-500/50 text-blue-100 hover:border-blue-400 shadow-md shadow-blue-950/30 hover:scale-[1.01]";
        contentNode = (
          <div className="flex flex-col items-center justify-center py-1">
            <div className="flex items-center space-x-1 text-[11px] sm:text-xs font-semibold text-blue-200">
              <span>🎯 {betsCount}点 ({totalBet.toLocaleString()}円)</span>
            </div>
            <div className="text-[11px] text-cyan-300 font-mono font-bold">
              AI的中特化配分
            </div>
          </div>
        );
      }
    } else {
      // レース前・スキップ / 判定外
      cardStyle = "bg-slate-950/40 border-slate-900 text-slate-600 hover:border-slate-800 hover:text-slate-500";
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
      className={`relative w-full rounded-xl border p-2.5 flex flex-col justify-between transition duration-150 text-left min-h-[82px] ${cardStyle} ${
        isSelected ? 'ring-2 ring-cyan-400 ring-offset-2 ring-offset-slate-950' : ''
      }`}
    >
      {/* Card Header: Race No & Deadline & Lamp Badge */}
      <div className="flex items-center justify-between w-full border-b border-white/5 pb-1">
        <div className="flex items-center space-x-1.5">
          <span className="font-bold font-mono text-xs text-slate-200">
            {race.race_no}R
          </span>

          {/* Lamp in Header */}
          {isGogo ? (
            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-gradient-to-r from-purple-900 to-pink-900 border border-pink-400 text-pink-200 shadow-[0_0_8px_rgba(236,72,153,0.8)] animate-pulse font-black italic text-[9px] tracking-tight">
              GOGO!
            </span>
          ) : isChance ? (
            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-gradient-to-r from-amber-900 to-orange-900 border border-amber-300 text-amber-200 shadow-[0_0_8px_rgba(245,158,11,0.7)] animate-pulse font-black italic text-[9px] tracking-tight">
              CHANCE
            </span>
          ) : null}
        </div>

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
        <div>
          {isResolved ? (
            hitStatus === 'hit' ? (
              <span className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
                profit < 0
                  ? 'text-amber-300 bg-amber-900/60 border border-amber-500/40'
                  : 'text-emerald-300 bg-emerald-900/60 border border-emerald-500/40'
              }`}>
                {profit < 0 ? '的中(ガミ)' : '的中'}
              </span>
            ) : hitStatus === 'miss' ? (
              <span className="px-1.5 py-0.2 rounded text-[9px] font-medium text-slate-500 bg-slate-900 border border-slate-800">
                ハズレ
              </span>
            ) : (
              <span className="px-1.5 py-0.2 rounded text-[9px] font-medium text-slate-600 bg-slate-950 border border-slate-900">
                見送り
              </span>
            )
          ) : isGo ? (
            isGogo ? (
              <span className="px-1.5 py-0.2 rounded text-[9px] font-black italic bg-gradient-to-r from-pink-600 to-purple-600 text-white border border-pink-400 shadow-[0_0_8px_rgba(236,72,153,0.8)] animate-pulse">
                🎰 GOGO!
              </span>
            ) : (
              <span className="px-1.5 py-0.2 rounded text-[9px] font-medium text-blue-300 bg-blue-900/60 border border-blue-500/40">
                投資GO
              </span>
            )
          ) : (
            <span className="px-1.5 py-0.2 rounded text-[9px] font-medium text-slate-600 bg-slate-950 border border-slate-900">
              見送り
            </span>
          )}
        </div>
      </div>
    </button>
  );
};
