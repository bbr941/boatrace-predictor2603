import React from 'react';
import { DashboardStats } from '../types';
import { TrendingUp, Award, DollarSign, Percent } from 'lucide-react';

interface SummaryStatsProps {
  stats: DashboardStats;
}

export const SummaryStats: React.FC<SummaryStatsProps> = ({ stats }) => {
  const isProfitPositive = stats.netProfit >= 0;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3.5">
      {/* 1. 的中率 & 成績 */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span>的中成績 / 率</span>
          <Award className="w-3.5 h-3.5 text-blue-400" />
        </div>
        <div>
          <div className="flex items-baseline space-x-1.5">
            <span className="text-xl sm:text-2xl font-bold font-mono text-white">
              {stats.hitRate.toFixed(1)}
            </span>
            <span className="text-xs text-slate-400 font-semibold">%</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center space-x-1.5">
            <span className="text-emerald-400 font-bold">{stats.hitCount} 的中</span>
            <span>/</span>
            <span className="text-rose-400 font-medium">{stats.missCount} 敗</span>
            <span className="text-slate-500 font-mono">({stats.resolvedRaces}確定)</span>
          </div>
        </div>
      </div>

      {/* 2. 確定純収支 */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span>本日確定収支</span>
          <TrendingUp className={`w-3.5 h-3.5 ${isProfitPositive ? 'text-emerald-400' : 'text-rose-400'}`} />
        </div>
        <div>
          <div className="flex items-baseline space-x-1">
            <span className={`text-xl sm:text-2xl font-bold font-mono ${isProfitPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
              {stats.netProfit > 0 ? `+${stats.netProfit.toLocaleString()}` : stats.netProfit.toLocaleString()}
            </span>
            <span className="text-xs text-slate-400">円</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
            <span>回収率:</span>
            <span className={`font-mono font-bold ${stats.recoveryRate >= 100 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {stats.recoveryRate.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* 3. 投資額 & 払戻金 */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span>投資・払戻金</span>
          <DollarSign className="w-3.5 h-3.5 text-amber-400" />
        </div>
        <div>
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-slate-400">払戻:</span>
            <span className="text-sm sm:text-base font-bold font-mono text-emerald-300">
              {stats.totalPayout.toLocaleString()}円
            </span>
          </div>
          <div className="flex items-baseline justify-between mt-1 text-[11px] text-slate-400">
            <span>投資:</span>
            <span className="font-mono text-slate-300">
              {stats.totalBet.toLocaleString()}円
            </span>
          </div>
        </div>
      </div>

      {/* 4. 対象レース進捗 */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span>勝負レース進捗</span>
          <Percent className="w-3.5 h-3.5 text-cyan-400" />
        </div>
        <div>
          <div className="flex items-baseline space-x-1.5">
            <span className="text-xl sm:text-2xl font-bold font-mono text-cyan-400">
              {stats.resolvedRaces}
            </span>
            <span className="text-xs text-slate-400 font-mono">/ {stats.investmentRaces} 参戦</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-cyan-500 h-full transition-all duration-300 rounded-full"
              style={{
                width: `${stats.investmentRaces > 0 ? (stats.resolvedRaces / stats.investmentRaces) * 100 : 0}%`
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
