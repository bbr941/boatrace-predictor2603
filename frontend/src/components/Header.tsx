import React from 'react';
import { RefreshCw, Calendar, Sparkles, Target, Crown, ShieldAlert } from 'lucide-react';
import { PredictionMode } from '../types';
import { isSupabaseConfigured } from '../lib/supabase';

interface HeaderProps {
  mode: PredictionMode;
  onModeChange: (m: PredictionMode) => void;
  selectedDate: string;
  onDateChange: (d: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
  lastUpdated: Date | null;
}

export const Header: React.FC<HeaderProps> = ({
  mode,
  onModeChange,
  selectedDate,
  onDateChange,
  onRefresh,
  isLoading,
  lastUpdated
}) => {
  return (
    <header className="bg-slate-900/90 backdrop-blur border-b border-slate-800 sticky top-0 z-30 px-3 py-2.5 sm:px-6">
      <div className="max-w-7xl mx-auto flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between">
        {/* Top: Logo & Title */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <span className="text-2xl">🚤</span>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-1.5">
                  BOATRACE AI <span className="text-xs px-2 py-0.5 rounded bg-blue-600/30 text-blue-400 font-mono border border-blue-500/30">v3.2 Edge</span>
                </h1>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Supabase直結・Cloudflare Pages爆速フロントエンド
              </p>
            </div>
          </div>

          {/* Quick Refresh on Mobile */}
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="sm:hidden p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition"
            title="手動更新"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-blue-400' : ''}`} />
          </button>
        </div>

        {/* Center: Strategy Mode Selector */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 self-stretch sm:self-auto">
          <button
            onClick={() => onModeChange('hit_focused')}
            className={`flex-1 sm:flex-initial flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              mode === 'hit_focused'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Target className="w-3.5 h-3.5" />
            <span>🎯 的中特化 (動的ダッチング)</span>
          </button>

          <button
            onClick={() => onModeChange('golden_baseline')}
            className={`flex-1 sm:flex-initial flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              mode === 'golden_baseline'
                ? 'bg-amber-600 text-white shadow-lg shadow-amber-600/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Crown className="w-3.5 h-3.5" />
            <span>👑 黄金ベースライン</span>
          </button>
        </div>

        {/* Right: Date Picker & Refresh Info */}
        <div className="flex items-center justify-between sm:justify-end space-x-3">
          <div className="flex items-center space-x-1 bg-slate-950 px-2 py-1 rounded-lg border border-slate-800">
            <Calendar className="w-3.5 h-3.5 text-slate-400" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => onDateChange(e.target.value)}
              className="bg-transparent text-xs text-slate-200 font-mono focus:outline-none cursor-pointer"
            />
          </div>

          <div className="hidden sm:flex items-center space-x-2 text-xs text-slate-400">
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition font-medium border border-slate-700"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-blue-400' : ''}`} />
              <span>更新</span>
            </button>
            {lastUpdated && (
              <span className="text-[11px] text-slate-500 font-mono">
                {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Supabase Status Alert (if mock mode) */}
      {!isSupabaseConfigured && (
        <div className="mt-2 max-w-7xl mx-auto bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-1.5 text-xs text-amber-300 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-4 h-4 flex-shrink-0 text-amber-400" />
            <span>
              <strong>モックプレビュー中:</strong> <code>.env</code> に <code>VITE_SUPABASE_ANON_KEY</code> を設定すると本番Supabaseとリアルタイム直接接続されます。
            </span>
          </div>
        </div>
      )}
    </header>
  );
};
