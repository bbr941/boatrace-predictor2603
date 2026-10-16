import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Target, Flame, CheckCircle2, Crown, Zap, Sparkles } from 'lucide-react';
import { PredictionMode } from '../types';

interface StrategyGuideProps {
  mode?: PredictionMode;
}

export const StrategyGuide: React.FC<StrategyGuideProps> = ({ mode = 'hit_focused' }) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);

  if (mode === 'golden_baseline') {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-lg transition-all duration-200">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-4 py-3 bg-gradient-to-r from-amber-950/40 via-slate-900 to-amber-950/40 hover:bg-slate-800/80 flex items-center justify-between text-left transition"
        >
          <div className="flex items-center space-x-2.5">
            <div className="p-1.5 rounded-lg bg-amber-600/20 text-amber-400 border border-amber-500/30">
              <Crown className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-sm sm:text-base text-slate-100">
                  【戦略】黄金ベースライン（厳選期待値投資）
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  SLSQPポートフォリオ最適化
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Gatekeeper水面判定 ＆ ケリー基準に基づく長期回収率特化の厳選投資モデル
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-slate-400 text-xs">
            <span className="hidden sm:inline">{isOpen ? '閉じる' : '詳細を展開'}</span>
            {isOpen ? <ChevronUp className="w-4 h-4 text-amber-400" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        {isOpen && (
          <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 space-y-3 text-xs text-slate-300">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30">
                  1. Gatekeeper
                </span>
                <p className="text-slate-400 mt-2 leading-relaxed">
                  イン勝率と水面クラスターから荒れる危険レースを徹底排除。
                </p>
              </div>
              <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-amber-600/20 text-amber-400 border border-amber-500/30">
                  2. SLSQP 最適化
                </span>
                <p className="text-slate-400 mt-2 leading-relaxed">
                  相関行列と同時ケリー基準により、期待値（EV）プラスの買い目のみ厳選配分。
                </p>
              </div>
              <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-600/20 text-emerald-400 border border-emerald-500/30">
                  3. 長期プラス収支
                </span>
                <p className="text-slate-400 mt-2 leading-relaxed">
                  無駄打ちを徹底回避し、勝負レースのみに資金を集中投下。
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // mode === 'hit_focused' (完全的中率特化 & GOGOランプ)
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-lg transition-all duration-200">
      {/* Accordion Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gradient-to-r from-blue-950/70 via-slate-900 to-pink-950/40 hover:bg-slate-800/80 flex items-center justify-between text-left transition"
      >
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-pink-600/20 text-pink-400 border border-pink-500/40 shadow-[0_0_10px_rgba(236,72,153,0.3)]">
            <Zap className="w-4 h-4 text-pink-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm sm:text-base text-slate-100 flex items-center gap-1.5">
                【完全的中率特化】AIダイレクトダッチング
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-black italic bg-gradient-to-r from-pink-600 to-purple-600 text-white border border-pink-400/50 shadow-[0_0_8px_rgba(236,72,153,0.6)] animate-pulse">
                🎰 GOGO! ランプ搭載
              </span>
            </div>
            <p className="text-xs text-slate-400">
              コロガシ・合成オッズ制約を完全撤廃 ➔ AI累積確率重視で的中率を極限追求
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-slate-400 text-xs">
          <span className="hidden sm:inline">{isOpen ? '閉じる' : '特化ルールを展開'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4 text-pink-400" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 space-y-4">
          {/* 3 Pillars Visual Flow */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Pillar 1 */}
            <div className="bg-slate-900/90 border border-blue-900/40 rounded-lg p-3 relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30 flex items-center gap-1">
                  <Target className="w-3 h-3" /> 的中率特化
                </span>
                <span className="text-xs font-mono text-slate-300">オッズ縛り撤廃</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                累積勝率上位へ集中
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                目標合成オッズなどの縛りを完全排除。Benterモデルが推計した累積勝率50%超の上位買い目を確率順に抽出し、的中機会を最優先。
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-blue-400 flex items-center justify-between font-mono">
                <span>狙い:</span>
                <span className="font-bold">徹底的な高的中率</span>
              </div>
            </div>

            {/* Pillar 2 */}
            <div className="bg-slate-900/90 border border-cyan-900/40 rounded-lg p-3 relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center gap-1">
                  <Sparkles className="w-3 h-3" /> 動的ダッチング
                </span>
                <span className="text-xs font-mono text-cyan-300">トリガミ回避傾斜</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                オッズ逆数比の均等払戻
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                各オッズの逆数比で資金を傾斜配分。どの選出買い目が的中しても、同一の払戻金を得られるよう自動最適配分。
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-cyan-400 flex items-center justify-between font-mono">
                <span>配分方式:</span>
                <span className="font-bold">均等傾斜ダッチング</span>
              </div>
            </div>

            {/* Pillar 3: GOGO Lamp */}
            <div className="bg-gradient-to-b from-pink-950/40 to-slate-900/90 border border-pink-500/40 rounded-lg p-3 relative shadow-[0_0_15px_rgba(236,72,153,0.15)]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-black px-2 py-0.5 rounded bg-pink-600/30 text-pink-300 border border-pink-400/50 flex items-center gap-1 animate-pulse">
                  <Flame className="w-3 h-3 text-pink-400" /> GOGO! ランプ
                </span>
                <span className="text-xs font-mono text-amber-300 font-bold">回収率110.6%</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1 flex items-center gap-1.5">
                <span>🎰 黄金緩和条件達成で点灯</span>
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                黄金ベースラインの条件を緩和化（本命P1≧55% × 1号艇モーター≧40% × 合成期待値≧1.15）。全レース買い目を表示しつつ、条件クリア時に <strong className="text-pink-400 font-bold">GOGO! ランプ</strong> が点灯！
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-pink-300 flex items-center justify-between font-mono">
                <span>検証実績:</span>
                <span className="font-bold text-amber-300">的中率53.9% / 回収率110.6%</span>
              </div>
            </div>
          </div>

          {/* Guidelines Bar */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-slate-400">
            <span className="text-slate-500 font-semibold">【モード特徴】:</span>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-pink-400" />
              <span>コロガシ戦略なし（単発レース完結・安定運用）</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
              <span>目標合成オッズ撤廃（AI勝率優先の超高的中率）</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-pink-400" />
              <span>GOGO! ランプ点灯レースは回収率100%超のチャンス！</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
