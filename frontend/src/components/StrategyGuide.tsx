import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Target, Flame, CheckCircle2, Crown, Zap, Sparkles } from 'lucide-react';
import { PredictionMode } from '../types';

interface StrategyGuideProps {
  mode?: PredictionMode;
}

export const StrategyGuide: React.FC<StrategyGuideProps> = ({ mode = 'hit_focused' }) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);

  if (mode === 'manshu_sniper') {
    return (
      <div className="bg-slate-900/80 border border-rose-500/30 rounded-xl overflow-hidden shadow-lg transition-all duration-200">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-4 py-3 bg-gradient-to-r from-rose-950/60 via-slate-900 to-orange-950/50 hover:bg-slate-800/80 flex items-center justify-between text-left transition"
        >
          <div className="flex items-center space-x-2.5">
            <div className="p-1.5 rounded-lg bg-rose-600/20 text-rose-400 border border-rose-500/40 shadow-[0_0_10px_rgba(244,63,94,0.3)]">
              <Flame className="w-4 h-4 text-rose-400" />
            </div>
            <div>
              <div className="flex items-center flex-wrap gap-2">
                <span className="font-bold text-sm sm:text-base text-slate-100">
                  【万舟スナイパー】イン過信崩壊 × 外枠穴オーバーレイ特化
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-black italic bg-gradient-to-r from-rose-600 to-orange-500 text-white border border-rose-400/50 shadow-[0_0_8px_rgba(244,63,94,0.6)] animate-pulse">
                  🔥 2026年OOT検証: 回収率 443.5% (Tier 1)
                </span>
              </div>
              <p className="text-xs text-slate-400">
                100〜500倍帯の歪みを100円均等×最大8点（1R最大800円）で射抜く超少額・超高回収スナイプ戦略
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-slate-400 text-xs">
            <span className="hidden sm:inline">{isOpen ? '閉じる' : '詳細を展開'}</span>
            {isOpen ? <ChevronUp className="w-4 h-4 text-rose-400" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        {isOpen && (
          <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 space-y-3 text-xs text-slate-300">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-slate-900/90 border border-rose-900/40 rounded-lg p-3">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-rose-600/20 text-rose-400 border border-rose-500/30">
                  1. イン過信の逆張り検知
                </span>
                <p className="text-slate-400 mt-2 leading-relaxed">
                  市場オッズ勝率に対してAI残差モデルが「1号艇は5%〜6%以上危ない」と判定したイン過信危険レースのみを抽出。
                </p>
              </div>
              <div className="bg-slate-900/90 border border-orange-900/40 rounded-lg p-3">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-orange-600/20 text-orange-400 border border-orange-500/30">
                  2. 外枠(4-6号艇)穴オーバーレイ
                </span>
                <p className="text-slate-400 mt-2 leading-relaxed">
                  4・5・6号艇のいずれかに市場が見落としている+4%〜+5%以上のAI残差エッジが発生しているレースに限定。
                </p>
              </div>
              <div className="bg-gradient-to-b from-rose-950/40 via-slate-900/90 to-orange-950/30 border border-rose-500/40 rounded-lg p-3">
                <span className="text-[11px] font-black px-2 py-0.5 rounded bg-rose-600/30 text-rose-300 border border-rose-400/50">
                  3. 2段階万舟シグナル (100円×最大8点)
                </span>
                <div className="text-[11px] text-slate-300 mt-2 space-y-1 leading-relaxed">
                  <div>
                    <strong className="text-rose-400">🔥 激熱万舟 (Tier 1)</strong>: 1号艇過信≧6% × 外枠穴≧5%
                    <div className="text-amber-300 font-mono text-[10px]">2026年OOT: 万舟的中率10.5% / 回収率443.5% (+106,820円)</div>
                  </div>
                  <div>
                    <strong className="text-amber-400">💣 万舟好機 (Tier 2)</strong>: 1号艇過信≧5% × 外枠穴≧4%
                    <div className="text-amber-300 font-mono text-[10px]">2026年OOT: 万舟的中率5.9% / 回収率243.2% (+96,240円)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

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
              <span className="px-2 py-0.5 rounded-full text-[10px] font-black italic bg-gradient-to-r from-pink-600 via-purple-600 to-amber-500 text-white border border-pink-400/50 shadow-[0_0_8px_rgba(236,72,153,0.6)] animate-pulse">
                🎰 GOGO! & ⭐ CHANCE 2段階ランプ搭載
              </span>
            </div>
            <p className="text-xs text-slate-400">
              全レース買い目を常時表示しつつ、回収率100%超確定の2段階シグナルを自動点灯
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
                <span className="text-xs font-mono text-cyan-300">均等払戻傾斜</span>
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

            {/* Pillar 3: 2-Tier Lamps */}
            <div className="bg-gradient-to-b from-purple-950/40 via-slate-900/90 to-amber-950/30 border border-pink-500/40 rounded-lg p-3 relative shadow-[0_0_15px_rgba(236,72,153,0.15)]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-black px-2 py-0.5 rounded bg-pink-600/30 text-pink-300 border border-pink-400/50 flex items-center gap-1 animate-pulse">
                  <Flame className="w-3 h-3 text-pink-400" /> 2段階ランプ
                </span>
                <span className="text-xs font-mono text-amber-300 font-bold">回収100%超検証済</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                🎰 GOGO! & ⭐ CHANCE
              </h4>
              <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
                <div>
                  <strong className="text-pink-400">🎰 GOGO! (超激アツ)</strong>: P1≧55% × モーター≧40% × 期待値≧1.15
                  <div className="text-amber-300 font-mono text-[10px]">実績: 的中率53.9% / 回収率110.6%</div>
                </div>
                <div>
                  <strong className="text-amber-400">⭐ CHANCE (好機)</strong>: P1≧55% × モーター≧36% × 合成≧2.0倍 × 期待値≧1.05
                  <div className="text-amber-300 font-mono text-[10px]">実績: 的中率39.6% / 回収率101.3% (総合101.9%)</div>
                </div>
              </div>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-pink-300 flex items-center justify-between font-mono">
                <span>シグナル:</span>
                <span className="font-bold text-amber-300">どちらも回収率100%超</span>
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
              <span>全レース買い目表示（クリック不要タイムライン表示対応）</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />
              <span>GOGO! または CHANCE 点灯レースは回収率100%超！</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
