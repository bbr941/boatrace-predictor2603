import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ShieldCheck, Flame, ArrowRight, PiggyBank, CheckCircle2, AlertTriangle } from 'lucide-react';

export const StrategyGuide: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(true);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-lg transition-all duration-200">
      {/* Accordion Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gradient-to-r from-blue-950/60 via-slate-900 to-indigo-950/60 hover:bg-slate-800/80 flex items-center justify-between text-left transition"
      >
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-blue-600/20 text-blue-400 border border-blue-500/30">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm sm:text-base text-slate-100">
                【戦略】プール型・変則2連コロガシ
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                目標合成オッズ 2.5倍
              </span>
            </div>
            <p className="text-xs text-slate-400">
              1戦目 1,000円 ➔ 500円プール ➔ 2戦目 2,000円投下で利益爆発＋破産リスク遮断
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-slate-400 text-xs">
          <span className="hidden sm:inline">{isOpen ? '閉じる' : '詳細ルールを展開'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4 text-blue-400" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 space-y-4">
          {/* 3 Steps Visual Flow */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Step 1 */}
            <div className="bg-slate-900/90 border border-blue-900/40 rounded-lg p-3 relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30">
                  STEP 1
                </span>
                <span className="text-xs font-mono text-slate-300">初期投資 1,000円</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                第1戦（基本ユニット）
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                合成オッズ <strong className="text-amber-300">2.5倍以上</strong> を目指し、Benter累積確率50%超の上位5点へダッチング均等傾斜配分。
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-emerald-400 flex items-center justify-between font-mono">
                <span>的中想定払戻:</span>
                <span className="font-bold">+2,500円〜</span>
              </div>
            </div>

            {/* Step 2 (Pool) */}
            <div className="bg-slate-900/90 border border-amber-900/40 rounded-lg p-3 relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-amber-600/20 text-amber-400 border border-amber-500/30 flex items-center gap-1">
                  <PiggyBank className="w-3 h-3" /> POOL
                </span>
                <span className="text-xs font-mono text-amber-300">防壁確保 500円</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                500円の利益プール
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                1戦目払戻（約2,500円）から <strong className="text-amber-300">500円を即座にプールへ避難</strong>。全額転がさないことで元本割れを阻止。
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-amber-400 flex items-center justify-between font-mono">
                <span>次回コロガシ原資:</span>
                <span className="font-bold">2,000円</span>
              </div>
            </div>

            {/* Step 3 (Final Roll) */}
            <div className="bg-slate-900/90 border border-emerald-900/40 rounded-lg p-3 relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                  <Flame className="w-3 h-3" /> STEP 2
                </span>
                <span className="text-xs font-mono text-emerald-300">転がし投資 2,000円</span>
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">
                第2戦（ブースト勝負）
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                2,000円を次対象レースへ再投入。的中時は <strong className="text-emerald-300">約5,000円払戻</strong>（サイクル純利益 <span className="font-bold text-emerald-400">+3,500円</span>）！
              </p>
              <div className="mt-2.5 pt-2 border-t border-slate-800 text-xs text-emerald-400 flex items-center justify-between font-mono">
                <span>万一ハズレでも:</span>
                <span className="text-slate-300 font-bold">プール500円残存</span>
              </div>
            </div>
          </div>

          {/* Golden Rules Bar */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-slate-400">
            <span className="text-slate-500 font-semibold">【運用鉄則】:</span>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
              <span>目標合成オッズ 2.5倍 (回収期待値確保)</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
              <span>上位5点ダッチング（配分傾斜により全的中パターンで均等利益）</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>2連勝でサイクル完了 ➔ 直ちに次サイクル（1,000円）へリセット</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
