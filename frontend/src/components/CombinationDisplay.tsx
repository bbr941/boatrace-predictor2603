import React from 'react';

// Boat color pill helper (1:白, 2:黒, 3:赤, 4:青, 5:黄, 6:緑)
export const BoatTag: React.FC<{ num: string; size?: 'sm' | 'md' }> = ({ num, size = 'md' }) => {
  const n = parseInt(num, 10);
  const colorMap: Record<number, string> = {
    1: 'bg-slate-100 text-slate-900 border-slate-300 font-bold',
    2: 'bg-slate-900 text-white border-slate-700 font-bold',
    3: 'bg-red-600 text-white border-red-500 font-bold',
    4: 'bg-blue-600 text-white border-blue-500 font-bold',
    5: 'bg-amber-400 text-slate-950 border-amber-300 font-bold',
    6: 'bg-emerald-600 text-white border-emerald-500 font-bold',
  };

  const sizeClass = size === 'sm' ? 'w-4 h-4 text-[10px]' : 'w-5 h-5 text-xs';

  return (
    <span className={`inline-flex items-center justify-center rounded border ${sizeClass} ${colorMap[n] || 'bg-slate-800 text-white'}`}>
      {num}
    </span>
  );
};

export const CombinationDisplay: React.FC<{ combo: string; size?: 'sm' | 'md' }> = ({ combo, size = 'md' }) => {
  const boats = combo.split('-');
  return (
    <div className="flex items-center space-x-1 inline-flex">
      {boats.map((b, idx) => (
        <React.Fragment key={idx}>
          <BoatTag num={b} size={size} />
          {idx < boats.length - 1 && <span className="text-slate-500 text-xs font-mono">-</span>}
        </React.Fragment>
      ))}
    </div>
  );
};
