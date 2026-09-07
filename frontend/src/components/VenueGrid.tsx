import React, { useMemo } from 'react';
import { HitFocusedPrediction, RacePrediction } from '../types';
import { RaceButton } from './RaceButton';
import { MapPin, Trophy } from 'lucide-react';

interface VenueGridProps {
  races: (HitFocusedPrediction | RacePrediction)[];
  selectedRaceId?: string | null;
  onSelectRace: (race: HitFocusedPrediction | RacePrediction) => void;
}

export const VenueGrid: React.FC<VenueGridProps> = ({ races, selectedRaceId, onSelectRace }) => {
  // Group races by venue_code & venue_name
  const groupedVenues = useMemo(() => {
    const map = new Map<number, { venueCode: number; venueName: string; races: (HitFocusedPrediction | RacePrediction)[] }>();

    races.forEach((race) => {
      const vCode = race.venue_code || 0;
      if (!map.has(vCode)) {
        map.set(vCode, {
          venueCode: vCode,
          venueName: race.venue_name || `会場 ${vCode}`,
          races: [],
        });
      }
      map.get(vCode)!.races.push(race);
    });

    // Sort venues by venue code
    const list = Array.from(map.values()).sort((a, b) => a.venueCode - b.venueCode);
    // Sort races within each venue by race_no
    list.forEach(v => v.races.sort((a, b) => a.race_no - b.race_no));
    return list;
  }, [races]);

  if (races.length === 0) {
    return (
      <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-12 text-center text-slate-400">
        <Trophy className="w-10 h-10 mx-auto text-slate-600 mb-3" />
        <p className="text-base font-medium">該当するレースデータがありません</p>
        <p className="text-xs text-slate-500 mt-1">日付を選択するか、更新ボタンを押してください</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {groupedVenues.map(({ venueCode, venueName, races: venueRaces }) => {
        // Venue total profit and hits
        const hits = venueRaces.filter(r => r.hit_status === 'hit').length;
        const venueProfit = venueRaces.reduce((sum, r) => sum + (r.profit || 0), 0);

        return (
          <div
            key={venueCode}
            className="bg-slate-900/60 border border-slate-800/90 rounded-2xl p-3.5 sm:p-5 shadow-xl backdrop-blur"
          >
            {/* Venue Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3.5">
              <div className="flex items-center space-x-2.5">
                <div className="p-1.5 rounded-lg bg-blue-600/10 text-blue-400 border border-blue-500/20">
                  <MapPin className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
                    <span>{venueName}</span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                      #{venueCode.toString().padStart(2, '0')}
                    </span>
                  </h3>
                </div>
              </div>

              {/* Venue Sub-stats */}
              <div className="flex items-center space-x-3 text-xs">
                <span className="text-slate-400">
                  的中: <strong className="text-emerald-400 font-mono">{hits}</strong> R
                </span>
                <span className="text-slate-700">|</span>
                <span className="text-slate-400">
                  収支:{' '}
                  <strong
                    className={`font-mono font-bold ${
                      venueProfit > 0
                        ? 'text-emerald-400'
                        : venueProfit < 0
                        ? 'text-rose-400'
                        : 'text-slate-300'
                    }`}
                  >
                    {venueProfit > 0 ? `+${venueProfit.toLocaleString()}` : venueProfit.toLocaleString()}円
                  </strong>
                </span>
              </div>
            </div>

            {/* Responsive Grid: 1行4列（PC）/ 1行2〜3列（スマホ） */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
              {venueRaces.map((race) => (
                <RaceButton
                  key={race.race_id}
                  race={race}
                  isSelected={selectedRaceId === race.race_id}
                  onClick={() => onSelectRace(race)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};
