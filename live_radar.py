"""
live_radar.py
🚤 BOATRACE AI 統合ライブモニター (CustomTkinter 3カラム・並列オッズ版)
- current_radar.json の更新を1秒ごとに監視
- 【左カラム: AI Radar】Gatekeeper P1・Sniper判定・水面気象・各モード投資サマリー
- 【中央カラム: 出走表＆直前情報】1〜6号艇のモーター率・展示タイム・チルト・ST
- 【右カラム: オッズ＆期待値】
    - 上段: 🎯 黄金ベースライン (Sniper / EV重視)
    - 下段: 🛡️ 的中特化 (Dutching / トリガミ回避)
"""

import os
import sys
import json
import time
import datetime
import tkinter as tk
import customtkinter as ctk

# テーマとアピアランス設定
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RADAR_JSON_PATH = os.path.join(CURRENT_DIR, "current_radar.json")

# 艇番ごとの配色 (背景色, 文字色)
BOAT_COLORS = {
    1: ("#FFFFFF", "#000000"),  # 1号艇: 白 (黒文字)
    2: ("#212121", "#FFFFFF"),  # 2号艇: 黒 (白文字)
    3: ("#E53935", "#FFFFFF"),  # 3号艇: 赤 (白文字)
    4: ("#1E88E5", "#FFFFFF"),  # 4号艇: 青 (白文字)
    5: ("#FDD835", "#000000"),  # 5号艇: 黄 (黒文字)
    6: ("#43A047", "#FFFFFF"),  # 6号艇: 緑 (白文字)
}


class IntegratedLiveRadar(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚤 BOATRACE AI Integrated Live Radar (Dual Strategy)")
        self.geometry("1100x680")
        self.minsize(1020, 600)

        self.last_mtime = 0
        self.boat_widgets = {}

        # UI レイアウト構築
        self.build_ui()

        # 初回データ読み込み & ポーリング開始
        self.poll_radar_file()

    def build_ui(self):
        # =====================================================================
        # 1. 最上部 ヘッダーバー
        # =====================================================================
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#18181E")
        self.header_frame.pack(fill="x", padx=12, pady=(10, 6))

        self.lbl_app_title = ctk.CTkLabel(
            self.header_frame,
            text="🚤 BOATRACE AI Live Radar",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00E5FF"
        )
        self.lbl_app_title.pack(side="left", padx=16, pady=8)

        self.lbl_race_name = ctk.CTkLabel(
            self.header_frame,
            text="待機中... (レース未取得)",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFFFFF"
        )
        self.lbl_race_name.pack(side="left", padx=20, pady=8)

        self.lbl_deadline = ctk.CTkLabel(
            self.header_frame,
            text="締切: --:--",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD600"
        )
        self.lbl_deadline.pack(side="right", padx=16, pady=8)

        # =====================================================================
        # 2. メイン 3カラム コンテナ
        # =====================================================================
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=12, pady=4)

        # ---------------------------------------------------------------------
        # 【左カラム: AI Radar】 (幅: 約310px)
        # ---------------------------------------------------------------------
        self.col_left = ctk.CTkFrame(self.main_container, width=310, corner_radius=10, fg_color="#1E1E26")
        self.col_left.pack(side="left", fill="both", padx=(0, 5), pady=0)
        self.col_left.pack_propagate(False)

        lbl_col1_title = ctk.CTkLabel(
            self.col_left, text="🛡️ AI Radar (評価 & 判定)",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#00E5FF"
        )
        lbl_col1_title.pack(anchor="w", padx=12, pady=(8, 2))

        # Gatekeeper P1 信頼度ボックス
        self.box_gk = ctk.CTkFrame(self.col_left, corner_radius=8, fg_color="#16161D")
        self.box_gk.pack(fill="x", padx=10, pady=3)

        self.lbl_p1_title = ctk.CTkLabel(self.box_gk, text="Gatekeeper P1 (1着勝率)", font=ctk.CTkFont(size=11), text_color="#888888")
        self.lbl_p1_title.pack(anchor="w", padx=10, pady=(5, 0))

        self.lbl_p1_val = ctk.CTkLabel(self.box_gk, text="--.-%", font=ctk.CTkFont(size=24, weight="bold"), text_color="#AAAAAA")
        self.lbl_p1_val.pack(anchor="w", padx=10, pady=(0, 2))

        self.progress_p1 = ctk.CTkProgressBar(self.box_gk, height=10, corner_radius=5)
        self.progress_p1.pack(fill="x", padx=10, pady=(1, 3))
        self.progress_p1.set(0.0)
        self.progress_p1.configure(progress_color="#666666")

        self.lbl_gk_threshold = ctk.CTkLabel(
            self.box_gk, text="基準閾値: 74.38% (黄金ベースライン)",
            font=ctk.CTkFont(size=10), text_color="#777777"
        )
        self.lbl_gk_threshold.pack(anchor="e", padx=10, pady=(0, 4))

        # 最終判定ステータスバッジ
        self.box_verdict = ctk.CTkFrame(self.col_left, corner_radius=8, fg_color="#16161D")
        self.box_verdict.pack(fill="x", padx=10, pady=3)

        self.lbl_verdict_badge = ctk.CTkLabel(
            self.box_verdict, text="⏳ 待機中",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFB300", fg_color="#332B00", corner_radius=6, padx=10, pady=3
        )
        self.lbl_verdict_badge.pack(padx=10, pady=(6, 3))

        self.lbl_verdict_msg = ctk.CTkLabel(
            self.box_verdict, text="推論実行を待機しています...",
            font=ctk.CTkFont(size=11), text_color="#BBBBBB", wraplength=270
        )
        self.lbl_verdict_msg.pack(padx=10, pady=(0, 6))

        # 水面・気象・環境カード
        self.box_env = ctk.CTkFrame(self.col_left, corner_radius=8, fg_color="#16161D")
        self.box_env.pack(fill="x", padx=10, pady=3)

        self.lbl_cluster_name = ctk.CTkLabel(self.box_env, text="🏟️ クラスタ: --", font=ctk.CTkFont(size=11, weight="bold"), text_color="#E0E0E0")
        self.lbl_cluster_name.pack(anchor="w", padx=10, pady=(5, 1))

        self.lbl_wave_stat = ctk.CTkLabel(self.box_env, text="🌊 波高: -- cm", font=ctk.CTkFont(size=11), text_color="#AAAAAA")
        self.lbl_wave_stat.pack(anchor="w", padx=10, pady=1)

        self.lbl_wind_stat = ctk.CTkLabel(self.box_env, text="🍃 風況: --", font=ctk.CTkFont(size=11), text_color="#AAAAAA")
        self.lbl_wind_stat.pack(anchor="w", padx=10, pady=(1, 5))

        # 投資サマリー（黄金 vs 的中特化）
        self.box_summary = ctk.CTkFrame(self.col_left, corner_radius=8, fg_color="#16161D")
        self.box_summary.pack(fill="x", padx=10, pady=3)

        self.lbl_golden_summary = ctk.CTkLabel(
            self.box_summary, text="🎯 黄金投資: 0 円 (0点)",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E676"
        )
        self.lbl_golden_summary.pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_hit_summary = ctk.CTkLabel(
            self.box_summary, text="🛡️ 的中投資: 0 円 (0点)",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E5FF"
        )
        self.lbl_hit_summary.pack(anchor="w", padx=10, pady=(0, 6))

        # ---------------------------------------------------------------------
        # 【中央カラム: 出走表＆直前情報】 (幅: 約440px)
        # ---------------------------------------------------------------------
        self.col_center = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="#1E1E26")
        self.col_center.pack(side="left", fill="both", expand=True, padx=3, pady=0)

        lbl_col2_title = ctk.CTkLabel(
            self.col_center, text="📋 出走表 & 直前展示タイム",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#00E5FF"
        )
        lbl_col2_title.pack(anchor="w", padx=12, pady=(8, 2))

        # 表ヘッダー
        self.table_header = ctk.CTkFrame(self.col_center, height=24, corner_radius=4, fg_color="#141419")
        self.table_header.pack(fill="x", padx=8, pady=(2, 3))

        headers = [("艇/選手", 90), ("モータ", 55), ("ボート", 50), ("展示T", 50), ("チルト", 45), ("ST", 45), ("P1勝率", 60)]
        for h_text, h_w in headers:
            lbl = ctk.CTkLabel(self.table_header, text=h_text, width=h_w, font=ctk.CTkFont(size=11, weight="bold"), text_color="#AAAAAA")
            lbl.pack(side="left", padx=2)

        # 1〜6号艇の行コンテナ
        self.boats_container = ctk.CTkFrame(self.col_center, fg_color="transparent")
        self.boats_container.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        for bn in range(1, 7):
            bg_col, fg_col = BOAT_COLORS[bn]
            row_frame = ctk.CTkFrame(self.boats_container, height=44, corner_radius=6, fg_color="#16161D")
            row_frame.pack(fill="x", pady=2)
            row_frame.pack_propagate(False)

            # 艇番バッジ
            badge = ctk.CTkLabel(
                row_frame, text=f"{bn}", width=24, height=24,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=bg_col, text_color=fg_col, corner_radius=4
            )
            badge.pack(side="left", padx=(6, 4), pady=6)

            # 選手名
            lbl_racer = ctk.CTkLabel(row_frame, text=f"{bn}号艇", width=62, anchor="w", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF")
            lbl_racer.pack(side="left", padx=2)

            # モーター率
            lbl_motor = ctk.CTkLabel(row_frame, text="--.-%", width=55, font=ctk.CTkFont(size=11), text_color="#E0E0E0")
            lbl_motor.pack(side="left", padx=2)

            # ボート率
            lbl_boat = ctk.CTkLabel(row_frame, text="--.-%", width=50, font=ctk.CTkFont(size=11), text_color="#999999")
            lbl_boat.pack(side="left", padx=2)

            # 展示タイム
            lbl_ex = ctk.CTkLabel(row_frame, text="-.--", width=50, font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD600")
            lbl_ex.pack(side="left", padx=2)

            # チルト
            lbl_tilt = ctk.CTkLabel(row_frame, text="-0.5", width=45, font=ctk.CTkFont(size=11), text_color="#999999")
            lbl_tilt.pack(side="left", padx=2)

            # ST
            lbl_st = ctk.CTkLabel(row_frame, text=".20", width=45, font=ctk.CTkFont(size=11), text_color="#E0E0E0")
            lbl_st.pack(side="left", padx=2)

            # P1勝率
            lbl_p1 = ctk.CTkLabel(row_frame, text="--.-%", width=60, font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E5FF")
            lbl_p1.pack(side="left", padx=2)

            self.boat_widgets[bn] = {
                'frame': row_frame,
                'racer': lbl_racer,
                'motor': lbl_motor,
                'boat': lbl_boat,
                'ex': lbl_ex,
                'tilt': lbl_tilt,
                'st': lbl_st,
                'p1': lbl_p1
            }

        # ---------------------------------------------------------------------
        # 【右カラム: オッズ＆期待値 (上下2段分割)】 (幅: 約320px)
        # ---------------------------------------------------------------------
        self.col_right = ctk.CTkFrame(self.main_container, width=320, corner_radius=10, fg_color="#1E1E26")
        self.col_right.pack(side="left", fill="both", padx=(3, 0), pady=0)
        self.col_right.pack_propagate(False)

        # 上段: 🎯 黄金ベースライン (Sniper / SLSQP)
        self.frame_golden_section = ctk.CTkFrame(self.col_right, fg_color="transparent")
        self.frame_golden_section.pack(fill="both", expand=True, padx=6, pady=(6, 3))

        self.lbl_golden_title = ctk.CTkLabel(
            self.frame_golden_section, text="🎯 黄金ベースライン (Sniper / EV)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E676"
        )
        self.lbl_golden_title.pack(anchor="w", padx=6, pady=(2, 2))

        self.scroll_golden = ctk.CTkScrollableFrame(self.frame_golden_section, fg_color="#16161D", corner_radius=6)
        self.scroll_golden.pack(fill="both", expand=True, padx=4, pady=(0, 2))

        # 下段: 🛡️ 的中特化 (Dutching / 累積50%)
        self.frame_hit_section = ctk.CTkFrame(self.col_right, fg_color="transparent")
        self.frame_hit_section.pack(fill="both", expand=True, padx=6, pady=(3, 6))

        self.lbl_hit_title = ctk.CTkLabel(
            self.frame_hit_section, text="🛡️ 的中特化 (Dutching / 累積50%)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF"
        )
        self.lbl_hit_title.pack(anchor="w", padx=6, pady=(2, 2))

        self.scroll_hit = ctk.CTkScrollableFrame(self.frame_hit_section, fg_color="#16161D", corner_radius=6)
        self.scroll_hit.pack(fill="both", expand=True, padx=4, pady=(0, 2))

        # =====================================================================
        # 3. 最下部 フッターバー
        # =====================================================================
        self.footer_frame = ctk.CTkFrame(self, height=26, fg_color="transparent")
        self.footer_frame.pack(fill="x", padx=16, pady=(1, 5))

        self.lbl_indicator = ctk.CTkLabel(
            self.footer_frame,
            text="🟢 リアルタイム監視中 (1秒ポーリング)",
            font=ctk.CTkFont(size=11),
            text_color="#00E676"
        )
        self.lbl_indicator.pack(side="left")

        self.lbl_updated = ctk.CTkLabel(
            self.footer_frame,
            text="最終更新: --:--:--",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.lbl_updated.pack(side="right")

    def poll_radar_file(self):
        """1秒ごとに current_radar.json の更新日時をチェック"""
        if os.path.exists(RADAR_JSON_PATH):
            try:
                mtime = os.path.getmtime(RADAR_JSON_PATH)
                if mtime > self.last_mtime:
                    self.last_mtime = mtime
                    with open(RADAR_JSON_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.update_ui(data)
            except Exception as e:
                pass
        else:
            self.lbl_indicator.configure(text="🟡 current_radar.json 待機中...", text_color="#FFB300")

        # 1000ms後に再呼び出し
        self.after(1000, self.poll_radar_file)

    def update_ui(self, data):
        """受信した統合データに基づき3カラム全体を描画"""
        race = data.get("race", {})
        ai = data.get("ai_eval", {})
        boats = data.get("boats", [])
        odds_golden = data.get("odds_golden") or data.get("odds", [])
        odds_hit = data.get("odds_hit_focused", [])
        updated_at = data.get("updated_at", "")

        # 1. ヘッダー更新
        race_name = race.get("race_name") or data.get("race_name", "不明なレース")
        deadline = race.get("deadline") or data.get("deadline", "--:--")
        top_boat = ai.get("top_boat") or data.get("top_boat", 1)

        self.lbl_race_name.configure(text=f"🏟️ {race_name}")
        self.lbl_deadline.configure(text=f"締切: {deadline} (本命: {top_boat}号艇)")

        # 2. 【左カラム: AI Radar】
        p1 = ai.get("p1_score") if "p1_score" in ai else data.get("p1_score", 0.0)
        self.lbl_p1_val.configure(text=f"{p1:.1%}")
        self.progress_p1.set(min(max(p1, 0.0), 1.0))

        if p1 >= 0.7438:
            self.progress_p1.configure(progress_color="#00E676")
            self.lbl_p1_val.configure(text_color="#00E676")
        elif p1 >= 0.60:
            self.progress_p1.configure(progress_color="#FFD600")
            self.lbl_p1_val.configure(text_color="#FFD600")
        else:
            self.progress_p1.configure(progress_color="#757575")
            self.lbl_p1_val.configure(text_color="#AAAAAA")

        status = ai.get("status") or data.get("status", "")
        status_msg = ai.get("status_message") or data.get("status_message", "")

        if status == "investment_go":
            self.lbl_verdict_badge.configure(text="🚀 投資GOサイン点灯", text_color="#00E676", fg_color="#00391C")
        elif status == "sniper_skipped":
            self.lbl_verdict_badge.configure(text="🎯 Sniper見送り", text_color="#FFB300", fg_color="#332B00")
        elif status == "gatekeeper_skipped":
            self.lbl_verdict_badge.configure(text="☕ Gatekeeper未達", text_color="#B0BEC5", fg_color="#263238")
        elif status == "skipped_cluster1":
            self.lbl_verdict_badge.configure(text="🛑 難水面除外", text_color="#FF5252", fg_color="#3B1111")
        elif status == "no_value_bets":
            self.lbl_verdict_badge.configure(text="🔍 EV未達見送り", text_color="#40C4FF", fg_color="#002B3D")
        else:
            self.lbl_verdict_badge.configure(text=f"ℹ️ {status}", text_color="#E0E0E0", fg_color="#333333")

        self.lbl_verdict_msg.configure(text=status_msg)

        c_name = race.get("cluster_name", "標準水面")
        c_id = race.get("cluster_id", 2)
        self.lbl_cluster_name.configure(text=f"🏟️ 会場: {c_name} (Cluster {c_id})")

        wave = ai.get("wave_height") if "wave_height" in ai else data.get("wave_height", 0.0)
        self.lbl_wave_stat.configure(text=f"🌊 波高: {wave:.1f} cm")

        # 投資サマリー
        tot_golden = ai.get("total_bet_golden") if "total_bet_golden" in ai else data.get("total_bet", 0)
        cnt_golden = ai.get("bets_count_golden") if "bets_count_golden" in ai else data.get("bets_count", 0)
        self.lbl_golden_summary.configure(text=f"🎯 黄金投資: {tot_golden:,} 円 ({cnt_golden}点)")

        tot_hit = ai.get("total_bet_hit", 0)
        cnt_hit = ai.get("bets_count_hit", 0)
        self.lbl_hit_summary.configure(text=f"🛡️ 的中投資: {tot_hit:,} 円 ({cnt_hit}点)")

        # 3. 【中央カラム: 出走表＆直前情報】
        if boats:
            for b in boats:
                bn = b.get("boat_number", 1)
                if bn in self.boat_widgets:
                    w = self.boat_widgets[bn]
                    r_name = b.get("racer_name", f"{bn}号艇")
                    r_id = b.get("racer_id", 0)
                    if r_id > 0 and r_name == f"{bn}号艇":
                        r_name = f"登番{r_id}"

                    w['racer'].configure(text=r_name[:6])
                    w['motor'].configure(text=f"{b.get('motor_rate', 0.0):.1f}%")
                    w['boat'].configure(text=f"{b.get('boat_rate', 0.0):.1f}%")
                    
                    ex_t = b.get('ex_time', 0.0)
                    w['ex'].configure(text=f"{ex_t:.2f}" if ex_t > 0 else "-.--")

                    tilt_v = b.get('tilt', -0.5)
                    w['tilt'].configure(text=f"{tilt_v:+.1f}")

                    st_v = b.get('st_time', 0.20)
                    w['st'].configure(text=f".{int(st_v*100):02d}" if st_v >= 0 else f"F.{int(abs(st_v)*100):02d}")

                    p1_v = b.get('p1_prob', 0.0)
                    w['p1'].configure(text=f"{p1_v:.1%}")

                    # 本命艇ハイライト
                    if bn == top_boat and p1 >= 0.7438:
                        w['frame'].configure(fg_color="#002A18")
                    else:
                        w['frame'].configure(fg_color="#16161D")

        # 4. 【右カラム上段: 黄金ベースライン】
        for child in self.scroll_golden.winfo_children():
            child.destroy()

        if odds_golden:
            for o in odds_golden:
                combo = o.get("combo", "")
                odds_val = o.get("odds", 0.0)
                prob_val = o.get("prob", 0.0)
                ev_val = o.get("ev", 0.0)
                rec_amt = o.get("recommended_amount", 0)
                exp_ret = o.get("expected_return", 0)
                is_rec = o.get("is_recommended", False) or (rec_amt > 0)

                card_bg = "#00391C" if is_rec else "#181820"
                border_col = "#00E676" if is_rec else "transparent"

                card = ctk.CTkFrame(self.scroll_golden, corner_radius=5, fg_color=card_bg, border_width=1 if is_rec else 0, border_color=border_col)
                card.pack(fill="x", pady=1, padx=1)

                top_row = ctk.CTkFrame(card, fg_color="transparent")
                top_row.pack(fill="x", padx=5, pady=(3, 1))

                lbl_c = ctk.CTkLabel(top_row, text=f"{combo}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF" if not is_rec else "#00E676")
                lbl_c.pack(side="left")

                lbl_o = ctk.CTkLabel(top_row, text=f"{odds_val:.1f}倍", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD600")
                lbl_o.pack(side="right")

                bot_row = ctk.CTkFrame(card, fg_color="transparent")
                bot_row.pack(fill="x", padx=5, pady=(0, 3))

                sub_txt = f"EV {ev_val:.2f} ({prob_val:.1%})"
                if is_rec:
                    sub_txt = f"🎯 {rec_amt:,}円 (払戻 {exp_ret:,}円) | EV {ev_val:.2f}"

                lbl_s = ctk.CTkLabel(bot_row, text=sub_txt, font=ctk.CTkFont(size=10), text_color="#00E5FF" if is_rec else "#777777")
                lbl_s.pack(side="left")
        else:
            lbl_none1 = ctk.CTkLabel(self.scroll_golden, text="買い目データなし", font=ctk.CTkFont(size=10), text_color="#777777")
            lbl_none1.pack(pady=10)

        # 5. 【右カラム下段: 的中特化 (Dutching)】
        for child in self.scroll_hit.winfo_children():
            child.destroy()

        if odds_hit:
            for o in odds_hit:
                combo = o.get("combo", "")
                odds_val = o.get("odds", 0.0)
                prob_val = o.get("prob", 0.0)
                rec_amt = o.get("recommended_amount", 0)
                exp_ret = o.get("expected_return", 0)
                profit = o.get("profit", 0)
                is_rec = o.get("is_recommended", False) or (rec_amt > 0)

                card_bg = "#002B3D" if is_rec else "#181820"
                border_col = "#00E5FF" if is_rec else "transparent"

                card = ctk.CTkFrame(self.scroll_hit, corner_radius=5, fg_color=card_bg, border_width=1 if is_rec else 0, border_color=border_col)
                card.pack(fill="x", pady=1, padx=1)

                top_row = ctk.CTkFrame(card, fg_color="transparent")
                top_row.pack(fill="x", padx=5, pady=(3, 1))

                lbl_c = ctk.CTkLabel(top_row, text=f"{combo}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF" if not is_rec else "#00E5FF")
                lbl_c.pack(side="left")

                lbl_o = ctk.CTkLabel(top_row, text=f"{odds_val:.1f}倍", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD600")
                lbl_o.pack(side="right")

                bot_row = ctk.CTkFrame(card, fg_color="transparent")
                bot_row.pack(fill="x", padx=5, pady=(0, 3))

                sub_txt = f"勝率 {prob_val:.1%}"
                if is_rec:
                    sub_txt = f"🛡️ {rec_amt:,}円 (払戻 {exp_ret:,}円 / 利益 {profit:+,}円)"

                lbl_s = ctk.CTkLabel(bot_row, text=sub_txt, font=ctk.CTkFont(size=10), text_color="#00E676" if is_rec else "#777777")
                lbl_s.pack(side="left")
        else:
            lbl_none2 = ctk.CTkLabel(self.scroll_hit, text="的中特化データなし", font=ctk.CTkFont(size=10), text_color="#777777")
            lbl_none2.pack(pady=10)

        # 6. フッター更新
        self.lbl_indicator.configure(text="🟢 リアルタイム監視中 (Live Connected)", text_color="#00E676")
        self.lbl_updated.configure(text=f"最終更新: {updated_at}")


if __name__ == "__main__":
    app = IntegratedLiveRadar()
    app.mainloop()
