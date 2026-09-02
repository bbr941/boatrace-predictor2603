"""
live_radar.py
🚤 BOATRACE AI 統合ライブモニター (CTkTabview タブストック＆上下2段レイアウト版)
- recent_radars.json の更新を1秒ごとに監視 (最大5レース分ストック)
- 【上段: 情報ビュー (60%)】
    - 左: AI Radar (Gatekeeper P1 / Sniper判定 / 水面気象 / 投資サマリー)
    - 中: 出走表＆直前情報 (1〜6号艇の展示タイム、モーター率、チルト、ST等)
    - 右: 全体オッズ (市場オッズ人気上位30件)
- 【下段: AI結論ビュー (40%) - 左右50:50 グリッド分割】
    - 左: 🎯 黄金ベースライン (Sniper / EV重視 / 資金配分)
    - 右: 🛡️ 的中特化 (Dutching / 累積50% / トリガミ回避 / 利益均等化)
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
RECENT_RADARS_PATH = os.path.join(CURRENT_DIR, "recent_radars.json")
CURRENT_RADAR_PATH = os.path.join(CURRENT_DIR, "current_radar.json")

# 艇番ごとの配色 (背景色, 文字色)
BOAT_COLORS = {
    1: ("#FFFFFF", "#000000"),  # 1号艇: 白 (黒文字)
    2: ("#212121", "#FFFFFF"),  # 2号艇: 黒 (白文字)
    3: ("#E53935", "#FFFFFF"),  # 3号艇: 赤 (白文字)
    4: ("#1E88E5", "#FFFFFF"),  # 4号艇: 青 (白文字)
    5: ("#FDD835", "#000000"),  # 5号艇: 黄 (黒文字)
    6: ("#43A047", "#FFFFFF"),  # 6号艇: 緑 (白文字)
}


class TabbedLiveDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚤 BOATRACE AI Integrated Dashboard (Multi-Race Stock)")
        self.geometry("1220x820")
        self.minsize(1100, 720)

        self.last_mtime = 0

        # UI レイアウト構築
        self.build_shell_ui()

        # 初回データ読み込み & ポーリング開始
        self.poll_radar_files()

    def build_shell_ui(self):
        # =====================================================================
        # 1. 最上部 ヘッダーバー
        # =====================================================================
        self.header_frame = ctk.CTkFrame(self, height=44, corner_radius=10, fg_color="#18181E")
        self.header_frame.pack(fill="x", padx=12, pady=(8, 4))

        self.lbl_app_title = ctk.CTkLabel(
            self.header_frame,
            text="🚤 BOATRACE AI Integrated Live Radar",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#00E5FF"
        )
        self.lbl_app_title.pack(side="left", padx=16, pady=8)

        self.lbl_stock_count = ctk.CTkLabel(
            self.header_frame,
            text="ストック: 0 / 5 レース",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#B0BEC5"
        )
        self.lbl_stock_count.pack(side="left", padx=16, pady=8)

        self.lbl_top_status = ctk.CTkLabel(
            self.header_frame,
            text="🟢 監視中",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00E676"
        )
        self.lbl_top_status.pack(side="right", padx=16, pady=8)

        # =====================================================================
        # 2. メイン タブビュー コンテナ (CTkTabview)
        # =====================================================================
        self.tabview = ctk.CTkTabview(
            self,
            corner_radius=10,
            fg_color="#14141A",
            segmented_button_fg_color="#1E1E28",
            segmented_button_selected_color="#0288D1",
            segmented_button_selected_hover_color="#0277BD",
            segmented_button_unselected_hover_color="#282836",
            text_color="#FFFFFF"
        )
        self.tabview.pack(fill="both", expand=True, padx=12, pady=4)

        # =====================================================================
        # 3. 最下部 フッターバー
        # =====================================================================
        self.footer_frame = ctk.CTkFrame(self, height=26, fg_color="transparent")
        self.footer_frame.pack(fill="x", padx=16, pady=(1, 6))

        self.lbl_footer_indicator = ctk.CTkLabel(
            self.footer_frame,
            text="🟢 リアルタイム監視中 (1秒ポーリング)",
            font=ctk.CTkFont(size=11),
            text_color="#00E676"
        )
        self.lbl_footer_indicator.pack(side="left")

        self.lbl_footer_updated = ctk.CTkLabel(
            self.footer_frame,
            text="最終更新: --:--:--",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.lbl_footer_updated.pack(side="right")

    def poll_radar_files(self):
        """1秒ごとに recent_radars.json (および current_radar.json) の更新日時をチェック"""
        target_path = RECENT_RADARS_PATH if os.path.exists(RECENT_RADARS_PATH) else CURRENT_RADAR_PATH

        if os.path.exists(target_path):
            try:
                mtime = os.path.getmtime(target_path)
                if mtime > self.last_mtime:
                    self.last_mtime = mtime
                    with open(target_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        races_list = data
                    elif isinstance(data, dict):
                        races_list = [data]
                    else:
                        races_list = []

                    self.update_all_tabs(races_list)
            except Exception as e:
                pass
        else:
            self.lbl_footer_indicator.configure(text="🟡 レースデータ待機中...", text_color="#FFB300")

        # 1000ms後に再呼び出し
        self.after(1000, self.poll_radar_files)

    def update_all_tabs(self, races_list):
        """最大5件のレースデータ全件に対してタブを動的生成"""
        if not races_list:
            return

        # 既存選択タブの退避
        current_selection = None
        try:
            current_selection = self.tabview.get()
        except Exception:
            pass

        # 全タブを安全にクリア
        for tab_name in list(self.tabview._tab_dict.keys()):
            try:
                self.tabview.delete(tab_name)
            except Exception:
                pass

        self.lbl_stock_count.configure(text=f"ストック: {len(races_list)} / 5 レース")

        first_tab_name = None
        target_tab_to_set = None
        created_tab_titles = set()

        # リスト内の全レースに対してタブを生成
        for idx, race_data in enumerate(races_list):
            race = race_data.get("race", {})
            v_name = race.get("venue_name") or race_data.get("venue_name", "会場")
            r_no = race.get("race_no") or race_data.get("race_no", idx + 1)
            deadline = race.get("deadline") or race_data.get("deadline", "--:--")

            base_title = f"{v_name} {r_no}R ({deadline})"
            tab_title = base_title
            
            # 重複防止フェイルセーフ (末尾に見えない空白を追加)
            while tab_title in created_tab_titles:
                tab_title += " "
            created_tab_titles.add(tab_title)

            if first_tab_name is None:
                first_tab_name = tab_title

            if current_selection and tab_title.strip() == current_selection.strip():
                target_tab_to_set = tab_title

            # タブ追加 & 画面構築
            tab_frame = self.tabview.add(tab_title)
            self.build_single_race_view(tab_frame, race_data)

        # 以前選択されていたタブ、または先頭タブを選択
        select_tab = target_tab_to_set or first_tab_name
        if select_tab:
            try:
                self.tabview.set(select_tab)
            except Exception:
                pass

        latest_updated = races_list[0].get("updated_at", "--:--:--") if races_list else "--:--:--"
        self.lbl_footer_updated.configure(text=f"最終更新: {latest_updated}")
        self.lbl_footer_indicator.configure(text="🟢 リアルタイム監視中 (Live Connected)", text_color="#00E676")

    def build_single_race_view(self, parent_frame, data):
        """1つのレースデータを【上段60% / 下段40%】の上下2段・確実なグリッド分割で描画"""
        race = data.get("race", {})
        ai = data.get("ai_eval", {})
        boats = data.get("boats", [])
        all_odds = data.get("all_odds", [])
        odds_golden = data.get("odds_golden") or data.get("odds", [])
        odds_hit = data.get("odds_hit_focused", [])

        # 親フレームのグリッド設定 (上段 60% / 下段 40%)
        parent_frame.grid_rowconfigure(0, weight=6)
        parent_frame.grid_rowconfigure(1, weight=4)
        parent_frame.grid_columnconfigure(0, weight=1)

        # =====================================================================
        # 【上段：情報ビュー (60%)】 - 3カラム グリッド
        # =====================================================================
        top_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))

        top_frame.grid_rowconfigure(0, weight=1)
        top_frame.grid_columnconfigure(0, weight=0, minsize=310)  # Radar
        top_frame.grid_columnconfigure(1, weight=1)               # 出走表
        top_frame.grid_columnconfigure(2, weight=0, minsize=340)  # 全体オッズ

        # ---------------------------------------------------------------------
        # 上段-左カラム: AI Radar
        # ---------------------------------------------------------------------
        col_radar = ctk.CTkFrame(top_frame, width=310, corner_radius=8, fg_color="#1E1E26")
        col_radar.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        col_radar.pack_propagate(False)

        lbl_r_title = ctk.CTkLabel(col_radar, text="🛡️ AI Radar (評価 & 判定)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        lbl_r_title.pack(anchor="w", padx=10, pady=(6, 2))

        # Gatekeeper P1
        box_gk = ctk.CTkFrame(col_radar, corner_radius=6, fg_color="#16161D")
        box_gk.pack(fill="x", padx=8, pady=2)

        p1 = ai.get("p1_score") if "p1_score" in ai else data.get("p1_score", 0.0)
        top_boat = ai.get("top_boat") or data.get("top_boat", 1)

        lbl_p1_head = ctk.CTkLabel(box_gk, text=f"Gatekeeper P1 (本命: {top_boat}号艇)", font=ctk.CTkFont(size=10), text_color="#888888")
        lbl_p1_head.pack(anchor="w", padx=8, pady=(4, 0))

        lbl_p1_val = ctk.CTkLabel(box_gk, text=f"{p1:.1%}", font=ctk.CTkFont(size=22, weight="bold"), text_color="#00E676" if p1 >= 0.7438 else "#AAAAAA")
        lbl_p1_val.pack(anchor="w", padx=8, pady=(0, 1))

        prog_p1 = ctk.CTkProgressBar(box_gk, height=8, corner_radius=4)
        prog_p1.pack(fill="x", padx=8, pady=(1, 2))
        prog_p1.set(min(max(p1, 0.0), 1.0))
        prog_p1.configure(progress_color="#00E676" if p1 >= 0.7438 else ("#FFD600" if p1 >= 0.60 else "#666666"))

        lbl_gk_th = ctk.CTkLabel(box_gk, text="基準閾値: 74.38% (黄金ベースライン)", font=ctk.CTkFont(size=9), text_color="#777777")
        lbl_gk_th.pack(anchor="e", padx=8, pady=(0, 4))

        # 判定ステータス
        box_status = ctk.CTkFrame(col_radar, corner_radius=6, fg_color="#16161D")
        box_status.pack(fill="x", padx=8, pady=2)

        status = ai.get("status") or data.get("status", "")
        status_msg = ai.get("status_message") or data.get("status_message", "")

        if status == "investment_go":
            badge_txt, badge_fg, badge_bg = "🚀 投資GOサイン点灯", "#00E676", "#00391C"
        elif status == "sniper_skipped":
            badge_txt, badge_fg, badge_bg = "🎯 Sniper見送り", "#FFB300", "#332B00"
        elif status == "gatekeeper_skipped":
            badge_txt, badge_fg, badge_bg = "☕ Gatekeeper未達", "#B0BEC5", "#263238"
        elif status == "skipped_cluster1":
            badge_txt, badge_fg, badge_bg = "🛑 難水面除外", "#FF5252", "#3B1111"
        elif status == "no_value_bets":
            badge_txt, badge_fg, badge_bg = "🔍 EV未達見送り", "#40C4FF", "#002B3D"
        else:
            badge_txt, badge_fg, badge_bg = f"ℹ️ {status}", "#E0E0E0", "#333333"

        lbl_badge = ctk.CTkLabel(box_status, text=badge_txt, font=ctk.CTkFont(size=13, weight="bold"), text_color=badge_fg, fg_color=badge_bg, corner_radius=5, padx=8, pady=2)
        lbl_badge.pack(padx=8, pady=(5, 2))

        lbl_s_msg = ctk.CTkLabel(box_status, text=status_msg, font=ctk.CTkFont(size=10), text_color="#CCCCCC", wraplength=280)
        lbl_s_msg.pack(padx=8, pady=(0, 5))

        # 水面環境＆サマリー
        box_env = ctk.CTkFrame(col_radar, corner_radius=6, fg_color="#16161D")
        box_env.pack(fill="x", padx=8, pady=2)

        c_name = race.get("cluster_name", "標準水面")
        c_id = race.get("cluster_id", 2)
        wave = ai.get("wave_height") if "wave_height" in ai else data.get("wave_height", 0.0)

        lbl_c_info = ctk.CTkLabel(box_env, text=f"🏟️ 会場: {c_name} (Cluster {c_id}) | 🌊 波: {wave:.1f}cm", font=ctk.CTkFont(size=10), text_color="#B0BEC5")
        lbl_c_info.pack(anchor="w", padx=8, pady=(4, 2))

        tot_golden = ai.get("total_bet_golden") if "total_bet_golden" in ai else data.get("total_bet", 0)
        cnt_golden = ai.get("bets_count_golden") if "bets_count_golden" in ai else data.get("bets_count", 0)
        tot_hit = ai.get("total_bet_hit", 0)
        cnt_hit = ai.get("bets_count_hit", 0)

        lbl_sum1 = ctk.CTkLabel(box_env, text=f"🎯 黄金投資: {tot_golden:,} 円 ({cnt_golden}点)", font=ctk.CTkFont(size=10, weight="bold"), text_color="#00E676")
        lbl_sum1.pack(anchor="w", padx=8, pady=0)

        lbl_sum2 = ctk.CTkLabel(box_env, text=f"🛡️ 的中投資: {tot_hit:,} 円 ({cnt_hit}点)", font=ctk.CTkFont(size=10, weight="bold"), text_color="#00E5FF")
        lbl_sum2.pack(anchor="w", padx=8, pady=(0, 4))

        # ---------------------------------------------------------------------
        # 上段-中央カラム: 出走表＆直前情報
        # ---------------------------------------------------------------------
        col_boats = ctk.CTkFrame(top_frame, corner_radius=8, fg_color="#1E1E26")
        col_boats.grid(row=0, column=1, sticky="nsew", padx=2, pady=0)

        lbl_b_title = ctk.CTkLabel(col_boats, text="📋 出走表 & 直前展示タイム", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        lbl_b_title.pack(anchor="w", padx=10, pady=(6, 2))

        # 表ヘッダー
        tbl_head = ctk.CTkFrame(col_boats, height=22, corner_radius=4, fg_color="#141419")
        tbl_head.pack(fill="x", padx=6, pady=(1, 2))

        headers = [("艇/選手", 95), ("モータ", 55), ("ボート", 50), ("展示T", 50), ("チルト", 45), ("ST", 45), ("P1勝率", 60)]
        for h_text, h_w in headers:
            lbl = ctk.CTkLabel(tbl_head, text=h_text, width=h_w, font=ctk.CTkFont(size=10, weight="bold"), text_color="#9E9E9E")
            lbl.pack(side="left", padx=1)

        boats_container = ctk.CTkFrame(col_boats, fg_color="transparent")
        boats_container.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        if boats:
            for b in boats:
                bn = b.get("boat_number", 1)
                bg_col, fg_col = BOAT_COLORS.get(bn, ("#FFFFFF", "#000000"))
                r_name = b.get("racer_name", f"{bn}号艇")
                r_id = b.get("racer_id", 0)
                if r_id > 0 and r_name == f"{bn}号艇":
                    r_name = f"登番{r_id}"

                is_top_honmei = (bn == top_boat and p1 >= 0.7438)
                row_bg = "#002A18" if is_top_honmei else "#16161D"

                row_f = ctk.CTkFrame(boats_container, height=36, corner_radius=5, fg_color=row_bg)
                row_f.pack(fill="x", pady=1)
                row_f.pack_propagate(False)

                badge = ctk.CTkLabel(row_f, text=f"{bn}", width=20, height=20, font=ctk.CTkFont(size=11, weight="bold"), fg_color=bg_col, text_color=fg_col, corner_radius=3)
                badge.pack(side="left", padx=(5, 3), pady=5)

                lbl_r = ctk.CTkLabel(row_f, text=r_name[:6], width=68, anchor="w", font=ctk.CTkFont(size=10, weight="bold"), text_color="#FFFFFF")
                lbl_r.pack(side="left", padx=1)

                lbl_m = ctk.CTkLabel(row_f, text=f"{b.get('motor_rate', 0.0):.1f}%", width=55, font=ctk.CTkFont(size=10), text_color="#E0E0E0")
                lbl_m.pack(side="left", padx=1)

                lbl_bt = ctk.CTkLabel(row_f, text=f"{b.get('boat_rate', 0.0):.1f}%", width=50, font=ctk.CTkFont(size=10), text_color="#888888")
                lbl_bt.pack(side="left", padx=1)

                ex_t = b.get('ex_time', 0.0)
                lbl_ex = ctk.CTkLabel(row_f, text=f"{ex_t:.2f}" if ex_t > 0 else "-.--", width=50, font=ctk.CTkFont(size=10, weight="bold"), text_color="#FFD600")
                lbl_ex.pack(side="left", padx=1)

                tilt_v = b.get('tilt', -0.5)
                lbl_tl = ctk.CTkLabel(row_f, text=f"{tilt_v:+.1f}", width=45, font=ctk.CTkFont(size=10), text_color="#888888")
                lbl_tl.pack(side="left", padx=1)

                st_v = b.get('st_time', 0.20)
                lbl_st = ctk.CTkLabel(row_f, text=f".{int(st_v*100):02d}" if st_v >= 0 else f"F.{int(abs(st_v)*100):02d}", width=45, font=ctk.CTkFont(size=10), text_color="#E0E0E0")
                lbl_st.pack(side="left", padx=1)

                p1_v = b.get('p1_prob', 0.0)
                lbl_p1 = ctk.CTkLabel(row_f, text=f"{p1_v:.1%}", width=60, font=ctk.CTkFont(size=10, weight="bold"), text_color="#00E5FF")
                lbl_p1.pack(side="left", padx=1)

        # ---------------------------------------------------------------------
        # 上段-右カラム: 全体オッズ (人気順上位30件)
        # ---------------------------------------------------------------------
        col_all_odds = ctk.CTkFrame(top_frame, width=340, corner_radius=8, fg_color="#1E1E26")
        col_all_odds.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=0)
        col_all_odds.pack_propagate(False)

        lbl_ao_title = ctk.CTkLabel(col_all_odds, text="📊 全体オッズ (人気順上位30件)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        lbl_ao_title.pack(anchor="w", padx=10, pady=(6, 2))

        scroll_ao = ctk.CTkScrollableFrame(col_all_odds, fg_color="#16161D", corner_radius=6)
        scroll_ao.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        if all_odds:
            for item in all_odds:
                combo = item.get("combo", "")
                o_val = item.get("odds", 0.0)
                p_val = item.get("prob", 0.0)
                ev_val = item.get("ev", 0.0)

                card = ctk.CTkFrame(scroll_ao, height=26, corner_radius=4, fg_color="#191922")
                card.pack(fill="x", pady=1, padx=1)
                card.pack_propagate(False)

                lbl_c = ctk.CTkLabel(card, text=combo, font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF")
                lbl_c.pack(side="left", padx=(6, 4), pady=2)

                lbl_p = ctk.CTkLabel(card, text=f"勝率 {p_val:.1%}" if p_val > 0 else "", font=ctk.CTkFont(size=9), text_color="#888888")
                lbl_p.pack(side="left", padx=2, pady=2)

                lbl_ev = ctk.CTkLabel(card, text=f"EV {ev_val:.2f}" if ev_val > 0 else "", font=ctk.CTkFont(size=9, weight="bold"), text_color="#00E5FF" if ev_val >= 1.25 else "#777777")
                lbl_ev.pack(side="right", padx=(2, 6), pady=2)

                lbl_o = ctk.CTkLabel(card, text=f"{o_val:.1f}倍", font=ctk.CTkFont(size=10, weight="bold"), text_color="#FFD600")
                lbl_o.pack(side="right", padx=4, pady=2)
        else:
            lbl_none = ctk.CTkLabel(scroll_ao, text="全体オッズデータなし", font=ctk.CTkFont(size=10), text_color="#777777")
            lbl_none.pack(pady=15)

        # =====================================================================
        # 【下段：AI結論ビュー (40%)】 - 左右50:50 グリッド分割
        # =====================================================================
        bot_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        bot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

        bot_frame.grid_rowconfigure(0, weight=1)
        bot_frame.grid_columnconfigure(0, weight=1)
        bot_frame.grid_columnconfigure(1, weight=1)

        # ---------------------------------------------------------------------
        # 下段-左カラム: 🎯 黄金ベースライン (Sniper / EV重視)
        # ---------------------------------------------------------------------
        col_golden = ctk.CTkFrame(bot_frame, corner_radius=8, fg_color="#1E1E26")
        col_golden.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        top_bar_g = ctk.CTkFrame(col_golden, fg_color="transparent")
        top_bar_g.pack(fill="x", padx=8, pady=(6, 2))

        lbl_g_title = ctk.CTkLabel(top_bar_g, text="🎯 黄金ベースライン (Sniper / EV重視)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E676")
        lbl_g_title.pack(side="left")

        lbl_g_sum = ctk.CTkLabel(top_bar_g, text=f"投資: {tot_golden:,}円 ({cnt_golden}点)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E676")
        lbl_g_sum.pack(side="right")

        scroll_g = ctk.CTkScrollableFrame(col_golden, fg_color="#16161D", corner_radius=6)
        scroll_g.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        rec_golden_items = [o for o in odds_golden if o.get("is_recommended") or o.get("recommended_amount", 0) > 0]
        other_golden_items = [o for o in odds_golden if not (o.get("is_recommended") or o.get("recommended_amount", 0) > 0)]
        display_golden = rec_golden_items if rec_golden_items else other_golden_items

        if display_golden and (rec_golden_items or status == "investment_go"):
            for o in display_golden:
                combo = o.get("combo", "")
                odds_val = o.get("odds", 0.0)
                prob_val = o.get("prob", 0.0)
                ev_val = o.get("ev", 0.0)
                rec_amt = o.get("recommended_amount", 0)
                exp_ret = o.get("expected_return", 0)
                is_rec = o.get("is_recommended", False) or (rec_amt > 0)

                card_bg = "#00391C" if is_rec else "#181820"
                border_col = "#00E676" if is_rec else "transparent"

                card = ctk.CTkFrame(scroll_g, corner_radius=5, fg_color=card_bg, border_width=1 if is_rec else 0, border_color=border_col)
                card.pack(fill="x", pady=1, padx=1)

                row1 = ctk.CTkFrame(card, fg_color="transparent")
                row1.pack(fill="x", padx=6, pady=(3, 1))

                lbl_c = ctk.CTkLabel(row1, text=f"{combo}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF" if not is_rec else "#00E676")
                lbl_c.pack(side="left")

                lbl_o = ctk.CTkLabel(row1, text=f"{odds_val:.1f}倍", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD600")
                lbl_o.pack(side="right")

                row2 = ctk.CTkFrame(card, fg_color="transparent")
                row2.pack(fill="x", padx=6, pady=(0, 3))

                sub_txt = f"EV {ev_val:.2f} (勝率 {prob_val:.1%})"
                if is_rec:
                    sub_txt = f"🎯 {rec_amt:,}円 (払戻見込: {exp_ret:,}円) | EV {ev_val:.2f}"

                lbl_s = ctk.CTkLabel(row2, text=sub_txt, font=ctk.CTkFont(size=10), text_color="#00E5FF" if is_rec else "#777777")
                lbl_s.pack(side="left")
        else:
            # 見送りカード表示
            no_bet_card = ctk.CTkFrame(scroll_g, corner_radius=6, fg_color="#181820")
            no_bet_card.pack(fill="x", pady=10, padx=6)
            lbl_no_bet = ctk.CTkLabel(
                no_bet_card,
                text="☕ 見送り (No Bet)\n条件を満たす高EV買い目が存在しないか、判定条件未達です",
                font=ctk.CTkFont(size=11),
                text_color="#888888"
            )
            lbl_no_bet.pack(pady=12, padx=10)

        # ---------------------------------------------------------------------
        # 下段-右カラム: 🛡️ 的中特化 (Dutching / 累積50%)
        # ---------------------------------------------------------------------
        col_hit = ctk.CTkFrame(bot_frame, corner_radius=8, fg_color="#1E1E26")
        col_hit.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        top_bar_h = ctk.CTkFrame(col_hit, fg_color="transparent")
        top_bar_h.pack(fill="x", padx=8, pady=(6, 2))

        lbl_h_title = ctk.CTkLabel(top_bar_h, text="🛡️ 的中特化 (Dutching / トリガミ回避)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00E5FF")
        lbl_h_title.pack(side="left")

        lbl_h_sum = ctk.CTkLabel(top_bar_h, text=f"投資: {tot_hit:,}円 ({cnt_hit}点)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E5FF")
        lbl_h_sum.pack(side="right")

        scroll_h = ctk.CTkScrollableFrame(col_hit, fg_color="#16161D", corner_radius=6)
        scroll_h.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        rec_hit_items = [o for o in odds_hit if o.get("is_recommended") or o.get("recommended_amount", 0) > 0]
        other_hit_items = [o for o in odds_hit if not (o.get("is_recommended") or o.get("recommended_amount", 0) > 0)]
        display_hit = rec_hit_items if rec_hit_items else other_hit_items

        if display_hit and (rec_hit_items or tot_hit > 0):
            for o in display_hit:
                combo = o.get("combo", "")
                odds_val = o.get("odds", 0.0)
                prob_val = o.get("prob", 0.0)
                rec_amt = o.get("recommended_amount", 0)
                exp_ret = o.get("expected_return", 0)
                profit = o.get("profit", 0)
                is_rec = o.get("is_recommended", False) or (rec_amt > 0)

                card_bg = "#002B3D" if is_rec else "#181820"
                border_col = "#00E5FF" if is_rec else "transparent"

                card = ctk.CTkFrame(scroll_h, corner_radius=5, fg_color=card_bg, border_width=1 if is_rec else 0, border_color=border_col)
                card.pack(fill="x", pady=1, padx=1)

                row1 = ctk.CTkFrame(card, fg_color="transparent")
                row1.pack(fill="x", padx=6, pady=(3, 1))

                lbl_c = ctk.CTkLabel(row1, text=f"{combo}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFFFFF" if not is_rec else "#00E5FF")
                lbl_c.pack(side="left")

                lbl_o = ctk.CTkLabel(row1, text=f"{odds_val:.1f}倍", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD600")
                lbl_o.pack(side="right")

                row2 = ctk.CTkFrame(card, fg_color="transparent")
                row2.pack(fill="x", padx=6, pady=(0, 3))

                sub_txt = f"勝率 {prob_val:.1%}"
                if is_rec:
                    sub_txt = f"🛡️ {rec_amt:,}円 (払戻 {exp_ret:,}円 / 利益 {profit:+,}円)"

                lbl_s = ctk.CTkLabel(row2, text=sub_txt, font=ctk.CTkFont(size=10), text_color="#00E676" if is_rec else "#777777")
                lbl_s.pack(side="left")
        else:
            # 見送りカード表示
            no_hit_card = ctk.CTkFrame(scroll_h, corner_radius=6, fg_color="#181820")
            no_hit_card.pack(fill="x", pady=10, padx=6)
            lbl_no_hit = ctk.CTkLabel(
                no_hit_card,
                text="☕ 見送り (No Bet)\nダッチング対象買い目なし",
                font=ctk.CTkFont(size=11),
                text_color="#888888"
            )
            lbl_no_hit.pack(pady=12, padx=10)


if __name__ == "__main__":
    app = TabbedLiveDashboard()
    app.mainloop()
