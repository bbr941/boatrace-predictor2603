"""
live_radar.py
🚤 BOATRACE AI リアルタイム・ライブモニター (CustomTkinter版)
- current_radar.json の更新を1秒ごとに監視し、最新の推論状態をリアルタイム表示
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

class LiveRadarApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚤 BOATRACE AI Live Radar")
        self.geometry("540x520")
        self.minsize(500, 480)

        self.last_mtime = 0

        # UI レイアウト構築
        self.build_ui()

        # 初回データ読み込み & ポーリング開始
        self.poll_radar_file()

    def build_ui(self):
        # 1. ヘッダーフレーム
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E1E24")
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))

        self.lbl_app_title = ctk.CTkLabel(
            self.header_frame,
            text="🚤 BOATRACE AI Live Radar",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00E5FF"
        )
        self.lbl_app_title.pack(anchor="w", padx=15, pady=(8, 2))

        self.lbl_race_name = ctk.CTkLabel(
            self.header_frame,
            text="待機中... (レース未取得)",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF"
        )
        self.lbl_race_name.pack(anchor="w", padx=15, pady=(0, 2))

        self.lbl_deadline = ctk.CTkLabel(
            self.header_frame,
            text="締切時刻: --:--",
            font=ctk.CTkFont(size=13),
            text_color="#AAAAAA"
        )
        self.lbl_deadline.pack(anchor="w", padx=15, pady=(0, 8))

        # 2. Gatekeeper P1 信頼度フレーム
        self.gk_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E1E24")
        self.gk_frame.pack(fill="x", padx=15, pady=5)

        self.gk_title_box = ctk.CTkFrame(self.gk_frame, fg_color="transparent")
        self.gk_title_box.pack(fill="x", padx=15, pady=(8, 2))

        self.lbl_gk_title = ctk.CTkLabel(
            self.gk_title_box,
            text="🛡️ Gatekeeper 1着信頼度 (P1)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#DDDDDD"
        )
        self.lbl_gk_title.pack(side="left")

        self.lbl_p1_val = ctk.CTkLabel(
            self.gk_title_box,
            text="--.-%",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#AAAAAA"
        )
        self.lbl_p1_val.pack(side="right")

        self.progress_p1 = ctk.CTkProgressBar(self.gk_frame, height=14, corner_radius=7)
        self.progress_p1.pack(fill="x", padx=15, pady=(4, 6))
        self.progress_p1.set(0.0)
        self.progress_p1.configure(progress_color="#666666")

        self.lbl_gk_desc = ctk.CTkLabel(
            self.gk_frame,
            text="基準閾値: 74.38% (黄金ベースライン)",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.lbl_gk_desc.pack(anchor="e", padx=15, pady=(0, 8))

        # 3. レース環境・モーター情報フレーム (2カラム)
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=15, pady=5)

        # モーター勝率カード
        self.card_motor = ctk.CTkFrame(self.stats_frame, corner_radius=8, fg_color="#1E1E24")
        self.card_motor.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.lbl_motor_title = ctk.CTkLabel(self.card_motor, text="1号艇 モーター勝率", font=ctk.CTkFont(size=11), text_color="#888888")
        self.lbl_motor_title.pack(padx=10, pady=(6, 0))

        self.lbl_motor_val = ctk.CTkLabel(self.card_motor, text="--.-%", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFFFFF")
        self.lbl_motor_val.pack(padx=10, pady=(0, 6))

        # 波高カード
        self.card_wave = ctk.CTkFrame(self.stats_frame, corner_radius=8, fg_color="#1E1E24")
        self.card_wave.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.lbl_wave_title = ctk.CTkLabel(self.card_wave, text="波高 (水面コンディション)", font=ctk.CTkFont(size=11), text_color="#888888")
        self.lbl_wave_title.pack(padx=10, pady=(6, 0))

        self.lbl_wave_val = ctk.CTkLabel(self.card_wave, text="-- cm", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFFFFF")
        self.lbl_wave_val.pack(padx=10, pady=(0, 6))

        # 4. 最終判定バッジ & ステータスメッセージ
        self.status_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#24242E")
        self.status_frame.pack(fill="x", padx=15, pady=8)

        self.lbl_status_badge = ctk.CTkLabel(
            self.status_frame,
            text="⏳ 待機中",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFB300",
            fg_color="#332B00",
            corner_radius=6,
            padx=12,
            pady=4
        )
        self.lbl_status_badge.pack(padx=12, pady=(10, 4))

        self.lbl_status_msg = ctk.CTkLabel(
            self.status_frame,
            text="auto_trader.py による次の推論実行を待機しています...",
            font=ctk.CTkFont(size=12),
            text_color="#CCCCCC",
            wraplength=480
        )
        self.lbl_status_msg.pack(padx=12, pady=(0, 10))

        # 5. 推奨買い目フレーム
        self.bets_frame = ctk.CTkFrame(self, corner_radius=8, fg_color="#1E1E24")
        self.bets_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.lbl_bets_title = ctk.CTkLabel(
            self.bets_frame,
            text="🎯 推奨ポートフォリオ (SLSQP最適化)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#AAAAAA"
        )
        self.lbl_bets_title.pack(anchor="w", padx=12, pady=(6, 2))

        self.txt_bets = ctk.CTkTextbox(
            self.bets_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#16161A",
            text_color="#EEEEEE"
        )
        self.txt_bets.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.txt_bets.insert("1.0", "推奨買い目はありません。")
        self.txt_bets.configure(state="disabled")

        # 6. フッター
        self.footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.footer_frame.pack(fill="x", padx=15, pady=(0, 10))

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
        """受信データに基づきGUIを更新"""
        race_name = data.get("race_name", "不明なレース")
        deadline = data.get("deadline", "--:--")
        p1 = data.get("p1_score", 0.0)
        top_boat = data.get("top_boat", 1)
        gk_passed = data.get("gatekeeper_passed", False)
        motor = data.get("motor_rate", 0.0)
        wave = data.get("wave_height", 0.0)
        status = data.get("status", "")
        status_msg = data.get("status_message", "")
        bets = data.get("bets", [])
        total_bet = data.get("total_bet", 0)
        updated_at = data.get("updated_at", "")

        # ヘッダー
        self.lbl_race_name.configure(text=race_name)
        self.lbl_deadline.configure(text=f"締切時刻: {deadline} (本命: {top_boat}号艇)")

        # P1 プログレスバー & 数値
        self.lbl_p1_val.configure(text=f"{p1:.1%}")
        self.progress_p1.set(min(max(p1, 0.0), 1.0))

        if p1 >= 0.7438:
            self.progress_p1.configure(progress_color="#00E676")  # エメラルドグリーン
            self.lbl_p1_val.configure(text_color="#00E676")
        elif p1 >= 0.60:
            self.progress_p1.configure(progress_color="#FFD600")  # イエロー
            self.lbl_p1_val.configure(text_color="#FFD600")
        else:
            self.progress_p1.configure(progress_color="#757575")  # グレー
            self.lbl_p1_val.configure(text_color="#AAAAAA")

        # モーター & 波高
        self.lbl_motor_val.configure(text=f"{motor:.1f}%")
        self.lbl_wave_val.configure(text=f"{wave:.1f} cm")

        # ステータスバッジ配色
        if status == "investment_go":
            self.lbl_status_badge.configure(
                text="🚀 投資GOサイン点灯",
                text_color="#00E676",
                fg_color="#00391C"
            )
        elif status == "sniper_skipped":
            self.lbl_status_badge.configure(
                text="🎯 Sniper見送り",
                text_color="#FFB300",
                fg_color="#332B00"
            )
        elif status == "gatekeeper_skipped":
            self.lbl_status_badge.configure(
                text="☕ Gatekeeper未達",
                text_color="#B0BEC5",
                fg_color="#263238"
            )
        elif status == "skipped_cluster1":
            self.lbl_status_badge.configure(
                text="🛑 難水面除外",
                text_color="#FF5252",
                fg_color="#3B1111"
            )
        elif status == "no_value_bets":
            self.lbl_status_badge.configure(
                text="🔍 EV未達見送り",
                text_color="#40C4FF",
                fg_color="#002B3D"
            )
        else:
            self.lbl_status_badge.configure(
                text=f"ℹ️ {status}",
                text_color="#E0E0E0",
                fg_color="#333333"
            )

        self.lbl_status_msg.configure(text=status_msg)

        # 買い目テキスト
        self.txt_bets.configure(state="normal")
        self.txt_bets.delete("1.0", "end")

        if bets:
            lines = [f"【推奨投資総額: {total_bet:,} 円 / {len(bets)}点】\n"]
            for b in bets:
                combo = b.get("combo", "")
                amt = b.get("amount", 0)
                odds = b.get("odds", 0.0)
                ev = b.get("ev", 0.0)
                exp_ret = b.get("expected_return", 0)
                lines.append(f"  ・{combo:<5} : {amt:>5,d} 円 (オッズ {odds:>5.1f}倍 | EV {ev:>4.2f} | 払戻見込 {exp_ret:>6,d}円)")
            self.txt_bets.insert("1.0", "\n".join(lines))
        else:
            self.txt_bets.insert("1.0", "（投資対象の買い目はありません）")

        self.txt_bets.configure(state="disabled")

        # フッター
        self.lbl_indicator.configure(text="🟢 リアルタイム監視中 (Live Connected)", text_color="#00E676")
        self.lbl_updated.configure(text=f"最終更新: {updated_at}")

if __name__ == "__main__":
    app = LiveRadarApp()
    app.mainloop()
