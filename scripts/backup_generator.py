#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data sintetis riwayat (preferences + recommendations + user_sessions)
dengan penomoran ID & quiz_attempt yang SELARAS dengan recommender.py:

- SESS_YYYYMMDD_NNN: diurutkan berdasarkan waktu start (first_ts) sesi pada hari tsb
- PREF_YYYYMMDD_NNN & REC_YYYYMMDD_NNN: diurutkan berdasarkan timestamp attempt
  (global chronological order untuk satu hari)
- quiz_attempt: 1..N per session (urutan attempt di sesi)

Jalankan:
  python scripts/generate_synthetic_history_fixed.py
"""

import os
import sys
import math
import random
import calendar
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

import pymysql
import pandas as pd
import numpy as np
from faker import Faker

from flask import Flask
from models.database import init_mysql
import model.recommender as reco  # pastikan modul sesuai struktur project

# ---- CONFIG ----
WIB = ZoneInfo("Asia/Jakarta")
SEED = 42
TARGET_MONTHLY = 1000             # tepat 1000/bulan
DAILY_MIN, DAILY_MAX = 28, 38     # 30-an per hari
TOPK = 3                          # simpan Top-3 di recommendations

YEARS_FORWARD = 3
# mulai 1 bulan depan (tanggal 1):
today = datetime.now(WIB).date()
START = (today.replace(day=1) + relativedelta(months=1))
END = START + relativedelta(years=YEARS_FORWARD) - timedelta(days=1)

# DB creds (samakan dengan app.py)
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASSWORD", "")
DB_NAME = os.getenv("MYSQL_DB", "db_ml_chaos")

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# ---- SETUP FAKER & RANDOM ----
random.seed(SEED)
fake = Faker("id_ID")

# ---- Flask App Initialization ----
_flask_app = None
_recommender_system = None

def init_flask_app():
    """Initialize Flask app from main application"""
    global _flask_app, _recommender_system
    try:
        _flask_app = Flask(__name__)
        _flask_app.config.update(
            MYSQL_HOST=DB_HOST,
            MYSQL_USER=DB_USER,
            MYSQL_PASSWORD=DB_PASS,
            MYSQL_DB=DB_NAME,
            MYSQL_CURSORCLASS='DictCursor',
            SECRET_KEY='dummy_key_for_synthetic_data'
        )
        # init DB & recommender
        init_mysql(_flask_app)
        reco.init_recommender(_flask_app)

        # ambil instance recommender yang sudah trained
        with _flask_app.app_context():
            _recommender_system = reco.recommender_system
        if _recommender_system is None or not getattr(reco.recommender_system, "trained", False):
            raise RuntimeError("Recommender system tidak siap atau tidak trained.")
        print("âœ… Flask app & recommender siap")
        return True
    except Exception as e:
        print(f"âŒ Error inisialisasi Flask app: {e}")
        return False

# ---- HELPER: koneksi & query ----
def get_conn():
    conn = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor, autocommit=False
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = '+07:00'")
    return conn

def load_menu_map(conn):
    """Load mapping nama_minuman -> menu_id"""
    with conn.cursor() as cur:
        cur.execute("SELECT menu_id, nama_minuman FROM menu_items WHERE availability = 'Tersedia'")
        rows = cur.fetchall()
    name_to_id = {r["nama_minuman"].strip().lower(): r["menu_id"] for r in rows}
    print(f"ðŸ“Š Loaded {len(name_to_id)} menu items")
    return name_to_id

# ---- ID sequencer sinkron dengan pola recommender.py ----
class IdSequencer:
    """
    Counter harian per prefix (SESS/PREF/REC) + seed dari DB per tanggal,
    supaya aman jika sudah ada data untuk hari tsb.
    """
    def __init__(self):
        self.counters = {"SESS": {}, "PREF": {}, "REC": {}}

    def _ensure_seed_for_day(self, cur, prefix: str, d: date):
        key = d.strftime("%Y%m%d")
        if key in self.counters[prefix]:
            return
        pattern = f"{prefix}_{key}_%"
        if prefix == "SESS":
            cur.execute("""
                SELECT COUNT(*) AS cnt
                  FROM user_sessions
                 WHERE session_id LIKE %s AND DATE(created_at)=%s
            """, (pattern, d))
        elif prefix == "PREF":
            cur.execute("""
                SELECT COUNT(*) AS cnt
                  FROM preferences
                 WHERE pref_id LIKE %s AND DATE(timestamp)=%s
            """, (pattern, d))
        elif prefix == "REC":
            cur.execute("""
                SELECT COUNT(*) AS cnt
                  FROM recommendations
                 WHERE rec_id LIKE %s AND DATE(generated_at)=%s
            """, (pattern, d))
        else:
            self.counters[prefix][key] = 0
            return
        row = cur.fetchone() or {}
        self.counters[prefix][key] = int(row.get("cnt", 0))

    def seed_day_from_db(self, cur, d: date):
        for p in ("SESS", "PREF", "REC"):
            self._ensure_seed_for_day(cur, p, d)

    def next(self, prefix: str, d: date, cur=None) -> str:
        # optional: jika cursor diberikan, pastikan sudah seeded
        if cur is not None:
            self._ensure_seed_for_day(cur, prefix, d)
        key = d.strftime("%Y%m%d")
        n = self.counters[prefix].get(key, 0) + 1
        self.counters[prefix][key] = n
        return f"{prefix}_{key}_{n:03d}"

# ---- Rencana harian ----
def daily_plan_for_month(year: int, month: int, target=TARGET_MONTHLY):
    days = calendar.monthrange(year, month)[1]
    base = target // days
    counts = [base] * days
    remainder = target - base * days
    idxs = list(range(days))
    random.shuffle(idxs)
    for i in idxs[:remainder]:
        counts[i] += 1
    # clamp
    for i in range(days):
        counts[i] = min(max(counts[i], DAILY_MIN), DAILY_MAX)
    # kecilkan/gedein sampai total == target
    diff = sum(counts) - target
    attempts = 0
    while diff != 0 and attempts < 200:
        i = random.randrange(days)
        if diff > 0 and counts[i] > DAILY_MIN:
            counts[i] -= 1; diff -= 1
        elif diff < 0 and counts[i] < DAILY_MAX:
            counts[i] += 1; diff += 1
        attempts += 1
    return counts

# ---- Sampler preferensi (selaras distribusi recommender fixed) ----
MOODS    = ["energi", "rileks", "menyegarkan", "manis"]
RASAS    = ["asam", "manis", "pahit", "netral"]  # balanced
TEKSTUR  = ["light", "creamy", "heavy", "bubbly"]
KAFEIN   = ["tinggi", "sedang", "rendah", "non-kafein"]
SUHU     = ["dingin", "panas", "bebas"]
BUDGET   = ["low", "mid", "high", "bebas"]

def sample_preferences():
    mood    = random.choices(MOODS,   weights=[0.25, 0.30, 0.30, 0.15])[0]
    rasa    = random.choices(RASAS,   weights=[0.25, 0.25, 0.25, 0.25])[0]
    tekstur = random.choices(TEKSTUR, weights=[0.35, 0.30, 0.10, 0.25])[0]
    kafein  = random.choices(KAFEIN,  weights=[0.25, 0.35, 0.25, 0.15])[0]
    suhu    = random.choices(SUHU,    weights=[0.55, 0.35, 0.10])[0]
    budget  = random.choices(BUDGET,  weights=[0.30, 0.45, 0.20, 0.05])[0]
    return {
        "mood": mood, "rasa": rasa, "tekstur": tekstur,
        "kafein": kafein, "suhu": suhu, "budget": budget
    }

def session_will_give_feedback(prob=0.65) -> bool:
    """
    Menentukan apakah user di sesi ini akan memberikan feedback di attempt terakhir.
    Default 65% memberikan feedback (silakan ubah sesuai realita datamu).
    """
    return random.random() < prob

# ---- Random waktu input 10:00â€“23:00 WIB ----
def random_waktu_input(d: date) -> datetime:
    start = datetime.combine(d, time(10, 0), tzinfo=WIB)
    end   = datetime.combine(d, time(23, 0), tzinfo=WIB)
    delta = end - start
    sec   = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=sec)

# ---- Compute rekomendasi via recommender (akurasi sama seperti web) ----
def compute_recommendations_accurate(preferences_dict, session_id, k=TOPK):
    if _recommender_system is None:
        raise RuntimeError("Recommender system belum diinisialisasi")
    prefs = {
        "mood":   preferences_dict["mood"],
        "rasa":   preferences_dict["rasa"],
        "tekstur":preferences_dict["tekstur"],
        "kafein": preferences_dict["kafein"],
        "suhu":   preferences_dict["suhu"],
        "budget": preferences_dict["budget"],
        "session_id": session_id,
        "nama_customer": None
    }
    with _flask_app.app_context():
        recs = _recommender_system.recommend(prefs)
        return recs[:k]

# ---- DB operations ----

def _normalize_session_status(status: str) -> str:
    # map istilah lama ke ENUM di DB
    mapping = {
        'finished': 'completed',
        'complete': 'completed',
        'done': 'completed',
    }
    status = mapping.get((status or '').lower(), status)
    return status if status in ('active', 'completed', 'abandoned') else 'active'

def ensure_session(cur, session_id, created_at, last_activity, total_quiz_attempts, status):
    status = _normalize_session_status(status)
    cur.execute("""
        INSERT INTO user_sessions (session_id, created_at, last_activity, status, total_quiz_attempts)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            last_activity = GREATEST(IFNULL(last_activity, VALUES(last_activity)), VALUES(last_activity)),
            status = VALUES(status),
            total_quiz_attempts = GREATEST(IFNULL(total_quiz_attempts, 0), VALUES(total_quiz_attempts))
    """, (session_id, created_at, last_activity, status, total_quiz_attempts))

def insert_preference(cur, pref_id, nama, ts, prefs, session_id, quiz_attempt, feedback_status="pending"):
    cur.execute("""
        INSERT INTO preferences
            (pref_id, nama_customer, timestamp, mood, rasa, tekstur, kafein, suhu, budget, 
             session_id, quiz_attempt, feedback_status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        pref_id, nama, ts, prefs["mood"], prefs["rasa"], prefs["tekstur"],
        prefs["kafein"], prefs["suhu"], prefs["budget"], session_id, quiz_attempt, feedback_status
    ))

def insert_recommendations(cur, name_to_id, recs, pref_id, session_id, quiz_attempt, ts, seq, cur_for_seq):
    """Insert recommendation records; rec_id disusun kronologis harian."""
    rows = []
    for rank, r in enumerate(recs, start=1):
        menu_name = r["nama_minuman"].strip().lower()
        menu_id = name_to_id.get(menu_name)
        if not menu_id:
            print(f"âš ï¸ Menu '{r['nama_minuman']}' tidak ditemukan di database")
            continue
        rec_id = seq.next("REC", ts.date(), cur=cur_for_seq)
        rows.append((
            rec_id, pref_id, menu_id, float(r["similarity"]), float(r["final_score"]),
            rank, 1, ts, session_id, quiz_attempt
        ))
    if rows:
        cur.executemany("""
            INSERT INTO recommendations
                (rec_id, pref_id, menu_id, similarity, final_score, rank_position, is_top3, 
                 generated_at, session_id, quiz_attempt)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)
    return len(rows)

def apply_feedback_realistic(cur, pref_id, session_id, quiz_attempt, ts):
    # 60% completed (75% positif), 40% pending
    roll = random.random()
    if roll < 0.60:
        feedback_value = 1 if random.random() < 0.75 else 0
        cur.execute("""
            UPDATE recommendations
               SET feedback=%s, feedback_timestamp=%s
             WHERE pref_id=%s AND session_id=%s AND quiz_attempt=%s
        """, (feedback_value, ts + timedelta(minutes=random.randint(1, 30)),
              pref_id, session_id, quiz_attempt))
        cur.execute("""
            UPDATE preferences
               SET feedback_status='completed'
             WHERE pref_id=%s AND session_id=%s AND quiz_attempt=%s
        """, (pref_id, session_id, quiz_attempt))
        return 'completed'
    return 'pending'

# ---- MAIN ----
def main():
    print(f"ðŸš€ Generate data sintetis {START} s.d. {END} (target {TARGET_MONTHLY}/bulan)")
    print(f"ðŸ“… Range: {(END - START).days} hari")

    if not init_flask_app():
        print("âŒ Gagal inisialisasi Flask app")
        return

    conn = get_conn()
    name_to_id = load_menu_map(conn)
    if not name_to_id:
        print("âŒ Tidak ada menu yang tersedia")
        return

    seq = IdSequencer()

    try:
        with conn.cursor() as cur:
            cur.execute("SET time_zone = '+07:00'")

            total_generated = 0
            total_sessions = 0

            current_date = START
            while current_date <= END:
                # Seed counter dari DB untuk tanggal ini
                seq.seed_day_from_db(cur, current_date)

                plan = daily_plan_for_month(current_date.year, current_date.month, TARGET_MONTHLY)
                days_in_month = len(plan)

                monthly_generated = 0
                monthly_sessions = 0

                for day_idx in range(days_in_month):
                    d = date(current_date.year, current_date.month, day_idx + 1)
                    if d < START or d > END:
                        continue

                    # penting: seed ulang (kalau di tanggal yg sama sudah ada data sebelumnya)
                    seq.seed_day_from_db(cur, d)

                    target_today = plan[day_idx]
                    if target_today == 0:
                        continue

                    # 1) Tentukan banyak sesi; bagi attempt ke sesi
                    avg_attempts_per_session = random.choice([2, 3, 3, 4])  # ~3
                    n_sessions = max(1, round(target_today / avg_attempts_per_session))

                    attempts_left = target_today
                    attempts_per_session = []
                    for si in range(n_sessions):
                        if si == n_sessions - 1:
                            attempts_per_session.append(attempts_left)
                        else:
                            max_attempts = min(5, attempts_left - (n_sessions - si - 1))
                            attempt_count = max(1, min(
                                max_attempts, int(random.gauss(avg_attempts_per_session, 1.0))
                            ))
                            attempts_per_session.append(attempt_count)
                            attempts_left -= attempt_count

                    # 2) Bangun "rencana sesi": waktu per sesi dan meta â€” belum kasih session_id
                    session_plans = []
                    for attempt_count in attempts_per_session:
                        if attempt_count <= 0:
                            continue
                        times = sorted([random_waktu_input(d) for _ in range(attempt_count)])
                        session_plans.append({
                            "times": times,
                            "first_ts": times[0],
                            "last_ts": times[-1],
                            "n_attempts": len(times),
                            "customer_name": fake.name(),
                        })

                    # 3) Urutkan sesi oleh first_ts, lalu assign session_id kronologis harian
                    session_plans.sort(key=lambda x: x["first_ts"])
                    for sp in session_plans:
                        sp["session_id"] = seq.next("SESS", d, cur=cur)

                    # 4) Buat events attempt lintas semua sesi, beri quiz_attempt per sesi,
                    #    LALU sort global by timestamp â†’ pemakaian PREF/REC akan mengikuti urutan ini
                    events = []
                    for sp in session_plans:
                        for qa, ts in enumerate(sp["times"], start=1):
                            events.append({
                                "ts": ts,
                                "session_id": sp["session_id"],
                                "quiz_attempt": qa,
                                "customer_name": sp["customer_name"],
                                "prefs": sample_preferences(),
                            })
                    events.sort(key=lambda e: e["ts"])  # <â€” kunci untuk kronologi PREF/REC

                    # 5) Eksekusi events kronologis â†’ generate PREF & REC (satu-satunya sumber kebenaran)
                    daily_generated = 0

                    # Tentukan sesi mana yang akan memberi feedback di attempt terakhir
                    will_fb = {sp["session_id"]: session_will_give_feedback() for sp in session_plans}
                    # Total attempt per sesi (untuk cek â€œterakhir atau bukanâ€)
                    tot_attempts = {sp["session_id"]: sp["n_attempts"] for sp in session_plans}
                    # Menyimpan status attempt terakhir per sesi
                    last_status_by_session = {}

                    for e in events:  # sudah di-sort global by timestamp
                        session_id   = e["session_id"]
                        quiz_attempt = e["quiz_attempt"]
                        ts           = e["ts"]
                        prefs        = e["prefs"]
                        customer     = e["customer_name"]

                        # 1) Hitung rekomendasi (akurat via recommender)
                        recs = compute_recommendations_accurate(prefs, session_id, k=TOPK)
                        if not recs:
                            print(f"âš ï¸ Tidak ada rekomendasi untuk {ts.date()} attempt {quiz_attempt}")
                            continue

                        # 2) Insert preference (status selalu 'pending' dulu)
                        pref_id = seq.next("PREF", ts.date(), cur=cur)   # ID per tanggal, sinkron sequencer
                        insert_preference(
                            cur, pref_id, customer, ts, prefs,
                            session_id, quiz_attempt, "pending"
                        )

                        # 3) Insert Top-K rekomendasi (feedback masih NULL)
                        _ = insert_recommendations(
                            cur, name_to_id, recs, pref_id,
                            session_id, quiz_attempt, ts, seq, cur_for_seq=cur
                        )

                        # 4) HANYA attempt terakhir yang boleh punya feedback
                        if quiz_attempt == tot_attempts[session_id] and will_fb[session_id]:
                            last_status = apply_feedback_realistic(cur, pref_id, session_id, quiz_attempt, ts)  # 'completed'/'pending'
                        else:
                            last_status = 'pending'
                        last_status_by_session[session_id] = last_status

                        daily_generated += 1

                    # 6) Upsert record sesi (status finished jika attempt terakhir completed)
                    for sp in session_plans:
                        sess_status = 'completed' if last_status_by_session.get(sp["session_id"]) == 'completed' else 'active'
                        ensure_session(cur, sp["session_id"], sp["first_ts"], sp["last_ts"], sp["n_attempts"], sess_status)

                    monthly_generated += daily_generated
                    monthly_sessions  += len(session_plans)

                    # commit per 5 hari
                    if d.day % 5 == 0 and not DRY_RUN:
                        conn.commit()
                        print(f"âœ… {d}: {daily_generated} entries generated (Monthly: {monthly_generated})")

                total_generated += monthly_generated
                total_sessions += monthly_sessions
                print(f"ðŸ“Š {current_date.strftime('%Y-%m')}: {monthly_generated} attempts, {monthly_sessions} sessions")

                # next month
                current_date = (current_date.replace(day=1) + relativedelta(months=1))

            # final commit / rollback
            if DRY_RUN:
                conn.rollback()
                print(f"ðŸ” [DRY_RUN] Total akan digenerate: {total_generated} attempts, {total_sessions} sessions")
            else:
                conn.commit()
                print(f"âœ… [DONE] Total digenerate: {total_generated} attempts, {total_sessions} sessions")

    except Exception as e:
        print(f"âŒ Error during generation: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()