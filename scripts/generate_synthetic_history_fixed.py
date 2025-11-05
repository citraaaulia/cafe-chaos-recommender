#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data sintetis riwayat dengan pola bisnis yang realistis - FIXED VERSION
- Total TEPAT 36,000 preferences selama 3 tahun (Agustus 2025 - Agustus 2028)
- Distribusi tahunan: Year 1 (~8,000), Year 2 (~12,000), Year 3 (~16,000)
- Pola pertumbuhan organik dengan fluktuasi seasonal yang natural
- 70% user sessions memberikan feedback, 50% dari feedback adalah positif
- Fixed scaling algorithm dengan proper rounding dan validation

Jalankan:
  python scripts/generate_synthetic_history_fixed_v2.py
"""

import os
import sys
import math
import random
import calendar
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_EVEN

import pymysql
import pandas as pd
import numpy as np
from faker import Faker

from flask import Flask
from models.database import init_mysql
import model.recommender as reco

# ---- CONFIG ----
WIB = ZoneInfo("Asia/Jakarta")
SEED = 42
TARGET_TOTAL = 36000             # Total target 3 tahun - MUST BE EXACT
TOPK = 3                         # Top-3 recommendations
FEEDBACK_SESSION_RATE = 0.70     # 70% sesi memberikan feedback
POSITIVE_FEEDBACK_RATE = 0.50    # 50% feedback adalah positif

START = date(2025, 8, 1)
END   = date(2028, 7, 31)  # Exactly 36 months: Aug 2025 - Jul 2028

# DB credentials
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASSWORD", "")
DB_NAME = os.getenv("MYSQL_DB", "db_ml_chaos")

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# ---- SETUP ----
random.seed(SEED)
np.random.seed(SEED)
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
        init_mysql(_flask_app)
        reco.init_recommender(_flask_app)

        with _flask_app.app_context():
            _recommender_system = reco.recommender_system
        if _recommender_system is None or not getattr(reco.recommender_system, "trained", False):
            raise RuntimeError("Recommender system tidak siap atau tidak trained.")
        print("✅ Flask app & recommender siap")
        return True
    except Exception as e:
        print(f"❌ Error inisialisasi Flask app: {e}")
        return False

def _parse_enum_or_varchar(type_str):
    t = (type_str or "").lower()
    if t.startswith("enum("):
        inside = t[t.find("(")+1:t.rfind(")")]
        allowed = [s.strip().strip("'") for s in inside.split(",")]
        return {"kind": "enum", "allowed": set(allowed)}
    if t.startswith("varchar("):
        n = int(t[t.find("(")+1:t.find(")")])
        return {"kind": "varchar", "maxlen": n}
    return {"kind": "other"}

def _get_session_status_meta(cur):
    cur.execute("SHOW COLUMNS FROM user_sessions LIKE 'status'")
    row = cur.fetchone() or {}
    return _parse_enum_or_varchar(row.get("Type", ""))

def _normalize_session_status_safe(status: str, meta: dict) -> str:
    s = (status or "").lower().strip()
    synonyms = {
        "completed": {"completed","complete","done","finished"},
        "finished":  {"finished","complete","done"},
        "active":    {"active","ongoing","in-progress"},
        "pending":   {"pending","wait","waiting","queued"},
        "abandoned": {"abandoned","cancelled","canceled","dropped"},
    }
    canon = "active"
    for k, vals in synonyms.items():
        if s in vals: canon = k; break

    if meta.get("kind") == "enum":
        allowed = meta["allowed"]
        for opt in [canon, "completed", "finished", "active", "pending", "abandoned"]:
            if opt in allowed: return opt
        return next(iter(allowed))

    if meta.get("kind") == "varchar":
        chosen = "finished" if (canon == "completed" and meta["maxlen"] < len("completed")) else canon
        return chosen[:meta["maxlen"]]

    return canon

# ---- FIXED BUSINESS GROWTH MODEL ----
def generate_organic_monthly_distribution():
    """
    Generate distribusi bulanan yang sangat realistis dengan target:
    Year 1: ~8,000 entries (22.2%)
    Year 2: ~12,000 entries (33.3%) 
    Year 3: ~16,000 entries (44.4%)
    
    FIXED: Base values disesuaikan untuk mencapai distribusi yang tepat
    """
    
    # YEAR 1: Exponential growth phase (Target: 8,000)
    year1_patterns = [
        # Q1: Building momentum - BOOSTED
        (450, "Soft launch dengan early adopter program"),
        (420, "Word-of-mouth mulai bekerja - februari boost"),
        (580, "Spring campaign - first major marketing push"),
        
        # Q2: Validation phase - ENHANCED
        (680, "Easter/Ramadan boost - cultural moment drives usage"),
        (750, "Product validation - positive feedback loop starts"),
        (720, "Mid-year consolidation - optimizing based on learnings"),
        
        # Q3: Viral growth - INCREASED
        (850, "Summer peak - social sharing increases organically"),
        (780, "Back-to-school adjustment - temporary dip"),
        (920, "Autumn word-of-mouth explosion - recommendation accuracy improves"),
        
        # Q4: Holiday momentum - STRONGER
        (1050, "Pre-holiday marketing - customer acquisition campaigns"),
        (1200, "Black Friday/holiday peak - highest conversion rates"),
        (1100, "Year-end retention - loyalty program effectiveness")
    ]
    
    # YEAR 2: Sustained growth (Target: 12,000)
    year2_patterns = [
        # Q1: New year momentum
        (1200, "New year resolution traffic - health-conscious choices peak"),
        (950, "Post-holiday normalization - expected seasonal adjustment"),
        (1300, "Spring marketing campaign - targeting increased after Y1 success"),
        
        # Q2: Marketing maturity
        (1450, "Easter/cultural events - proven seasonal strategy"),
        (1350, "Steady growth - word-of-mouth now primary driver"),
        (1180, "Summer preparation - menu optimization for season"),
        
        # Q3: Operational excellence
        (1250, "Summer peak execution - improved recommendation engine"),
        (980, "Late summer dip - natural beverage preference shift"),
        (1220, "Back-to-school recovery - targeted student campaigns"),
        
        # Q4: Holiday optimization
        (1400, "Pre-holiday preparation - data-driven approach"),
        (1550, "Holiday peak - optimized for maximum satisfaction"),
        (1170, "Year-end analysis - customer lifetime value focus")
    ]
    
    # YEAR 3: Market leadership (Target: 16,000) - ADJUSTED DOWN SLIGHTLY
    year3_patterns = [
        # Q1: Market expansion
        (1400, "Market leadership established - brand recognition high"),
        (1150, "Strategic pause - analyzing expansion opportunities"),
        (1580, "Partnership launches - B2B relationships drive growth"),
        
        # Q2: Premium growth
        (1700, "Premium feature launch - higher-value customer acquisition"),
        (1600, "Sustained premium growth - customer satisfaction at peak"),
        (1350, "Mid-year optimization - operational efficiency improvements"),
        
        # Q3: Efficiency focus
        (1550, "Operational excellence - AI recommendations highly accurate"),
        (1200, "Strategic consolidation - focusing on high-value segments"),
        (1420, "Autumn expansion - new market penetration"),
        
        # Q4: Strategic growth
        (1680, "Holiday season leadership - dominant market position"),
        (1800, "Peak performance - all systems optimized"),
        (1570, "Strategic planning - preparing for next phase")
    ]
    
    # Compile all patterns
    all_patterns = [
        (year1_patterns, "Year 1 - Startup & Exponential Growth"),
        (year2_patterns, "Year 2 - Sustained Growth & Optimization"),
        (year3_patterns, "Year 3 - Market Leadership & Expansion")
    ]
    
    monthly_data = []
    all_targets = []
    
    # Add realistic randomness while maintaining business logic
    for year_idx, (year_patterns, phase_name) in enumerate(all_patterns):
        print(f"\n📈 {phase_name}")
        year_total = 0
        
        for month_idx, (base_target, business_reason) in enumerate(year_patterns):
            # Add natural variance (±5% to maintain better control)
            variance = random.uniform(-0.05, 0.05)
            monthly_target = int(base_target * (1 + variance))
            
            # Ensure minimum viable business
            monthly_target = max(monthly_target, 200)
            
            month_name = calendar.month_name[(month_idx % 12) + 1]
            monthly_data.append({
                'target': monthly_target,
                'reason': business_reason,
                'month_name': month_name,
                'year': year_idx + 1
            })
            all_targets.append(monthly_target)
            year_total += monthly_target
            print(f"  {month_name}: {monthly_target:,} - {business_reason}")
        
        print(f"  📊 {phase_name} Total: {year_total:,}")
    
    return all_targets, monthly_data

def ensure_exact_total(monthly_targets, target_total):
    """
    FIXED: Ensure exact total dengan intelligent distribution
    """
    current_total = sum(monthly_targets)
    diff = target_total - current_total
    
    print(f"🔧 Adjusting total: {current_total:,} → {target_total:,} (diff: {diff:+,})")
    
    if diff == 0:
        return monthly_targets
    
    # Create a copy to avoid modifying original
    adjusted = monthly_targets[:]
    
    # Sort indices by month value (prefer adjusting larger months)
    month_indices = list(range(len(adjusted)))
    
    attempts = 0
    max_attempts = abs(diff) * 2  # Prevent infinite loops
    
    while diff != 0 and attempts < max_attempts:
        if diff > 0:
            # Add to months, prefer later months (higher growth period)
            # Weighted random selection favoring months 12-35 (Year 2-3)
            weights = []
            for i in range(len(adjusted)):
                if i < 12:  # Year 1
                    weights.append(1)
                elif i < 24:  # Year 2
                    weights.append(2)
                else:  # Year 3
                    weights.append(3)
            
            idx = random.choices(month_indices, weights=weights)[0]
            adjusted[idx] += 1
            diff -= 1
        else:
            # Remove from months, prefer early months with higher values
            # Only remove if month has >200 (minimum viable)
            valid_indices = [i for i in month_indices if adjusted[i] > 200]
            if valid_indices:
                # Prefer removing from Year 1 (indices 0-11)
                year1_indices = [i for i in valid_indices if i < 12]
                if year1_indices:
                    idx = random.choice(year1_indices)
                else:
                    idx = random.choice(valid_indices)
                adjusted[idx] -= 1
                diff += 1
            else:
                break
        
        attempts += 1
    
    final_total = sum(adjusted)
    print(f"✅ Final adjustment: {final_total:,} (attempts: {attempts})")
    
    if final_total != target_total:
        print(f"⚠️  Could not achieve exact total. Diff: {target_total - final_total:+,}")
    
    return adjusted

# ---- DB CONNECTION & UTILITIES ----
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
    print(f"📊 Loaded {len(name_to_id)} menu items")
    return name_to_id

# ---- ID SEQUENCER ----
class IdSequencer:
    def __init__(self):
        self.counters = {"SESS": {}, "PREF": {}, "REC": {}}

    def _ensure_seed_for_day(self, cur, prefix: str, d: date):
        key = d.strftime("%Y%m%d")
        if key in self.counters[prefix]:
            return
        pattern = f"{prefix}_{key}_%"
        if prefix == "SESS":
            cur.execute("""
                SELECT COUNT(*) AS cnt FROM user_sessions
                WHERE session_id LIKE %s AND DATE(created_at)=%s
            """, (pattern, d))
        elif prefix == "PREF":
            cur.execute("""
                SELECT COUNT(*) AS cnt FROM preferences
                WHERE pref_id LIKE %s AND DATE(timestamp)=%s
            """, (pattern, d))
        elif prefix == "REC":
            cur.execute("""
                SELECT COUNT(*) AS cnt FROM recommendations
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
        if cur is not None:
            self._ensure_seed_for_day(cur, prefix, d)
        key = d.strftime("%Y%m%d")
        n = self.counters[prefix].get(key, 0) + 1
        self.counters[prefix][key] = n
        return f"{prefix}_{key}_{n:03d}"

# ---- PREFERENCE SAMPLING ----
MOODS    = ["energi", "rileks", "menyegarkan", "manis"]
RASAS    = ["asam", "manis", "pahit", "netral"]
TEKSTUR  = ["light", "creamy", "heavy", "bubbly"]
KAFEIN   = ["tinggi", "sedang", "rendah", "non-kafein"]
SUHU     = ["dingin", "panas", "bebas"]
BUDGET   = ["low", "mid", "high", "bebas"]

def sample_preferences_seasonally_aware(month: int):
    """Sample preferences dengan seasonal awareness"""
    # Base weights
    mood_weights = [0.25, 0.30, 0.30, 0.15]
    suhu_weights = [0.55, 0.35, 0.10]
    budget_weights = [0.30, 0.45, 0.20, 0.05]
    
    # Seasonal adjustments
    if month in [6, 7, 8, 9]:  # Musim kemarau
        suhu_weights = [0.70, 0.20, 0.10]
        mood_weights = [0.20, 0.25, 0.40, 0.15]
    elif month in [12, 1, 2]:  # Musim hujan/dingin
        suhu_weights = [0.40, 0.50, 0.10]
        mood_weights = [0.35, 0.30, 0.20, 0.15]
    elif month in [11, 12]:  # Holiday season
        budget_weights = [0.20, 0.40, 0.35, 0.05]
        mood_weights = [0.20, 0.25, 0.25, 0.30]
    
    mood = random.choices(MOODS, weights=mood_weights)[0]
    rasa = random.choices(RASAS, weights=[0.25, 0.25, 0.25, 0.25])[0]
    tekstur = random.choices(TEKSTUR, weights=[0.35, 0.30, 0.10, 0.25])[0]
    kafein = random.choices(KAFEIN, weights=[0.25, 0.35, 0.25, 0.15])[0]
    suhu = random.choices(SUHU, weights=suhu_weights)[0]
    budget = random.choices(BUDGET, weights=budget_weights)[0]
    
    return {
        "mood": mood, "rasa": rasa, "tekstur": tekstur,
        "kafein": kafein, "suhu": suhu, "budget": budget
    }

def session_will_give_feedback(prob=FEEDBACK_SESSION_RATE) -> bool:
    """70% user sessions memberikan feedback"""
    return random.random() < prob

def feedback_will_be_positive(prob=POSITIVE_FEEDBACK_RATE) -> bool:
    """50% feedback adalah positif"""
    return random.random() < prob

# ---- TIME SAMPLING - FIXED ----
def random_waktu_input(d: date) -> datetime:
    """Generate random input time dengan weighted distribution - FIXED"""
    # FIXED: Ensure exact match between hours and weights
    hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    weights = [1, 2, 2, 3, 4, 3, 5, 5, 4, 3, 2, 2, 4, 4, 3, 2, 1]
    
    # Verify lengths match
    assert len(hours) == len(weights), f"Mismatch: {len(hours)} hours vs {len(weights)} weights"
    
    hour = random.choices(hours, weights=weights)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    dt = datetime.combine(d, time(hour, minute, second), tzinfo=WIB)
    return dt

# ---- RECOMMENDATION ENGINE ----
def compute_recommendations_accurate(preferences_dict, session_id, k=TOPK):
    """Compute recommendations menggunakan trained recommender system"""
    if _recommender_system is None:
        raise RuntimeError("Recommender system belum diinisialisasi")
    
    prefs = {
        "mood": preferences_dict["mood"],
        "rasa": preferences_dict["rasa"],
        "tekstur": preferences_dict["tekstur"],
        "kafein": preferences_dict["kafein"],
        "suhu": preferences_dict["suhu"],
        "budget": preferences_dict["budget"],
        "session_id": session_id,
        "nama_customer": None
    }
    
    with _flask_app.app_context():
        recs = _recommender_system.recommend(prefs)
        return recs[:k]

# ---- DATABASE OPERATIONS ----
_status_meta_cache = None

def ensure_session(cur, session_id, created_at, last_activity, total_quiz_attempts, status):
    global _status_meta_cache
    if _status_meta_cache is None:
        _status_meta_cache = _get_session_status_meta(cur)

    safe_status = _normalize_session_status_safe(status, _status_meta_cache)

    cur.execute("""
        INSERT INTO user_sessions (session_id, created_at, last_activity, status, total_quiz_attempts)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            last_activity = GREATEST(IFNULL(last_activity, VALUES(last_activity)), VALUES(last_activity)),
            status = VALUES(status),
            total_quiz_attempts = GREATEST(IFNULL(total_quiz_attempts, 0), VALUES(total_quiz_attempts))
    """, (session_id, created_at, last_activity, safe_status, total_quiz_attempts))

def insert_preference(cur, pref_id, nama, ts, prefs, session_id, quiz_attempt, feedback_status="pending"):
    """Insert preference record"""
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
    """Insert recommendation records"""
    rows = []
    for rank, r in enumerate(recs, start=1):
        menu_name = r["nama_minuman"].strip().lower()
        menu_id = name_to_id.get(menu_name)
        if not menu_id:
            print(f"⚠️ Menu '{r['nama_minuman']}' tidak ditemukan di database")
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

def apply_feedback_realistic(cur, pref_id, session_id, quiz_attempt, ts, will_be_positive):
    """Apply feedback dengan distribusi realistis"""
    feedback_value = 1 if will_be_positive else 0
    feedback_delay = random.randint(1, 45)  # 1-45 menit delay
    
    cur.execute("""
        UPDATE recommendations
           SET feedback=%s, feedback_timestamp=%s
         WHERE pref_id=%s AND session_id=%s AND quiz_attempt=%s
    """, (feedback_value, ts + timedelta(minutes=feedback_delay),
          pref_id, session_id, quiz_attempt))
    
    cur.execute("""
        UPDATE preferences
           SET feedback_status='completed'
         WHERE pref_id=%s AND session_id=%s AND quiz_attempt=%s
    """, (pref_id, session_id, quiz_attempt))
    
    return 'completed'

# ---- ENHANCED DAILY DISTRIBUTION ----
def distribute_monthly_to_daily_organic(monthly_target, year, month):
    """Distribute bulanan ke harian dengan pola realistis"""
    days_in_month = calendar.monthrange(year, month)[1]
    daily_targets = []
    base_daily = monthly_target / days_in_month
    
    for day in range(1, days_in_month + 1):
        current_date = date(year, month, day)
        weekday = current_date.weekday()
        
        # Weekday patterns
        if weekday == 0:  # Monday
            multiplier = random.uniform(0.70, 0.85)
        elif weekday in [1, 2, 3]:  # Tue-Thu
            multiplier = random.uniform(0.95, 1.10)
        elif weekday == 4:  # Friday
            multiplier = random.uniform(1.15, 1.30)
        elif weekday == 5:  # Saturday
            multiplier = random.uniform(1.20, 1.40)
        else:  # Sunday
            multiplier = random.uniform(1.05, 1.25)
        
        # Month progression
        if day <= 5:
            trend_factor = random.uniform(1.05, 1.15)
        elif day <= 15:
            trend_factor = random.uniform(0.95, 1.05)
        else:
            trend_factor = random.uniform(0.90, 1.10)
        
        # Special dates
        if day in [1, 15]:  # Payday
            trend_factor *= random.uniform(1.10, 1.20)
        elif day == 17 and month == 8:  # Independence Day
            trend_factor *= random.uniform(1.15, 1.25)
        
        # Holiday seasons
        if month == 12 and day >= 20:
            trend_factor *= random.uniform(1.20, 1.35)
        elif month == 1 and day <= 10:
            trend_factor *= random.uniform(1.10, 1.25)
        
        daily_target = int(base_daily * multiplier * trend_factor)
        daily_target = max(daily_target, 5)  # Minimum
        daily_targets.append(daily_target)
    
    # Ensure monthly target is met exactly
    current_total = sum(daily_targets)
    diff = monthly_target - current_total
    
    while diff != 0:
        for i in range(len(daily_targets)):
            if diff == 0:
                break
            if diff > 0:
                # Favor weekend days for increases
                weekend_boost = 2 if date(year, month, i+1).weekday() >= 4 else 1
                if random.randint(1, 3) <= weekend_boost:
                    daily_targets[i] += 1
                    diff -= 1
            elif diff < 0 and daily_targets[i] > 5:
                # Favor Monday for decreases
                monday_penalty = 2 if date(year, month, i+1).weekday() == 0 else 1
                if random.randint(1, 3) <= monday_penalty:
                    daily_targets[i] -= 1
                    diff += 1
    
    return daily_targets

# ---- MAIN EXECUTION ----
def main():
    print(f"🚀 Generate FIXED synthetic data for ML Chaos Recommendation System")
    print(f"📅 Period: {START} to {END} (exactly 36 months)")
    print(f"🎯 Target: {TARGET_TOTAL:,} preferences (MUST BE EXACT)")
    print(f"📊 Feedback Rate: {FEEDBACK_SESSION_RATE*100:.0f}% sessions, {POSITIVE_FEEDBACK_RATE*100:.0f}% positive")
    
    if not init_flask_app():
        print("❌ Failed to initialize Flask app")
        return

    # Generate and fix monthly distribution
    monthly_targets, monthly_data = generate_organic_monthly_distribution()
    
    # Calculate exact months needed
    def _count_months(start_date, end_date):
        return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    need = _count_months(START, END)
    print(f"🔢 Need exactly {need} months of data")
    
    # Ensure we have exactly the right number of months
    if len(monthly_targets) < need:
        # Extend by repeating pattern
        pattern = monthly_targets[-12:]  # Last year pattern
        while len(monthly_targets) < need:
            monthly_targets.extend(pattern)
            monthly_data.extend(monthly_data[-12:])  # Extend metadata too
    
    # Trim to exact length
    monthly_targets = monthly_targets[:need]
    monthly_data = monthly_data[:need]
    
    # FIXED: Ensure exact total with proper algorithm
    monthly_targets = ensure_exact_total(monthly_targets, TARGET_TOTAL)
    
    # Verify total
    actual_total = sum(monthly_targets)
    print(f"✅ Verified total: {actual_total:,} (target: {TARGET_TOTAL:,})")
    
    if actual_total != TARGET_TOTAL:
        print(f"❌ CRITICAL: Total mismatch! Cannot proceed.")
        return
    
    # Print yearly distribution for verification
    year_totals = {}
    for i, target in enumerate(monthly_targets):
        year = 1 + (i // 12)
        if year not in year_totals:
            year_totals[year] = 0
        year_totals[year] += target
    
    print(f"\n📈 YEARLY DISTRIBUTION:")
    for year, total in year_totals.items():
        percentage = (total / TARGET_TOTAL) * 100
        print(f"   Year {year}: {total:,} ({percentage:.1f}%)")
    
    conn = get_conn()
    name_to_id = load_menu_map(conn)
    if not name_to_id:
        print("❌ No available menu items")
        return

    seq = IdSequencer()

    try:
        with conn.cursor() as cur:
            cur.execute("SET time_zone = '+07:00'")
            
            total_generated = 0
            total_sessions = 0
            total_feedback_completed = 0
            total_positive_feedback = 0
            
            current_date = START
            month_idx = 0
            
            while current_date <= END and month_idx < len(monthly_targets):
                # Get monthly target and business context
                monthly_target = monthly_targets[month_idx]
                business_context = monthly_data[month_idx] if month_idx < len(monthly_data) else {"reason": "Generated pattern"}
                year, month = current_date.year, current_date.month
                
                # Distribute to daily targets with organic patterns
                daily_targets = distribute_monthly_to_daily_organic(monthly_target, year, month)
                
                print(f"\n📅 Processing {year}-{month:02d} (Target: {monthly_target:,})")
                print(f"   📈 Business Reason: {business_context.get('reason', 'N/A')}")
                
                monthly_generated = 0
                monthly_sessions = 0
                monthly_feedback_completed = 0
                monthly_positive_feedback = 0
                
                # Process each day with organic patterns
                days_in_month = calendar.monthrange(year, month)[1]
                for day_idx in range(days_in_month):
                    d = date(year, month, day_idx + 1)
                    if d < START or d > END:
                        continue
                    
                    seq.seed_day_from_db(cur, d)
                    target_today = daily_targets[day_idx]
                    
                    if target_today == 0:
                        continue
                    
                    # Calculate sessions for today with realistic patterns
                    # Business intelligence: 2-5 attempts per session average
                    # Higher satisfaction = fewer attempts needed
                    satisfaction_factor = min(1.0, (month_idx + 1) / 12)  # Improves over time
                    base_attempts = 4.0 - (satisfaction_factor * 1.5)  # 4.0 -> 2.5 over time
                    avg_attempts = max(2.0, base_attempts + random.uniform(-0.5, 0.5))
                    
                    n_sessions = max(1, round(target_today / avg_attempts))
                    
                    # Distribute attempts across sessions with realistic variance
                    attempts_left = target_today
                    attempts_per_session = []
                    
                    for si in range(n_sessions):
                        if si == n_sessions - 1:  # Last session gets remainder
                            attempts_per_session.append(max(1, attempts_left))
                        else:
                            # Normal distribution around average, but bounded
                            max_attempts = min(6, attempts_left - (n_sessions - si - 1))
                            min_attempts = 1
                            
                            # Use normal distribution for more realistic patterns
                            attempt_count = int(np.clip(
                                np.random.normal(avg_attempts, 1.2), 
                                min_attempts, max_attempts
                            ))
                            
                            attempts_per_session.append(attempt_count)
                            attempts_left -= attempt_count
                    
                    # Create session plans with realistic timing patterns
                    session_plans = []
                    for attempt_count in attempts_per_session:
                        if attempt_count <= 0:
                            continue
                        
                        # Generate session times with business-hour awareness
                        times = []
                        for _ in range(attempt_count):
                            # Ensure each attempt is at least 2-15 minutes apart
                            if times:
                                last_time = times[-1]
                                min_gap = timedelta(minutes=random.randint(2, 15))
                                earliest_next = last_time + min_gap
                                
                                # Try to generate time after gap, but within business hours
                                for _ in range(10):  # Max 10 attempts to find valid time
                                    candidate = random_waktu_input(d)
                                    if candidate >= earliest_next:
                                        times.append(candidate)
                                        break
                                else:
                                    # Fallback: just add the gap
                                    times.append(earliest_next)
                            else:
                                times.append(random_waktu_input(d))
                        
                        times.sort()  # Ensure chronological order
                        
                        session_plans.append({
                            "times": times,
                            "first_ts": times[0],
                            "last_ts": times[-1],
                            "n_attempts": len(times),
                            "customer_name": fake.name(),
                        })
                    
                    # Sort sessions by start time, assign sequential IDs
                    session_plans.sort(key=lambda x: x["first_ts"])
                    for sp in session_plans:
                        sp["session_id"] = seq.next("SESS", d, cur=cur)
                    
                    # Create chronological events across all sessions
                    events = []
                    for sp in session_plans:
                        for qa, ts in enumerate(sp["times"], start=1):
                            events.append({
                                "ts": ts,
                                "session_id": sp["session_id"],
                                "quiz_attempt": qa,
                                "customer_name": sp["customer_name"],
                                "prefs": sample_preferences_seasonally_aware(month),
                            })
                    
                    # Sort events chronologically for proper PREF/REC ID sequencing
                    events.sort(key=lambda e: e["ts"])
                    
                    # Determine which sessions will give feedback (70% business rule)
                    will_fb = {sp["session_id"]: session_will_give_feedback() for sp in session_plans}
                    tot_attempts = {sp["session_id"]: sp["n_attempts"] for sp in session_plans}
                    last_status_by_session = {}
                    
                    # Process events chronologically
                    daily_generated = 0
                    daily_feedback_completed = 0
                    daily_positive_feedback = 0
                    
                    for e in events:
                        session_id = e["session_id"]
                        quiz_attempt = e["quiz_attempt"]
                        ts = e["ts"]
                        prefs = e["prefs"]
                        customer = e["customer_name"]
                        
                        # Generate recommendations using trained system with retry mechanism
                        recs = None
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                recs = compute_recommendations_accurate(prefs, session_id, k=TOPK)
                                if recs:
                                    break
                            except Exception as rec_error:
                                if attempt == max_retries - 1:
                                    print(f"⚠️ Recommendation failed after {max_retries} attempts for {ts.date()}: {rec_error}")
                                    continue
                        
                        if not recs:
                            print(f"⚠️ No recommendations generated for {ts.date()} attempt {quiz_attempt}")
                            continue
                        
                        # Insert preference
                        pref_id = seq.next("PREF", ts.date(), cur=cur)
                        insert_preference(cur, pref_id, customer, ts, prefs, session_id, quiz_attempt, "pending")
                        
                        # Insert recommendations
                        recs_inserted = insert_recommendations(cur, name_to_id, recs, pref_id, session_id, quiz_attempt, ts, seq, cur_for_seq=cur)
                        
                        if recs_inserted == 0:
                            print(f"⚠️ No recommendations inserted for {ts.date()} - skipping")
                            continue
                        
                        # Apply feedback logic (only to final attempt of feedback-giving sessions)
                        if quiz_attempt == tot_attempts[session_id] and will_fb[session_id]:
                            will_be_positive = feedback_will_be_positive()
                            last_status = apply_feedback_realistic(cur, pref_id, session_id, quiz_attempt, ts, will_be_positive)
                            daily_feedback_completed += 1
                            if will_be_positive:
                                daily_positive_feedback += 1
                        else:
                            last_status = 'pending'
                        
                        last_status_by_session[session_id] = last_status
                        daily_generated += 1
                    
                    # Update session records with final status
                    for sp in session_plans:
                        sess_status = 'completed' if last_status_by_session.get(sp["session_id"]) == 'completed' else 'active'
                        ensure_session(cur, sp["session_id"], sp["first_ts"], sp["last_ts"], sp["n_attempts"], sess_status)
                    
                    monthly_generated += daily_generated
                    monthly_sessions += len(session_plans)
                    monthly_feedback_completed += daily_feedback_completed
                    monthly_positive_feedback += daily_positive_feedback
                    
                    # Commit every 1000 entries for progress tracking and safety
                    if total_generated > 0 and total_generated % 1000 == 0 and not DRY_RUN:
                        conn.commit()
                        print(f"  ⏳ Progress: {total_generated:,} entries processed...")
                
                total_generated += monthly_generated
                total_sessions += monthly_sessions
                total_feedback_completed += monthly_feedback_completed
                total_positive_feedback += monthly_positive_feedback
                
                # Monthly summary with business insights
                feedback_rate = (monthly_feedback_completed / monthly_sessions * 100) if monthly_sessions > 0 else 0
                positive_rate = (monthly_positive_feedback / monthly_feedback_completed * 100) if monthly_feedback_completed > 0 else 0
                
                print(f"📊 {year}-{month:02d} Complete: {monthly_generated:,} preferences, {monthly_sessions:,} sessions")
                print(f"   📈 Feedback: {monthly_feedback_completed} sessions ({feedback_rate:.1f}%), {positive_rate:.1f}% positive")
                print(f"   🎯 Target vs Actual: {monthly_target:,} vs {monthly_generated:,} ({((monthly_generated/monthly_target)-1)*100:+.1f}%)")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                month_idx += 1
                
                # Monthly commit for safety
                if not DRY_RUN:
                    conn.commit()
            
            # Final commit or rollback with comprehensive summary
            if DRY_RUN:
                conn.rollback()
                print(f"\n📝 [DRY RUN] Would generate:")
                print(f"   📊 Total: {total_generated:,} preferences, {total_sessions:,} sessions")
                print(f"   📈 Feedback: {total_feedback_completed:,} completed sessions")
                print(f"   ⭐ Positive: {total_positive_feedback:,} positive feedback")
            else:
                conn.commit()
                print(f"\n✅ [GENERATION COMPLETED SUCCESSFULLY]")
                print(f"   📊 Generated: {total_generated:,} preferences, {total_sessions:,} sessions")
                print(f"   🎯 Target Achievement: {(total_generated/TARGET_TOTAL)*100:.1f}% of {TARGET_TOTAL:,}")
                print(f"   📈 Business Metrics:")
                print(f"      • Average monthly: {(total_generated/36):,.0f} preferences/month")
                print(f"      • Session conversion: {(total_generated/total_sessions):,.1f} preferences/session")
                print(f"      • Feedback participation: {(total_feedback_completed/total_sessions)*100:.1f}% of sessions")
                print(f"      • Satisfaction rate: {(total_positive_feedback/total_feedback_completed)*100:.1f}% positive")
                
                # Comprehensive verification queries
                print(f"\n🔍 RUNNING VERIFICATION QUERIES...")
                
                # Total count verification
                cur.execute("""
                    SELECT COUNT(*) as total_prefs
                    FROM preferences 
                    WHERE timestamp >= %s AND timestamp <= %s
                """, (START, END))
                db_total = cur.fetchone()['total_prefs']
                print(f"   📊 Database Total: {db_total:,} (Expected: {TARGET_TOTAL:,})")
                
                if db_total != TARGET_TOTAL:
                    print(f"   ❌ MISMATCH DETECTED! Difference: {db_total - TARGET_TOTAL:+,}")
                else:
                    print(f"   ✅ PERFECT MATCH!")
                
                # Yearly distribution verification
                cur.execute("""
                    SELECT 
                        CASE 
                            WHEN YEAR(timestamp) = 2025 THEN 1
                            WHEN YEAR(timestamp) = 2026 THEN 2  
                            WHEN YEAR(timestamp) = 2027 THEN 3
                            WHEN YEAR(timestamp) = 2028 THEN 3
                            ELSE 0
                        END as year_group,
                        COUNT(*) as year_total
                    FROM preferences
                    WHERE timestamp >= %s AND timestamp <= %s
                    GROUP BY year_group
                    ORDER BY year_group
                """, (START, END))
                year_results = cur.fetchall()
                
                print(f"   📈 Yearly Distribution:")
                for row in year_results:
                    year_group = row['year_group'] 
                    year_total = row['year_total']
                    percentage = (year_total / TARGET_TOTAL) * 100
                    print(f"      Year {year_group}: {year_total:,} ({percentage:.1f}%)")
                
                # Feedback verification
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT s.session_id) as total_sessions,
                        COUNT(DISTINCT CASE WHEN r.feedback IS NOT NULL THEN s.session_id END) as feedback_sessions,
                        ROUND(100.0 * COUNT(DISTINCT CASE WHEN r.feedback IS NOT NULL THEN s.session_id END) / COUNT(DISTINCT s.session_id), 1) as feedback_rate
                    FROM user_sessions s
                    LEFT JOIN recommendations r ON r.session_id = s.session_id
                    WHERE s.created_at >= %s AND s.created_at <= %s
                """, (START, END))
                feedback_stats = cur.fetchone()
                
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as positive_feedback,
                        ROUND(100.0 * SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as positive_rate
                    FROM recommendations
                    WHERE feedback IS NOT NULL 
                      AND generated_at >= %s AND generated_at <= %s
                """, (START, END))
                satisfaction_stats = cur.fetchone()
                
                print(f"   📈 Feedback Analytics:")
                print(f"      • Sessions with feedback: {feedback_stats['feedback_rate']}%")
                print(f"      • Positive feedback rate: {satisfaction_stats['positive_rate']}%")
                
                # Monthly distribution sample (first 6 months)
                cur.execute("""
                    SELECT 
                        CONCAT(YEAR(timestamp), '-', LPAD(MONTH(timestamp), 2, '0')) AS month,
                        COUNT(*) AS monthly_count
                    FROM preferences
                    WHERE timestamp >= %s AND timestamp <= %s
                    GROUP BY YEAR(timestamp), MONTH(timestamp)
                    ORDER BY YEAR(timestamp), MONTH(timestamp)
                """, (START, END))
                monthly_sample = cur.fetchall()
                
                print(f"   📅 Monthly Distribution Sample (first 6 months):")
                for row in monthly_sample:
                    print(f"      {row['month']}: {row['monthly_count']:,} preferences")
                
                print(f"\n🚀 DATA GENERATION COMPLETE!")
                print(f"📊 Ready for Power BI dashboard with:")
                print(f"   • Natural organic growth patterns")
                print(f"   • Seasonal beverage preference variations")  
                print(f"   • Realistic feedback and satisfaction trends")
                print(f"   • Business hour usage analytics")
                print(f"   • Customer journey insights")
                
                print(f"\n🎯 Run the SQL validation queries to confirm all requirements are met!")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()