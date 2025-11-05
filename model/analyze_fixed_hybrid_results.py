# analyze_fixed_hybrid_results.py
import argparse
import os
import pickle
from collections import defaultdict

# ==== Default ke saved_models di ROOT project ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "saved_models"))
MODEL_PREFIX = ""  # kosongkan agar semua .pkl dipertimbangkan

# ---------- Util ----------
def resolve_model_dir(arg_dir: str | None) -> str:
    """Gunakan argumen jika ada; kalau tidak, pakai saved_models di ROOT."""
    return os.path.abspath(arg_dir) if arg_dir else DEFAULT_MODEL_DIR

def find_candidates(model_dir, prefix=MODEL_PREFIX):
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        return []
    files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return [os.path.join(model_dir, f) for f in files]

def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def bar(score, mx, length=30):
    if mx <= 0:
        return "░" * length
    filled = int(round((score / mx) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)

def fmt_w(v):
    return f"{v:+0.4f}"

def ensure_keys(data):
    if not isinstance(data, dict):
        raise ValueError("Model payload bukan dict.")
    if "question_importance" not in data:
        raise KeyError("Tidak ada kunci 'question_importance' di model.")
    if "user_vector_weights" not in data:
        raise KeyError("Tidak ada kunci 'user_vector_weights' di model.")
    return data["question_importance"], data["user_vector_weights"], data.get("feature_cols", [])

# ---------- Cetak ----------
def print_question_importance(q_imp):
    print("📊 QUESTION IMPORTANCE RANKING (NEUTRAL MODE PREFERRED)")
    print("=" * 90)
    if not q_imp:
        print("❌ question_importance kosong.")
        return None, None
    mx = max(q_imp.values())
    ordered = sorted(q_imp.items(), key=lambda x: x[1], reverse=True)
    for i, (q, s) in enumerate(ordered, 1):
        print(f"{i}. {q.upper():<12}: {s:0.4f} |{bar(s, mx, 30)}|")
    max_q = max(q_imp, key=q_imp.get)
    min_q = min(q_imp, key=q_imp.get)
    print(f"\n💡 Insight: {max_q.upper()} terbesar, {min_q.upper()} terkecil.")
    print("=" * 90)
    return ordered, (max_q, min_q)

def print_feature_weights(uvec, q_imp):
    print("🔍 FEATURE WEIGHTS DETAIL PER OPSI JAWABAN")
    print("=" * 90)

    order = sorted(uvec.keys(), key=lambda q: -q_imp.get(q, 0.0))
    for q in order:
        print(f"\n{'='*20} {q.upper()} {'='*20}")
        print(f"Importance Weight: {q_imp.get(q, 0.0):0.4f}")
        print("-" * 70)

        answers = uvec.get(q, {})
        for ans in sorted(answers.keys()):
            weights = answers[ans] or {}
            items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
            plus = [kv for kv in items if kv[1] > 0]
            minus = [kv for kv in items if kv[1] < 0]

            print(f"\n📌 {ans.upper()}:")
            if plus:
                print("   🔺 MENINGKATKAN:")
                for feat, w in plus:
                    print(f"    ↑ {feat:<22}: {fmt_w(w):>8}")
            if minus:
                print("   🔻 MENURUNKAN:")
                for feat, w in minus:
                    print(f"    ↓ {feat:<22}: {fmt_w(w):>8}")

def print_summary(uvec):
    print("=" * 90)
    print("📈 SUMMARY STATISTICS")
    print("=" * 90)
    feat_counts = defaultdict(int)
    feat_abs_sum = defaultdict(float)

    for q in uvec:
        for a in uvec[q]:
            for f, w in (uvec[q][a] or {}).items():
                feat_counts[f] += 1
                feat_abs_sum[f] += abs(w)

    total_questions = len(uvec)
    print("📊 Model Overview:")
    print(f"   • Total Questions: {total_questions}")
    ranking = sorted(feat_counts.keys(), key=lambda f: -feat_abs_sum[f])
    for i, f in enumerate(ranking, 1):
        avg_abs = (feat_abs_sum[f] / feat_counts[f]) if feat_counts[f] else 0.0
        print(f"   {i}. {f:<22}: Used {feat_counts[f]:<2} times, Avg |weight|: {avg_abs:0.4f}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Analyze question ranking & feature weights from saved model")
    ap.add_argument("--model", help="Path ke file model .pkl")
    ap.add_argument("--model-dir", default=None, help="Folder tempat model disimpan (default: ROOT/saved_models)")
    ap.add_argument("--list", action="store_true", help="List kandidat model lalu keluar")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)

    if args.list:
        cands = find_candidates(model_dir)
        print(f"📂 Candidate models in: {model_dir}")
        if not cands:
            print(" - (none)")
            return
        for p in cands:
            print(" -", os.path.basename(p))
        return

    model_path = None
    if args.model and os.path.isfile(args.model):
        model_path = os.path.abspath(args.model)
        model_dir = os.path.dirname(model_path)
    else:
        cands = find_candidates(model_dir)
        if cands:
            model_path = cands[0]

    if not model_path:
        print("❌ Tidak menemukan file model.")
        print(f"   Dicari di: {model_dir}")
        print("   Tips: retrain model dari app, atau sebutkan --model path spesifik.")
        return

    print("=" * 90)
    print("📦 MODEL LOADED:", os.path.basename(model_path))
    print("📁 From directory:", model_dir)
    print("=" * 90)

    try:
        data = load_model(model_path)
        q_imp, uvec, _ = ensure_keys(data)
    except Exception as e:
        print("❌ Gagal memuat/validasi model:", e)
        return

    print_question_importance(q_imp)
    print_feature_weights(uvec, q_imp)
    print_summary(uvec)

if __name__ == "__main__":
    main()
