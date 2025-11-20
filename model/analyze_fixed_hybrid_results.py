# analyze_fixed_hybrid_results.py
import argparse
import os
import pickle
from collections import defaultdict
from statistics import mean

# ==== Default ke saved_models di ROOT project ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "saved_models"))
MODEL_PREFIX = ""  # bisa diisi "enhanced_hybrid_fixed_v3_" kalau mau dibatasi


# ---------- Util umum ----------

def resolve_model_dir(arg_dir: str | None) -> str:
    """Gunakan argumen jika ada; kalau tidak, pakai saved_models di ROOT."""
    return os.path.abspath(arg_dir) if arg_dir else DEFAULT_MODEL_DIR


def list_model_files(model_dir: str, prefix: str = MODEL_PREFIX) -> list[str]:
    """Mengembalikan daftar file .pkl yang ada di model_dir (opsional filter prefix), diurutkan dari yang terbaru."""
    if not os.path.isdir(model_dir):
        return []

    files = []
    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(".pkl"):
            continue
        if prefix and not fname.startswith(prefix):
            continue
        full_path = os.path.join(model_dir, fname)
        if os.path.isfile(full_path):
            files.append(full_path)

    # urutkan dari yang terbaru berdasarkan mtime
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def pick_model_path(model_dir: str, explicit: str | None = None) -> str:
    """Pilih path model:
    - jika explicit diisi, pakai itu (absolute-kan jika perlu)
    - kalau tidak, pakai file terbaru di model_dir
    """
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model yang diminta tidak ditemukan: {path}")
        return path

    files = list_model_files(model_dir)
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file .pkl yang ditemukan di direktori model: {model_dir}"
        )
    return files[0]


def load_model(path: str) -> dict:
    """Muat artefak model dari file pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Isi file model bukan dict seperti yang diharapkan.")
    return data


def ensure_keys(data: dict):
    """Pastikan kunci-kunci utama ada, dan kembalikan dalam bentuk terstruktur."""
    q_imp = data.get("question_importance") or {}
    uvec = data.get("user_vector_weights") or {}
    perf = data.get("model_performance") or {}
    feature_cols = data.get("feature_cols") or []
    metadata = data.get("metadata") or {}

    if not q_imp:
        raise ValueError("question_importance kosong atau tidak ditemukan dalam artefak model.")
    if not uvec:
        raise ValueError("user_vector_weights kosong atau tidak ditemukan dalam artefak model.")

    return q_imp, uvec, perf, feature_cols, metadata


def print_header(model_path: str, feature_cols: list[str], metadata: dict):
    print("=" * 90)
    print("📦 ANALISIS MODEL ENHANCED HYBRID – FIXED V3")
    print("=" * 90)
    print(f"Model file : {model_path}")
    print(f"Jumlah fitur aktif : {len(feature_cols)}")
    if feature_cols:
        # tampilkan beberapa fitur pertama saja
        preview = ", ".join(feature_cols[:10])
        if len(feature_cols) > 10:
            preview += ", ..."
        print(f"Contoh fitur   : {preview}")

    if metadata:
        print("\n📝 Metadata singkat:")
        algo = metadata.get("algorithm", "N/A")
        version = metadata.get("version", "N/A")
        notes = metadata.get("notes", [])
        print(f"  - Algorithm : {algo}")
        print(f"  - Version   : {version}")
        if notes:
            print("  - Notes     :")
            for n in notes:
                print(f"      • {n}")
    print("=" * 90)
    print()


# ---------- Analisis Question Importance ----------

def print_question_importance(q_imp: dict[str, float]):
    print("📊 QUESTION IMPORTANCE (RANKING PERTANYAAN)")
    print("=" * 90)

    if not q_imp:
        print("Tidak ada data question_importance.")
        print()
        return

    # Urutkan dari yang paling penting
    ordered = sorted(q_imp.items(), key=lambda x: -x[1])
    max_val = max(abs(v) for _, v in ordered) or 1.0

    print(f"{'Rank':<6}{'Question':<20}{'Weight':>10}  Bar")
    print("-" * 90)
    for i, (q, w) in enumerate(ordered, start=1):
        # bar sederhana proporsional terhadap bobot
        bar_len = int((abs(w) / max_val) * 30)
        bar = "█" * bar_len
        print(f"{i:<6}{q:<20}{w:>10.4f}  {bar}")
    print()


# ---------- Analisis Bobot Fitur ----------

def print_feature_weights(uvec: dict, q_imp: dict[str, float], top_n: int = 10):
    print("🔍 FEATURE WEIGHTS DETAIL PER OPSI JAWABAN")
    print("=" * 90)

    if not uvec:
        print("Tidak ada data user_vector_weights.")
        print()
        return

    # Urut pertanyaan berdasar importance
    question_order = sorted(uvec.keys(), key=lambda q: -q_imp.get(q, 0.0))

    for q in question_order:
        print(f"\n{'='*20} {q.upper()} {'='*20}")
        print(f"Importance Weight: {q_imp.get(q, 0.0):0.4f}")
        print("-" * 70)

        answers = uvec.get(q, {})
        if not answers:
            print("  (Tidak ada bobot untuk pertanyaan ini)")
            continue

        for ans in sorted(answers.keys()):
            weights = answers[ans] or {}
            if not weights:
                print(f"\n  Jawaban: {ans}")
                print("    (tidak ada fitur aktif)")
                continue

            # urutkan fitur berdasarkan |weight| terbesar
            ordered_feats = sorted(weights.items(),
                                   key=lambda x: -abs(x[1]))
            print(f"\n  Jawaban: {ans}")
            print(f"  {'Fitur':<30}{'Bobot':>10}")
            print("  " + "-" * 42)
            for feat, w in ordered_feats[:top_n]:
                print(f"  {feat:<30}{w:>10.4f}")

    print()


# ---------- Analisis Model Performance (R², MSE, dst.) ----------

def print_model_performance(perf: dict, q_imp: dict[str, float]):
    print("📈 MODEL PERFORMANCE (R², MSE, N_SAMPLES, CORRELATION)")
    print("=" * 90)

    if not perf:
        print("Tidak ada kunci 'model_performance' di artefak model.")
        print("Kemungkinan model dilatih dengan versi lama yang belum menyimpan metrik ini.")
        print()
        return

    # Flatten untuk ringkasan global
    all_r2 = []
    all_mse = []
    all_corr = []
    n_total_samples = 0

    # Urut pertanyaan berdasarkan importance
    question_order = sorted(perf.keys(), key=lambda q: -q_imp.get(q, 0.0))

    for q in question_order:
        answers = perf.get(q, {})
        if not answers:
            continue

        r2_list = []
        mse_list = []
        corr_list = []
        n_list = []
        mean_co_list = []

        for ans, metrics in answers.items():
            r2_val = float(metrics.get("r2", 0.0))
            mse_val = float(metrics.get("mse", 0.0))
            n_val = int(metrics.get("n_samples", 0))
            co_mean = float(metrics.get("mean_cooccurrence", 0.0))
            corr = float(metrics.get("correlation_score", 0.5))

            r2_list.append(r2_val)
            mse_list.append(mse_val)
            corr_list.append(corr)
            n_list.append(n_val)
            mean_co_list.append(co_mean)

            all_r2.append(r2_val)
            all_mse.append(mse_val)
            all_corr.append(corr)
            n_total_samples += n_val

        if not r2_list:
            continue

        avg_r2 = mean(r2_list)
        avg_mse = mean(mse_list)
        avg_corr = mean(corr_list)
        avg_mean_co = mean(mean_co_list)
        total_n = sum(n_list)

        print(f"\n➡️ Pertanyaan: {q}")
        print(f"   Importance        : {q_imp.get(q, 0.0):0.4f}")
        print(f"   Jumlah kombinasi  : {len(answers)}")
        print(f"   Total sampel      : {total_n}")
        print(f"   R² rata-rata      : {avg_r2:0.4f}")
        print(f"   MSE rata-rata     : {avg_mse:0.4f}")
        print(f"   Mean co-occurrence: {avg_mean_co:0.4f}")
        print(f"   Corr score rata²  : {avg_corr:0.4f}")

        # Highlight kombinasi terbaik dan terburuk berdasarkan R²
        best_ans = max(answers.items(), key=lambda kv: float(kv[1].get("r2", 0.0)))
        worst_ans = min(answers.items(), key=lambda kv: float(kv[1].get("r2", 0.0)))
        print("   Kombinasi terbaik (R²):")
        print(f"     - Jawaban   : {best_ans[0]}")
        print(f"     - R²        : {best_ans[1].get('r2', 0.0):0.4f}")
        print(f"     - n_samples : {best_ans[1].get('n_samples', 0)}")
        print("   Kombinasi terendah (R²):")
        print(f"     - Jawaban   : {worst_ans[0]}")
        print(f"     - R²        : {worst_ans[1].get('r2', 0.0):0.4f}")
        print(f"     - n_samples : {worst_ans[1].get('n_samples', 0)}")

    # Ringkasan global
    if all_r2:
        print("\n📌 Ringkasan Global Model Performance")
        print("-" * 90)
        print(f"  R² rata-rata semua kombinasi        : {mean(all_r2):0.4f}")
        print(f"  MSE rata-rata semua kombinasi       : {mean(all_mse):0.4f}")
        print(f"  Corr score rata-rata semua kombinasi: {mean(all_corr):0.4f}")
        print(f"  Total sampel (agregat n_samples)    : {n_total_samples}")
    else:
        print("\nTidak ada kombinasi question–answer yang memiliki metrik performance.")

    print()


# ---------- Ringkasan Global Fitur ----------

def print_summary(uvec: dict):
    print("📚 SUMMARY STATISTICS – GLOBAL FEATURE IMPORTANCE")
    print("=" * 90)

    if not uvec:
        print("Tidak ada data user_vector_weights.")
        print()
        return

    feat_counts = defaultdict(int)
    feat_abs_sum = defaultdict(float)

    for q in uvec:
        for a in uvec[q]:
            for f, w in (uvec[q][a] or {}).items():
                feat_counts[f] += 1
                feat_abs_sum[f] += abs(w)

    if not feat_counts:
        print("Tidak ada fitur yang memiliki bobot.")
        print()
        return

    # hitung rata-rata |weight| per fitur
    feat_avg_abs = {
        f: (feat_abs_sum[f] / feat_counts[f]) for f in feat_counts
    }

    ordered = sorted(feat_avg_abs.items(), key=lambda x: -x[1])

    print(f"{'Fitur':<30}{'Avg |weight|':>12}{'Dipakai di #kombinasi':>24}")
    print("-" * 90)
    for f, avg_w in ordered[:30]:
        print(f"{f:<30}{avg_w:>12.4f}{feat_counts[f]:>24}")

    print("=" * 90)
    print()


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Analisis bobot dan kinerja model Enhanced Hybrid Fixed v3."
    )
    parser.add_argument(
        "-d", "--model-dir",
        help=f"Direktori model (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "-m", "--model",
        help="Path spesifik ke file model .pkl (jika tidak diisi, otomatis ambil file terbaru di model-dir)."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="Hanya menampilkan daftar file model yang tersedia lalu keluar."
    )
    parser.add_argument(
        "-k", "--top-n",
        type=int,
        default=10,
        help="Jumlah fitur teratas per jawaban yang ditampilkan (default: 10)."
    )

    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)

    # Mode list-only
    if args.list:
        print(f"📂 Daftar model di: {model_dir}")
        files = list_model_files(model_dir)
        if not files:
            print("  (Tidak ada file .pkl yang ditemukan)")
            return
        for i, path in enumerate(files, start=1):
            fname = os.path.basename(path)
            mtime = os.path.getmtime(path)
            print(f"{i:2d}. {fname}  (mtime={mtime})")
        return

    # Pilih model dan lakukan analisis penuh
    try:
        model_path = pick_model_path(model_dir, args.model)
    except Exception as e:
        print("❌ Gagal memilih model:", e)
        return

    try:
        data = load_model(model_path)
        q_imp, uvec, perf, feature_cols, metadata = ensure_keys(data)
    except Exception as e:
        print("❌ Gagal memuat/validasi model:", e)
        return

    print_header(model_path, feature_cols, metadata)
    print_question_importance(q_imp)
    print_feature_weights(uvec, q_imp, top_n=args.top_n)
    print_model_performance(perf, q_imp)
    print_summary(uvec)


if __name__ == "__main__":
    main()
