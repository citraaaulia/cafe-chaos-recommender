"""
inspect_features_fixed.py
--------------------------------------------------------
Inspeksi hasil normalisasi fitur dengan Flask app context
--------------------------------------------------------
"""

import os
import sys
import pandas as pd
import traceback

# Setup path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Definisi kolom untuk laporan
COLUMNS_FOR_REPORT = [
    'nama_minuman',
    'rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih', 'rasa_netral',
    'kafein_score', 'tingkat_kafein',
    'tekstur_LIGHT', 'tekstur_CREAMY', 'tekstur_BUBBLY', 'tekstur_HEAVY',
    'sweetness_score', 'carbonated_score',
    'harga', 'original_harga'
]

# Output files
CSV_FULL = os.path.join(ROOT_DIR, "hasil_normalisasi_fitur_lengkap.csv")
CSV_SAMPLE = os.path.join(ROOT_DIR, "lampiran_Tabel_4_1_2_A_sample.csv")
HTML_PREVIEW = os.path.join(ROOT_DIR, "preview_normalisasi_fitur.html")


def create_flask_app():
    """Buat Flask app untuk mendapatkan context"""
    try:
        from app import create_app
        print("Membuat Flask application context...")
        app = create_app()
        return app
    except Exception as e:
        print(f"Error membuat Flask app: {e}")
        raise


def import_recommender_system():
    """Import EnhancedHybridRecommendationSystem"""
    try:
        from model.recommender import EnhancedHybridRecommendationSystem
        print("Import dari model.recommender berhasil")
        return EnhancedHybridRecommendationSystem
    except ImportError:
        try:
            from model.recommender import EnhancedHybridRecommendationSystem
            print("Import dari recommender berhasil")
            return EnhancedHybridRecommendationSystem
        except ImportError as e:
            raise ImportError(
                f"Gagal mengimpor EnhancedHybridRecommendationSystem.\n"
                f"Detail error: {e}"
            )


def get_recommender_with_context(app, RecommenderClass):
    """
    Dapatkan instance recommender DALAM Flask app context
    """
    with app.app_context():
        try:
            print("Membuat instance EnhancedHybridRecommendationSystem...")
            recommender = RecommenderClass()
            
            if not hasattr(recommender, 'menu_df'):
                raise AttributeError("Instance tidak memiliki atribut 'menu_df'")
            
            if recommender.menu_df is None or recommender.menu_df.empty:
                raise ValueError("menu_df kosong atau None")
            
            print(f"Berhasil memuat {len(recommender.menu_df)} menu items")
            
            # Copy DataFrame agar bisa digunakan di luar context
            menu_df_copy = recommender.menu_df.copy()
            return menu_df_copy
            
        except Exception as e:
            print(f"Error saat membuat instance: {e}")
            traceback.print_exc()
            raise


def validate_and_prepare_columns(df, requested_cols):
    """Validasi kolom yang tersedia"""
    available_cols = []
    missing_cols = []
    
    for col in requested_cols:
        if col in df.columns:
            available_cols.append(col)
        else:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\nKolom tidak ditemukan: {missing_cols}")
    
    print(f"\nKolom tersedia: {len(available_cols)}/{len(requested_cols)}")
    return available_cols


def display_texture_summary(df):
    """Tampilkan ringkasan distribusi tekstur"""
    print("\n== Ringkasan Distribusi Tekstur ==")
    texture_cols = [c for c in df.columns if c.startswith('tekstur_')]
    
    if not texture_cols:
        print("Tidak ada kolom tekstur ditemukan")
        return
    
    for col in texture_cols:
        count = df[col].sum()
        pct = (count / len(df)) * 100
        print(f"{col}: {int(count)} items ({pct:.1f}%)")


def display_taste_summary(df):
    """Tampilkan ringkasan fitur rasa"""
    print("\n== Ringkasan Fitur Rasa (Normalized 0-1) ==")
    rasa_cols = ['rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih', 'rasa_netral']
    
    for col in rasa_cols:
        if col in df.columns:
            print(f"{col:15s}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")


def create_feature_summary_table(df):
    """Buat tabel ringkasan statistik fitur"""
    summary_data = []
    
    # Fitur rasa
    rasa_cols = ['rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih', 'rasa_netral']
    for col in rasa_cols:
        if col in df.columns:
            summary_data.append({
                'Kategori': 'Rasa',
                'Fitur': col,
                'Min': f"{df[col].min():.3f}",
                'Max': f"{df[col].max():.3f}",
                'Mean': f"{df[col].mean():.3f}",
                'Std': f"{df[col].std():.3f}"
            })
    
    # Fitur tekstur
    texture_cols = [c for c in df.columns if c.startswith('tekstur_')]
    for col in texture_cols:
        count = df[col].sum()
        pct = (count / len(df)) * 100
        summary_data.append({
            'Kategori': 'Tekstur',
            'Fitur': col,
            'Min': '0',
            'Max': '1',
            'Count': f"{int(count)}",
            'Percentage': f"{pct:.1f}%"
        })
    
    # Fitur lainnya
    other_features = ['kafein_score', 'sweetness_score', 'carbonated_score', 'harga']
    for col in other_features:
        if col in df.columns:
            summary_data.append({
                'Kategori': 'Lainnya',
                'Fitur': col,
                'Min': f"{df[col].min():.3f}",
                'Max': f"{df[col].max():.3f}",
                'Mean': f"{df[col].mean():.3f}",
                'Std': f"{df[col].std():.3f}"
            })
    
    return pd.DataFrame(summary_data)


def main():
    try:
        print("=" * 70)
        print("INSPEKSI HASIL NORMALISASI & REKAYASA FITUR")
        print("=" * 70)
        
        # 1. Buat Flask app untuk context
        app = create_flask_app()
        
        # 2. Import recommender class
        RecommenderClass = import_recommender_system()
        
        # 3. Dapatkan data DALAM app context
        menu_df = get_recommender_with_context(app, RecommenderClass)
        
        # 4. Dari sini tidak perlu context lagi karena sudah copy DataFrame
        print(f"\nJumlah total menu: {len(menu_df)}")
        print(f"Jumlah fitur: {len(menu_df.columns)}")
        
        # 5. Validasi kolom
        available_cols = validate_and_prepare_columns(menu_df, COLUMNS_FOR_REPORT)
        
        if not available_cols:
            raise ValueError("Tidak ada kolom yang tersedia untuk ditampilkan")
        
        # 6. Tampilkan ringkasan
        display_taste_summary(menu_df)
        display_texture_summary(menu_df)
        
        # 7. Tampilkan 10 baris pertama
        print("\n" + "=" * 70)
        print("PRATINJAU 10 BARIS PERTAMA")
        print("=" * 70)
        preview_df = menu_df[available_cols].head(10)
        print(preview_df.to_string(index=False))
        
        # 8. Export full dataset
        print(f"\n{'=' * 70}")
        print("EKSPOR DATA")
        print("=" * 70)
        menu_df[available_cols].to_csv(CSV_FULL, index=False)
        print(f"1. Dataset lengkap: {CSV_FULL}")
        
        # 9. Export sample untuk lampiran
        preview_df.to_csv(CSV_SAMPLE, index=False)
        print(f"2. Sampel lampiran: {CSV_SAMPLE}")
        
        # 10. Buat tabel ringkasan statistik
        summary_df = create_feature_summary_table(menu_df)
        summary_csv = os.path.join(ROOT_DIR, "ringkasan_statistik_fitur.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"3. Ringkasan statistik: {summary_csv}")
        
        print("\n== Ringkasan Statistik Fitur ==")
        print(summary_df.to_string(index=False))
        
        # 11. Export HTML preview
        try:
            html_content = f"""
            <html>
            <head>
                <title>Preview Normalisasi Fitur</title>
                <meta charset="UTF-8">
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 20px; 
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0;
                    }}
                    th, td {{ 
                        border: 1px solid #ddd; 
                        padding: 8px; 
                        text-align: left; 
                        font-size: 12px;
                    }}
                    th {{ 
                        background-color: #4CAF50; 
                        color: white;
                        font-weight: bold;
                    }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    h1 {{ 
                        color: #333; 
                        border-bottom: 3px solid #4CAF50;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #555;
                        margin-top: 30px;
                    }}
                    .info-box {{
                        background-color: #e8f5e9;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Preview Hasil Normalisasi & Rekayasa Fitur</h1>
                    <div class="info-box">
                        <p><strong>Total menu items:</strong> {len(menu_df)}</p>
                        <p><strong>Total fitur:</strong> {len(menu_df.columns)}</p>
                        <p><strong>Fitur yang ditampilkan:</strong> {len(available_cols)}</p>
                    </div>
                    
                    <h2>20 Baris Pertama Dataset</h2>
                    {menu_df[available_cols].head(20).to_html(index=False)}
                    
                    <h2>Ringkasan Statistik Fitur</h2>
                    {summary_df.to_html(index=False)}
                </div>
            </body>
            </html>
            """
            with open(HTML_PREVIEW, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"4. Preview HTML: {HTML_PREVIEW}")
        except Exception as e:
            print(f"Gagal membuat HTML preview: {e}")
        
        # 12. Summary
        print("\n" + "=" * 70)
        print("SELESAI")
        print("=" * 70)
        print("\nFile yang dihasilkan:")
        print(f"  - {os.path.basename(CSV_FULL)} (untuk referensi lengkap)")
        print(f"  - {os.path.basename(CSV_SAMPLE)} (untuk Lampiran Tabel 4.1.2-A)")
        print(f"  - {os.path.basename(summary_csv)} (untuk analisis statistik)")
        print(f"  - {os.path.basename(HTML_PREVIEW)} (untuk preview visual)")
        print("\nProses inspeksi berhasil!")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("TERJADI ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nTraceback lengkap:")
        traceback.print_exc()
        print("\nTips perbaikan:")
        print("1. Pastikan Flask app (app.py) dengan create_app() tersedia")
        print("2. Pastikan database MySQL/MariaDB sedang berjalan")
        print("3. Pastikan kredensial database di models/database.py benar")
        print("4. Pastikan tabel menu_items memiliki data")
        sys.exit(1)


if __name__ == "__main__":
    main()