#!/usr/bin/env python3
"""
Quick Export Script untuk Export Training Data
Simplified version untuk export cepat tanpa interaksi user
"""

import os
import sys
from datetime import datetime

# Add current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from export_training_data import TrainingDataExporter

def quick_export_all():
    """Quick export semua data dengan konfigurasi default"""
    print("🚀 Quick Export - All Training Data")
    print("=" * 50)
    
    # Initialize exporter
    exporter = TrainingDataExporter(output_dir="quick_exported_data")
    
    # Initialize recommender
    if not exporter.initialize_recommender():
        print("❌ Failed to initialize recommender system")
        return False
    
    # Export all data dengan konfigurasi default
    results = exporter.export_all_training_data(
        n_users=1500,  # Bisa dikurangi untuk testing: 100-500
        random_seed=42
    )
    
    # Check hasil
    success_count = sum(1 for result in results.values() if result is not None)
    total_count = len(results)
    
    print(f"\n📊 Export Results: {success_count}/{total_count} successful")
    
    if success_count > 0:
        print(f"✅ Data exported to: {exporter.output_dir}")
        return True
    else:
        print("❌ No data was exported successfully")
        return False

def quick_export_sample():
    """Quick export dengan sample data yang lebih kecil untuk testing"""
    print("🚀 Quick Export - Sample Data (Small)")
    print("=" * 50)
    
    exporter = TrainingDataExporter(output_dir="sample_exported_data")
    
    if not exporter.initialize_recommender():
        print("❌ Failed to initialize recommender system")
        return False
    
    # Export dengan sample kecil untuk testing
    print("🔄 Exporting sample data (500 users)...")
    results = exporter.export_all_training_data(
        n_users=500,   # Sample kecil
        random_seed=42
    )
    
    success_count = sum(1 for result in results.values() if result is not None)
    print(f"\n📊 Sample Export Results: {success_count}/4 successful")
    
    return success_count > 0

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "sample":
            quick_export_sample()
        elif sys.argv[1] == "full":
            quick_export_all()
        else:
            print("Usage: python quick_export.py [sample|full]")
            print("  sample: Export 500 users for testing")
            print("  full:   Export 1500 users (default)")
    else:
        # Default: full export
        quick_export_all()