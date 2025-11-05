"""
Cleanup Script untuk menghapus model PCA lama dan memastikan Enhanced model
File: cleanup_old_models.py

Tujuan: Membersihkan model PCA lama yang mungkin masih terbaca oleh analyzer
dan memastikan Enhanced model yang benar digunakan.

Author: ML Chaos Team
Version: 1.0
"""

import os
import glob
import shutil
from datetime import datetime

class ModelCleanupTool:
    """Tool untuk membersihkan model lama dan organize model files"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        self.backup_dir = os.path.join(model_dir, "backup_old_models")
        
    def scan_models(self):
        """Scan semua model files di direktori"""
        if not os.path.exists(self.model_dir):
            print(f"❌ Direktori {self.model_dir} tidak ditemukan!")
            return {
                'enhanced_models': [],
                'pca_models': [],
                'other_models': [],
                'total_files': 0
            }
        
        # Get all pickle files
        all_pkl_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
        
        enhanced_models = []
        pca_models = []
        other_models = []
        
        for file_path in all_pkl_files:
            filename = os.path.basename(file_path)
            
            if 'enhanced' in filename.lower():
                enhanced_models.append(file_path)
            elif 'pca' in filename.lower():
                pca_models.append(file_path)
            else:
                # Try to determine type by loading metadata
                try:
                    import pickle
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check if it's enhanced model by structure
                    if (isinstance(data, dict) and 
                        'training_config' in data and
                        'model_performance' in data):
                        enhanced_models.append(file_path)
                    elif (isinstance(data, dict) and 
                          ('pca_components' in data or 'explained_variance' in data)):
                        pca_models.append(file_path)
                    else:
                        other_models.append(file_path)
                except:
                    other_models.append(file_path)
        
        return {
            'enhanced_models': enhanced_models,
            'pca_models': pca_models,
            'other_models': other_models,
            'total_files': len(all_pkl_files)
        }
    
    def display_scan_results(self, scan_results):
        """Display hasil scan model files"""
        print("🔍 SCAN RESULTS - MODEL FILES")
        print("="*60)
        
        print(f"📊 Total files found: {scan_results['total_files']}")
        print()
        
        if scan_results['enhanced_models']:
            print("✅ ENHANCED MODELS (CURRENT):")
            for model in scan_results['enhanced_models']:
                filename = os.path.basename(model)
                file_size = os.path.getsize(model) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(model))
                print(f"   📁 {filename}")
                print(f"      Size: {file_size:.1f} KB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("❌ ENHANCED MODELS: None found")
        
        print()
        
        if scan_results['pca_models']:
            print("⚠️  PCA MODELS (OLD - SHOULD BE REMOVED):")
            for model in scan_results['pca_models']:
                filename = os.path.basename(model)
                file_size = os.path.getsize(model) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(model))
                print(f"   📁 {filename}")
                print(f"      Size: {file_size:.1f} KB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("✅ PCA MODELS: None found (Good!)")
        
        print()
        
        if scan_results['other_models']:
            print("❓ OTHER MODELS (UNKNOWN TYPE):")
            for model in scan_results['other_models']:
                filename = os.path.basename(model)
                file_size = os.path.getsize(model) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(model))
                print(f"   📁 {filename}")
                print(f"      Size: {file_size:.1f} KB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("✅ OTHER MODELS: None found")
    
    def create_backup_directory(self):
        """Create backup directory if not exists"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            print(f"📁 Created backup directory: {self.backup_dir}")
        return True
    
    def backup_and_remove_pca_models(self, scan_results, confirm=True):
        """Backup dan remove PCA models"""
        pca_models = scan_results['pca_models']
        
        if not pca_models:
            print("✅ No PCA models to remove")
            return True
        
        print(f"\n🗑️  REMOVING PCA MODELS")
        print("="*50)
        print(f"Found {len(pca_models)} PCA model(s) to remove:")
        
        for model in pca_models:
            print(f"   • {os.path.basename(model)}")
        
        if confirm:
            try:
                response = input("\nProceed with backup and removal? (y/n): ").lower().strip()
                if response not in ['y', 'yes', 'ya']:
                    print("❌ Operation cancelled by user")
                    return False
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Operation cancelled")
                return False
        
        # Create backup directory
        self.create_backup_directory()
        
        removed_count = 0
        for model_path in pca_models:
            try:
                filename = os.path.basename(model_path)
                backup_path = os.path.join(self.backup_dir, filename)
                
                # Copy to backup first
                shutil.copy2(model_path, backup_path)
                
                # Then remove original
                os.remove(model_path)
                
                print(f"✅ Removed: {filename} (backed up)")
                removed_count += 1
                
            except Exception as e:
                print(f"❌ Error removing {filename}: {e}")
        
        print(f"\n📊 Summary: {removed_count}/{len(pca_models)} PCA models removed")
        print(f"💾 Backup location: {self.backup_dir}")
        
        return removed_count == len(pca_models)
    
    def force_retrain_enhanced_model(self):
        """Force retrain Enhanced model dengan hapus model lama"""
        print("\n🔄 FORCE RETRAIN ENHANCED MODEL")
        print("="*50)
        
        # Scan current models
        scan_results = self.scan_models()
        
        enhanced_models = scan_results['enhanced_models']
        if enhanced_models:
            print(f"Found {len(enhanced_models)} existing Enhanced model(s):")
            for model in enhanced_models:
                print(f"   • {os.path.basename(model)}")
            
            try:
                response = input("\nRemove existing Enhanced models to force retrain? (y/n): ").lower().strip()
                if response in ['y', 'yes', 'ya']:
                    
                    # Create backup directory
                    self.create_backup_directory()
                    
                    for model_path in enhanced_models:
                        try:
                            filename = os.path.basename(model_path)
                            backup_path = os.path.join(self.backup_dir, f"old_{filename}")
                            
                            # Backup then remove
                            shutil.copy2(model_path, backup_path)
                            os.remove(model_path)
                            
                            print(f"✅ Removed Enhanced model: {filename} (backed up)")
                            
                        except Exception as e:
                            print(f"❌ Error removing {filename}: {e}")
                    
                    print("\n✅ Enhanced models removed. Next app.py run will retrain.")
                    return True
                else:
                    print("❌ Operation cancelled")
                    return False
                    
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Operation cancelled")
                return False
        else:
            print("ℹ️  No existing Enhanced models found. Next app.py run will train new model.")
            return True
    
    def run_interactive_cleanup(self):
        """Run interactive cleanup process"""
        print("🧹 ML CHAOS - MODEL CLEANUP TOOL")
        print("="*60)
        print("Tool untuk membersihkan model PCA lama dan organize Enhanced models")
        print()
        
        # Scan existing models
        scan_results = self.scan_models()
        self.display_scan_results(scan_results)
        
        if scan_results['total_files'] == 0:
            print("\n✅ No model files found. Clean state!")
            return True
        
        # Check if cleanup needed
        needs_cleanup = len(scan_results['pca_models']) > 0
        has_enhanced = len(scan_results['enhanced_models']) > 0
        
        print("\n🎯 RECOMMENDED ACTIONS:")
        print("="*40)
        
        if needs_cleanup:
            print("⚠️  CLEANUP NEEDED:")
            print("   1. Remove PCA models (obsolete)")
            if not has_enhanced:
                print("   2. Retrain Enhanced model")
        else:
            print("✅ NO CLEANUP NEEDED:")
            
        if has_enhanced:
            print("✅ Enhanced models found (Good!)")
        else:
            print("⚠️  No Enhanced models - will be created on next app.py run")
        
        print("\n🛠️  AVAILABLE ACTIONS:")
        print("="*40)
        print("1. Remove PCA models only")
        print("2. Force retrain Enhanced models")
        print("3. Full cleanup (remove all models)")
        print("4. Exit without changes")
        
        try:
            choice = input("\nSelect action (1-4): ").strip()
            
            if choice == '1':
                return self.backup_and_remove_pca_models(scan_results)
                
            elif choice == '2':
                return self.force_retrain_enhanced_model()
                
            elif choice == '3':
                print("\n🗑️  FULL CLEANUP")
                print("This will remove ALL model files and force complete retrain")
                confirm = input("Are you sure? (y/n): ").lower().strip()
                
                if confirm in ['y', 'yes', 'ya']:
                    self.create_backup_directory()
                    
                    all_models = (scan_results['enhanced_models'] + 
                                scan_results['pca_models'] + 
                                scan_results['other_models'])
                    
                    removed_count = 0
                    for model_path in all_models:
                        try:
                            filename = os.path.basename(model_path)
                            backup_path = os.path.join(self.backup_dir, f"full_cleanup_{filename}")
                            
                            shutil.copy2(model_path, backup_path)
                            os.remove(model_path)
                            
                            print(f"✅ Removed: {filename}")
                            removed_count += 1
                            
                        except Exception as e:
                            print(f"❌ Error removing {filename}: {e}")
                    
                    print(f"\n📊 Full cleanup complete: {removed_count} files removed")
                    print("🔄 Next app.py run will create fresh Enhanced models")
                    return True
                else:
                    print("❌ Full cleanup cancelled")
                    return False
                    
            elif choice == '4':
                print("✅ Exiting without changes")
                return True
                
            else:
                print("❌ Invalid choice")
                return False
                
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Operation cancelled")
            return False
    
    def quick_cleanup(self):
        """Quick cleanup - remove PCA models only"""
        print("🚀 QUICK CLEANUP - Removing PCA models only")
        print("="*50)
        
        scan_results = self.scan_models()
        return self.backup_and_remove_pca_models(scan_results, confirm=False)

def main():
    """Main function"""
    import sys
    
    cleanup_tool = ModelCleanupTool()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick cleanup without interaction
            success = cleanup_tool.quick_cleanup()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == '--help':
            print("Model Cleanup Tool")
            print("Usage:")
            print("  python cleanup_old_models.py           # Interactive mode")
            print("  python cleanup_old_models.py --quick   # Quick PCA removal")
            print("  python cleanup_old_models.py --help    # Show this help")
            sys.exit(0)
    
    # Interactive mode
    try:
        success = cleanup_tool.run_interactive_cleanup()
        
        if success:
            print("\n✅ Cleanup completed successfully!")
            print("💡 Next steps:")
            print("   1. Run 'python run.py' to start app with Enhanced model")
            print("   2. Run 'python model/analyze_weights.py' to analyze Enhanced weights")
        else:
            print("\n⚠️  Cleanup completed with some issues")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()