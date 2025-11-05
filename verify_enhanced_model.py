"""
Enhanced Model Verification Script
File: verify_enhanced_model.py

Tujuan: Memverifikasi bahwa sistem menggunakan Enhanced model yang benar
dan bukan model PCA lama.

Author: ML Chaos Team
Version: 1.0
"""

import os
import sys
import pickle
import glob
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'model' else current_dir
sys.path.insert(0, project_root)

class EnhancedModelVerifier:
    """Verifier untuk memastikan Enhanced model yang benar digunakan"""
    
    def __init__(self):
        self.project_root = project_root
        self.model_dir = os.path.join(project_root, 'saved_models')
        self.current_model_info = None
        
    def scan_and_identify_models(self):
        """Scan dan identifikasi semua model files"""
        if not os.path.exists(self.model_dir):
            return {
                'status': 'no_model_dir',
                'message': f'Model directory tidak ditemukan: {self.model_dir}',
                'models': []
            }
        
        pkl_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
        
        if not pkl_files:
            return {
                'status': 'no_models',
                'message': 'Tidak ada model files (.pkl) ditemukan',
                'models': []
            }
        
        models_info = []
        
        for file_path in pkl_files:
            try:
                model_info = self._analyze_model_file(file_path)
                models_info.append(model_info)
            except Exception as e:
                models_info.append({
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'type': 'error',
                    'error': str(e),
                    'size_kb': os.path.getsize(file_path) / 1024,
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path))
                })
        
        # Sort by modification time (newest first)
        models_info.sort(key=lambda x: x.get('modified', datetime.min), reverse=True)
        
        return {
            'status': 'success',
            'message': f'Found {len(models_info)} model file(s)',
            'models': models_info
        }
    
    def _analyze_model_file(self, file_path):
        """Analyze individual model file untuk menentukan tipe"""
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            model_info = {
                'file_path': file_path,
                'filename': filename,
                'size_kb': file_size,
                'modified': modified,
                'loadable': True
            }
            
            # Determine model type based on structure
            if isinstance(data, dict):
                
                # Check for Enhanced model indicators
                enhanced_indicators = [
                    'user_vector_weights',
                    'question_importance', 
                    'model_performance',
                    'training_config',
                    'feature_cols'
                ]
                
                enhanced_score = sum(1 for indicator in enhanced_indicators if indicator in data)
                
                # Check for PCA indicators
                pca_indicators = [
                    'pca_components',
                    'explained_variance',
                    'pca_model',
                    'principal_components'
                ]
                
                pca_score = sum(1 for indicator in pca_indicators if indicator in data)
                
                if enhanced_score >= 3:
                    model_info.update({
                        'type': 'enhanced',
                        'confidence': 'high' if enhanced_score >= 4 else 'medium',
                        'training_date': data.get('trained_date'),
                        'dataset_shape': data.get('dataset_shape'),
                        'training_method': data.get('training_config', {}).get('method') if data.get('training_config') else None,
                        'questions': list(data.get('user_vector_weights', {}).keys()) if data.get('user_vector_weights') else [],
                        'model_hash': data.get('model_hash'),
                        'feature_count': len(data.get('feature_cols', [])) if data.get('feature_cols') else 0
                    })
                    
                    # Enhanced model performance check
                    if 'model_performance' in data:
                        performance_data = data['model_performance']
                        if isinstance(performance_data, dict):
                            total_r2_scores = []
                            for question_perf in performance_data.values():
                                if isinstance(question_perf, dict):
                                    for answer_perf in question_perf.values():
                                        if isinstance(answer_perf, dict) and 'r2' in answer_perf:
                                            total_r2_scores.append(answer_perf['r2'])
                            
                            if total_r2_scores:
                                import numpy as np
                                model_info['avg_r2_score'] = np.mean(total_r2_scores)
                                model_info['performance_quality'] = (
                                    'excellent' if model_info['avg_r2_score'] > 0.8 else
                                    'good' if model_info['avg_r2_score'] > 0.6 else
                                    'fair' if model_info['avg_r2_score'] > 0.4 else
                                    'poor'
                                )
                
                elif pca_score >= 1 or 'pca' in filename.lower():
                    model_info.update({
                        'type': 'pca',
                        'confidence': 'high' if pca_score >= 2 else 'medium',
                        'warning': 'This is an old PCA model - should be removed',
                        'training_date': data.get('trained_date'),
                        'dataset_shape': data.get('dataset_shape'),
                        'questions': list(data.get('user_vector_weights', {}).keys()) if data.get('user_vector_weights') else [],
                        'pca_components': data.get('pca_components'),
                        'explained_variance': data.get('explained_variance')
                    })
                
                else:
                    model_info.update({
                        'type': 'unknown',
                        'confidence': 'low',
                        'available_keys': list(data.keys())[:10],  # Show first 10 keys
                        'warning': 'Unknown model format'
                    })
            
            else:
                model_info.update({
                    'type': 'invalid',
                    'confidence': 'high',
                    'data_type': str(type(data)),
                    'error': 'Model data is not a dictionary'
                })
            
            return model_info
            
        except Exception as e:
            return {
                'file_path': file_path,
                'filename': filename,
                'size_kb': file_size,
                'modified': modified,
                'type': 'error',
                'confidence': 'high',
                'error': str(e),
                'loadable': False
            }
    
    def display_model_analysis(self, scan_result):
        """Display hasil analisis model files"""
        print("🔍 ENHANCED MODEL VERIFICATION REPORT")
        print("="*80)
        
        if scan_result['status'] != 'success':
            print(f"❌ {scan_result['message']}")
            return False
        
        models = scan_result['models']
        enhanced_models = [m for m in models if m.get('type') == 'enhanced']
        pca_models = [m for m in models if m.get('type') == 'pca']
        other_models = [m for m in models if m.get('type') not in ['enhanced', 'pca']]
        
        print(f"📊 Total models found: {len(models)}")
        print(f"   • Enhanced models: {len(enhanced_models)}")
        print(f"   • PCA models: {len(pca_models)}")
        print(f"   • Other/Unknown: {len(other_models)}")
        print()
        
        # Display Enhanced models
        if enhanced_models:
            print("✅ ENHANCED MODELS (RECOMMENDED)")
            print("-" * 60)
            for i, model in enumerate(enhanced_models, 1):
                self._display_enhanced_model_details(model, i)
        else:
            print("❌ NO ENHANCED MODELS FOUND!")
            print("   Sistem akan menggunakan model lama atau error")
        
        print()
        
        # Display PCA models (should be removed)
        if pca_models:
            print("⚠️  PCA MODELS (OBSOLETE - SHOULD BE REMOVED)")
            print("-" * 60)
            for i, model in enumerate(pca_models, 1):
                self._display_pca_model_details(model, i)
            print("\n💡 Recommendation: Remove PCA models using cleanup_old_models.py")
        
        # Display other models
        if other_models:
            print("\n❓ OTHER/UNKNOWN MODELS")
            print("-" * 40)
            for i, model in enumerate(other_models, 1):
                self._display_other_model_details(model, i)
        
        return True
    
    def _display_enhanced_model_details(self, model, index):
        """Display details untuk Enhanced model"""
        print(f"{index}. 📁 {model['filename']}")
        print(f"   Type: Enhanced Statistical Model ✅")
        print(f"   Size: {model['size_kb']:.1f} KB")
        print(f"   Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if model.get('training_date'):
            print(f"   Trained: {model['training_date']}")
        
        if model.get('training_method'):
            print(f"   Method: {model['training_method']}")
        
        if model.get('dataset_shape'):
            print(f"   Dataset: {model['dataset_shape']} (rows, cols)")
        
        if model.get('feature_count'):
            print(f"   Features: {model['feature_count']}")
        
        if model.get('questions'):
            print(f"   Questions: {', '.join(model['questions'])}")
        
        if model.get('avg_r2_score') is not None:
            r2_score = model['avg_r2_score']
            quality = model.get('performance_quality', 'unknown')
            print(f"   Performance: R² = {r2_score:.4f} ({quality})")
        
        print(f"   Confidence: {model.get('confidence', 'unknown').upper()}")
        print()
    
    def _display_pca_model_details(self, model, index):
        """Display details untuk PCA model"""
        print(f"{index}. 📁 {model['filename']} ⚠️")
        print(f"   Type: PCA Model (Obsolete)")
        print(f"   Size: {model['size_kb']:.1f} KB")
        print(f"   Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Warning: {model.get('warning', 'Should be removed')}")
        
        if model.get('questions'):
            print(f"   Questions: {', '.join(model['questions'])}")
        
        print()
    
    def _display_other_model_details(self, model, index):
        """Display details untuk other/unknown models"""
        print(f"{index}. 📁 {model['filename']}")
        print(f"   Type: {model.get('type', 'unknown').upper()}")
        print(f"   Size: {model['size_kb']:.1f} KB")
        print(f"   Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if model.get('error'):
            print(f"   Error: {model['error']}")
        elif model.get('available_keys'):
            print(f"   Keys: {', '.join(model['available_keys'])}")
        
        print()
    
    def get_current_system_model(self):
        """Simulate getting current model yang akan diload oleh sistem"""
        try:
            # Import recommender untuk cek model yang akan diload
            sys.path.insert(0, os.path.join(self.project_root, 'model'))
            
            # Check which model file would be loaded
            from model.recommender import HybridRecommendationSystem
            
            # Create temporary instance to check model loading
            temp_system = HybridRecommendationSystem(model_dir=self.model_dir)
            
            if hasattr(temp_system, 'model_hash') and temp_system.model_hash:
                model_filename = temp_system._get_model_filename()
                model_path = os.path.join(self.model_dir, model_filename)
                
                if os.path.exists(model_path):
                    return {
                        'status': 'found',
                        'model_path': model_path,
                        'filename': model_filename,
                        'model_hash': temp_system.model_hash,
                        'trained': temp_system.trained
                    }
            
            return {
                'status': 'not_found',
                'message': 'No compatible Enhanced model found for current dataset'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking current system model: {e}'
            }
    
    def display_system_status(self):
        """Display status sistem saat ini"""
        print("\n🎯 CURRENT SYSTEM STATUS")
        print("="*60)
        
        current_model = self.get_current_system_model()
        
        if current_model['status'] == 'found':
            print("✅ SYSTEM READY")
            print(f"   Current model: {current_model['filename']}")
            print(f"   Model hash: {current_model['model_hash']}")
            print(f"   Trained: {'Yes' if current_model['trained'] else 'No'}")
            print(f"   Path: {current_model['model_path']}")
            
        elif current_model['status'] == 'not_found':
            print("⚠️  SYSTEM NOT READY")
            print(f"   Issue: {current_model['message']}")
            print("   Next app.py run will train new Enhanced model")
            
        else:  # error
            print("❌ SYSTEM ERROR")
            print(f"   Error: {current_model['message']}")
    
    def generate_recommendations(self):
        """Generate recommendations berdasarkan analysis"""
        print("\n💡 RECOMMENDATIONS")
        print("="*50)
        
        scan_result = self.scan_and_identify_models()
        
        if scan_result['status'] != 'success':
            print("1. Create saved_models directory")
            print("2. Run 'python run.py' to generate Enhanced model")
            return
        
        models = scan_result['models']
        enhanced_models = [m for m in models if m.get('type') == 'enhanced']
        pca_models = [m for m in models if m.get('type') == 'pca']
        
        recommendations = []
        
        if not enhanced_models:
            recommendations.append("🔴 CRITICAL: Generate Enhanced model")
            recommendations.append("   → Run 'python run.py' to create Enhanced model")
        else:
            recommendations.append("✅ Enhanced models available")
        
        if pca_models:
            recommendations.append("🟡 CLEANUP: Remove obsolete PCA models")
            recommendations.append("   → Run 'python cleanup_old_models.py --quick'")
        
        if enhanced_models and not pca_models:
            recommendations.append("✅ System is properly configured")
            recommendations.append("   → Run 'python model/analyze_weights.py' to analyze weights")
        
        for rec in recommendations:
            print(rec)
    
    def run_complete_verification(self):
        """Run complete verification process"""
        print("🚀 ML CHAOS - ENHANCED MODEL VERIFICATION")
        print("="*80)
        print("Memverifikasi bahwa sistem menggunakan Enhanced model yang benar...")
        print()
        
        # Scan and analyze models
        scan_result = self.scan_and_identify_models()
        success = self.display_model_analysis(scan_result)
        
        if success:
            # Display current system status
            self.display_system_status()
            
            # Generate recommendations
            self.generate_recommendations()
            
            print("\n🎯 NEXT STEPS")
            print("="*40)
            print("1. Fix any issues identified above")
            print("2. Run 'python run.py' to start system with Enhanced model")
            print("3. Run 'python model/analyze_weights.py' to analyze Enhanced weights")
            print("4. Verify that analyzer shows Enhanced statistical results")
        
        return success

def main():
    """Main function"""
    import sys
    
    verifier = EnhancedModelVerifier()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick check without detailed analysis
            scan_result = verifier.scan_and_identify_models()
            if scan_result['status'] == 'success':
                models = scan_result['models']
                enhanced_count = len([m for m in models if m.get('type') == 'enhanced'])
                pca_count = len([m for m in models if m.get('type') == 'pca'])
                
                if enhanced_count > 0 and pca_count == 0:
                    print("✅ VERIFICATION PASSED: Enhanced model ready, no PCA models")
                    sys.exit(0)
                elif enhanced_count > 0 and pca_count > 0:
                    print("⚠️  VERIFICATION WARNING: Enhanced model found but PCA models present")
                    sys.exit(1)
                else:
                    print("❌ VERIFICATION FAILED: No Enhanced models found")
                    sys.exit(2)
            else:
                print("❌ VERIFICATION ERROR: Cannot scan models")
                sys.exit(3)
                
        elif sys.argv[1] == '--help':
            print("Enhanced Model Verification Tool")
            print("Usage:")
            print("  python verify_enhanced_model.py           # Full verification")
            print("  python verify_enhanced_model.py --quick   # Quick check only")
            print("  python verify_enhanced_model.py --help    # Show this help")
            sys.exit(0)
    
    # Full verification mode
    try:
        success = verifier.run_complete_verification()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()