import pandas as pd
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
import os
from datetime import datetime
import sys
from flask import Flask
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config.config import Config
from models.database import execute_query
from models.database import init_mysql


app = Flask(__name__)
app.config.from_object(Config)
init_mysql(app)


# Import your recommender system
# Adjust this import path according to your project structure
from recommender import EnhancedHybridRecommendationSystem

class TrainingDataExporter:
    """
    Export training data yang digunakan oleh Enhanced Hybrid Recommendation System
    untuk keperluan analisis dan debugging
    """
    
    def __init__(self, output_dir="exported_data"):
        self.output_dir = output_dir
        self.recommender = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created output directory: {output_dir}")
    
    def initialize_recommender(self):
        """Initialize recommender system untuk mengakses data dan methods"""
        try:
            print("🔄 Initializing recommender system...")
            self.recommender = EnhancedHybridRecommendationSystem()
            
            if self.recommender.menu_df is None or self.recommender.menu_df.empty:
                raise RuntimeError("No menu data available!")
                
            print(f"✅ Recommender initialized with {len(self.recommender.menu_df)} menu items")
            return True
        except Exception as e:
            print(f"❌ Error initializing recommender: {e}")
            return False
    
    def export_menu_data(self):
        """Export menu data yang sudah dipreprocess"""
        if self.recommender is None or self.recommender.menu_df.empty:
            print("❌ No menu data available")
            return None
            
        try:
            filename = f"menu_data_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Export menu data dengan semua features
            menu_export = self.recommender.menu_df.copy()
            menu_export.to_csv(filepath, index=False)
            
            print(f"✅ Menu data exported to: {filepath}")
            print(f"   - Rows: {len(menu_export)}")
            print(f"   - Columns: {len(menu_export.columns)}")
            print(f"   - Features: {self.recommender.feature_cols}")
            
            return filepath
        except Exception as e:
            print(f"❌ Error exporting menu data: {e}")
            return None
    
    def generate_and_export_dummy_data(self, n_users=1500, random_seed=42):
        """Generate dan export dummy user data yang digunakan untuk training"""
        if self.recommender is None:
            print("❌ Recommender not initialized")
            return None
            
        try:
            print(f"🔄 Generating dummy data for {n_users} users...")
            
            # Use the same method as in the recommender
            dummy_data = self.recommender._generate_fixed_dummy_data(n_users, random_seed)
            
            # Convert to DataFrame format
            dummy_df = pd.DataFrame(dummy_data)
            
            # Expand selected_items into separate columns for easier analysis
            max_items = max(len(entry['selected_items']) for entry in dummy_data)
            
            for i in range(max_items):
                dummy_df[f'selected_item_{i+1}'] = dummy_df['selected_items'].apply(
                    lambda x: x[i] if i < len(x) else None
                )
            
            # Add metadata
            dummy_df['n_selected_items'] = dummy_df['selected_items'].apply(len)
            dummy_df['export_timestamp'] = datetime.now()
            
            # Remove the list column for CSV export
            dummy_df_export = dummy_df.drop('selected_items', axis=1)
            
            # Export to CSV
            filename = f"dummy_user_data_{n_users}users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            dummy_df_export.to_csv(filepath, index=False)
            
            print(f"✅ Dummy user data exported to: {filepath}")
            print(f"   - Users: {len(dummy_df_export)}")
            print(f"   - Max items per user: {max_items}")
            
            # Export summary statistics
            self._export_dummy_data_summary(dummy_df, n_users, random_seed)
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error generating dummy data: {e}")
            return None
    
    def _export_dummy_data_summary(self, dummy_df, n_users, random_seed):
        """Export summary statistics dari dummy data"""
        try:
            summary_data = {
                'generation_info': {
                    'n_users': n_users,
                    'random_seed': random_seed,
                    'export_timestamp': datetime.now().isoformat(),
                    'total_generated_users': len(dummy_df)
                },
                'preference_distributions': {},
                'selection_statistics': {}
            }
            
            # Calculate preference distributions
            questions = ['taste', 'mood', 'texture', 'caffeine', 'temperature', 'budget']
            for question in questions:
                if question in dummy_df.columns:
                    dist = dummy_df[question].value_counts(normalize=True).to_dict()
                    summary_data['preference_distributions'][question] = dist
            
            # Calculate selection statistics
            summary_data['selection_statistics'] = {
                'avg_items_per_user': dummy_df['n_selected_items'].mean(),
                'min_items_per_user': dummy_df['n_selected_items'].min(),
                'max_items_per_user': dummy_df['n_selected_items'].max(),
                'std_items_per_user': dummy_df['n_selected_items'].std()
            }
            
            # Convert to DataFrame and export
            summary_rows = []
            
            # Add generation info
            for key, value in summary_data['generation_info'].items():
                summary_rows.append({'category': 'generation_info', 'metric': key, 'value': value})
            
            # Add preference distributions
            for question, dist in summary_data['preference_distributions'].items():
                for answer, percentage in dist.items():
                    summary_rows.append({
                        'category': 'preference_distribution', 
                        'metric': f'{question}_{answer}', 
                        'value': percentage
                    })
            
            # Add selection statistics
            for metric, value in summary_data['selection_statistics'].items():
                summary_rows.append({'category': 'selection_stats', 'metric': metric, 'value': value})
            
            summary_df = pd.DataFrame(summary_rows)
            
            filename = f"dummy_data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            summary_df.to_csv(filepath, index=False)
            print(f"✅ Dummy data summary exported to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error exporting summary: {e}")
    
    def export_cooccurrence_matrix(self, n_users=1500, random_seed=42):
        """Generate dan export co-occurrence matrix"""
        if self.recommender is None:
            print("❌ Recommender not initialized")
            return None
            
        try:
            print("🔄 Generating co-occurrence matrix...")
            
            # Generate dummy data
            dummy_data = self.recommender._generate_fixed_dummy_data(n_users, random_seed)
            
            # Build co-occurrence matrix using the same method
            co_occurrence = self.recommender._build_balanced_cooccurrence_matrix(dummy_data)
            
            # Convert to DataFrame
            cooccurrence_rows = []
            for (question, answer, item1, item2), count in co_occurrence.items():
                cooccurrence_rows.append({
                    'question': question,
                    'answer': answer,
                    'item1': item1,
                    'item2': item2,
                    'co_occurrence_count': count,
                    'export_timestamp': datetime.now()
                })
            
            cooccurrence_df = pd.DataFrame(cooccurrence_rows)
            
            # Export to CSV
            filename = f"cooccurrence_matrix_{n_users}users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            cooccurrence_df.to_csv(filepath, index=False)
            
            print(f"✅ Co-occurrence matrix exported to: {filepath}")
            print(f"   - Total co-occurrence pairs: {len(cooccurrence_df)}")
            print(f"   - Questions covered: {sorted(cooccurrence_df['question'].unique())}")
            
            # Export summary by question
            question_summary = cooccurrence_df.groupby(['question', 'answer']).agg({
                'co_occurrence_count': ['count', 'sum', 'mean', 'std']
            }).round(4)
            
            question_summary.columns = ['pair_count', 'total_cooccurrence', 'avg_cooccurrence', 'std_cooccurrence']
            question_summary = question_summary.reset_index()
            
            summary_filename = f"cooccurrence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_filepath = os.path.join(self.output_dir, summary_filename)
            
            question_summary.to_csv(summary_filepath, index=False)
            print(f"✅ Co-occurrence summary exported to: {summary_filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error exporting co-occurrence matrix: {e}")
            return None
    
    def export_regression_dataset(self, n_users=1500, random_seed=42):
        """Generate dan export regression dataset yang digunakan untuk training"""
        if self.recommender is None:
            print("❌ Recommender not initialized")
            return None
            
        try:
            print("🔄 Generating regression dataset...")
            
            # Generate dummy data and co-occurrence matrix
            dummy_data = self.recommender._generate_fixed_dummy_data(n_users, random_seed)
            co_occurrence = self.recommender._build_balanced_cooccurrence_matrix(dummy_data)
            
            # Build regression dataset using the same method
            regression_df = self.recommender._build_balanced_regression_dataset(co_occurrence)
            
            # Add metadata
            regression_df['export_timestamp'] = datetime.now()
            regression_df['source_users'] = n_users
            regression_df['source_seed'] = random_seed
            
            # Export to CSV
            filename = f"regression_dataset_{n_users}users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            regression_df.to_csv(filepath, index=False)
            
            print(f"✅ Regression dataset exported to: {filepath}")
            print(f"   - Total rows: {len(regression_df)}")
            print(f"   - Feature columns: {len([col for col in regression_df.columns if col not in ['question', 'answer', 'co_occurrence', 'export_timestamp', 'source_users', 'source_seed']])}")
            print(f"   - Questions: {sorted(regression_df['question'].unique())}")
            
            # Export feature summary
            feature_cols = [col for col in regression_df.columns 
                          if col not in ['question', 'answer', 'co_occurrence', 'export_timestamp', 'source_users', 'source_seed']]
            
            feature_summary = regression_df[feature_cols].describe().T
            feature_summary['feature_name'] = feature_summary.index
            feature_summary = feature_summary.reset_index(drop=True)
            
            feature_filename = f"regression_features_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            feature_filepath = os.path.join(self.output_dir, feature_filename)
            
            feature_summary.to_csv(feature_filepath, index=False)
            print(f"✅ Feature summary exported to: {feature_filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error exporting regression dataset: {e}")
            return None
    
    def export_all_training_data(self, n_users=1500, random_seed=42):
        """Export semua data yang digunakan dalam training process"""
        print("🚀 Starting complete training data export...")
        print("=" * 60)
        
        results = {
            'menu_data': None,
            'dummy_data': None,
            'cooccurrence_matrix': None,
            'regression_dataset': None
        }
        
        # Export menu data
        print("\n1. Exporting menu data...")
        results['menu_data'] = self.export_menu_data()
        
        # Export dummy user data
        print("\n2. Exporting dummy user data...")
        results['dummy_data'] = self.generate_and_export_dummy_data(n_users, random_seed)
        
        # Export co-occurrence matrix
        print("\n3. Exporting co-occurrence matrix...")
        results['cooccurrence_matrix'] = self.export_cooccurrence_matrix(n_users, random_seed)
        
        # Export regression dataset
        print("\n4. Exporting regression dataset...")
        results['regression_dataset'] = self.export_regression_dataset(n_users, random_seed)
        
        # Create export summary
        print("\n5. Creating export summary...")
        self._create_export_summary(results, n_users, random_seed)
        
        print("\n" + "=" * 60)
        print("✅ Complete training data export finished!")
        print(f"📁 All files saved to: {self.output_dir}")
        
        return results
    
    def _create_export_summary(self, results, n_users, random_seed):
        """Create summary file dengan informasi semua exports"""
        try:
            summary_data = {
                'export_info': {
                    'export_timestamp': datetime.now().isoformat(),
                    'n_users': n_users,
                    'random_seed': random_seed,
                    'output_directory': self.output_dir
                },
                'exported_files': {
                    'menu_data': results['menu_data'],
                    'dummy_data': results['dummy_data'], 
                    'cooccurrence_matrix': results['cooccurrence_matrix'],
                    'regression_dataset': results['regression_dataset']
                },
                'export_status': {
                    'menu_data': 'SUCCESS' if results['menu_data'] else 'FAILED',
                    'dummy_data': 'SUCCESS' if results['dummy_data'] else 'FAILED',
                    'cooccurrence_matrix': 'SUCCESS' if results['cooccurrence_matrix'] else 'FAILED',
                    'regression_dataset': 'SUCCESS' if results['regression_dataset'] else 'FAILED'
                }
            }
            
            # Convert to rows for CSV
            summary_rows = []
            for category, data in summary_data.items():
                for key, value in data.items():
                    summary_rows.append({
                        'category': category,
                        'key': key,
                        'value': str(value)
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            
            filename = f"export_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            summary_df.to_csv(filepath, index=False)
            print(f"✅ Export summary saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error creating export summary: {e}")


def main():
    """Main function untuk menjalankan export"""
    print("🚀 Training Data Exporter for Enhanced Hybrid Recommendation System")
    print("=" * 80)
    
    # Initialize exporter
    exporter = TrainingDataExporter(output_dir="exported_training_data")
    
    # Initialize recommender system
    if not exporter.initialize_recommender():
        print("❌ Failed to initialize recommender system. Exiting.")
        return
    
    # Configuration
    n_users = 1500  # Adjust sesuai kebutuhan
    random_seed = 42
    
    print(f"\n📋 Export Configuration:")
    print(f"   - Number of users: {n_users}")
    print(f"   - Random seed: {random_seed}")
    print(f"   - Output directory: {exporter.output_dir}")
    
    # Ask user what to export
    print("\n🎯 What would you like to export?")
    print("   1. Menu data only")
    print("   2. Dummy user data only")
    print("   3. Co-occurrence matrix only")
    print("   4. Regression dataset only")
    print("   5. All training data (recommended)")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            exporter.export_menu_data()
        elif choice == "2":
            exporter.generate_and_export_dummy_data(n_users, random_seed)
        elif choice == "3":
            exporter.export_cooccurrence_matrix(n_users, random_seed)
        elif choice == "4":
            exporter.export_regression_dataset(n_users, random_seed)
        elif choice == "5":
            exporter.export_all_training_data(n_users, random_seed)
        else:
            print("❌ Invalid choice. Exporting all data by default...")
            exporter.export_all_training_data(n_users, random_seed)
            
    except KeyboardInterrupt:
        print("\n⚠️ Export interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during export: {e}")

if __name__ == "__main__":
    with app.app_context():
        main()
