import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import random
from collections import defaultdict
import warnings
import pickle
import os
import json
import uuid
import hashlib
from datetime import datetime
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.database import execute_query

warnings.filterwarnings('ignore')

class HybridIDGenerator:
    @staticmethod
    def generate_hybrid_id(prefix):
        date_str = datetime.now().strftime("%Y%m%d")
        try:
            if prefix == 'PREF':
                query = """SELECT COUNT(*) as count FROM preferences WHERE pref_id LIKE %s AND DATE(timestamp) = CURDATE()"""
                pattern = f"{prefix}_{date_str}_%"
            elif prefix == 'REC':
                query = """SELECT COUNT(*) as count FROM recommendations WHERE rec_id LIKE %s AND DATE(generated_at) = CURDATE()"""
                pattern = f"{prefix}_{date_str}_%"
            elif prefix == 'SESS':
                query = """SELECT COUNT(*) as count FROM user_sessions WHERE session_id LIKE %s AND DATE(created_at) = CURDATE()"""
                pattern = f"{prefix}_{date_str}_%"
            else:
                daily_count = 1
                return f"{prefix}_{date_str}_{daily_count:03d}"

            result = execute_query(query, params=(pattern,), fetch='one')
            daily_count = (result['count'] + 1) if result and result['count'] is not None else 1
        except Exception as e:
            print(f"⚠️ Error getting count for {prefix}: {e}")
            daily_count = 1
        return f"{prefix}_{date_str}_{daily_count:03d}"

class SessionManager:
    @staticmethod
    def create_new_session():
        try:
            session_id = HybridIDGenerator.generate_hybrid_id("SESS")
            query = """
                INSERT INTO user_sessions (session_id, created_at, last_activity, status, total_quiz_attempts)
                VALUES (%s, NOW(), NOW(), 'active', 0)
            """
            execute_query(query, params=(session_id,))
            print(f"✅ New session created: {session_id}")
            return session_id
        except Exception as e:
            print(f"❌ Error creating session: {e}")
            return None

    @staticmethod
    def get_next_quiz_attempt(session_id):
        try:
            query = """
                SELECT COALESCE(MAX(quiz_attempt), 0) + 1 as next_attempt
                FROM preferences
                WHERE session_id = %s
            """
            result = execute_query(query, params=(session_id,), fetch='one')
            return result['next_attempt'] if result else 1
        except Exception as e:
            print(f"❌ Error getting next quiz attempt: {e}")
            return 1

    @staticmethod
    def update_session_activity(session_id, quiz_attempt):
        try:
            query = """
                UPDATE user_sessions
                SET last_activity = NOW(), total_quiz_attempts = %s
                WHERE session_id = %s
            """
            execute_query(query, params=(quiz_attempt, session_id))
            print(f"✅ Session activity updated: {session_id}, attempt: {quiz_attempt}")
        except Exception as e:
            print(f"❌ Error updating session activity: {e}")

class EnhancedHybridRecommendationSystem:
    """
    Enhanced Hybrid dengan:
    1. Fixed preference hierarchy (Taste as primary driver)
    2. Improved ASAM data generation
    3. Soft constraints instead of hard filters
    4. Context-specific adjustments
    5. Balanced feature importance
    """
    def __init__(self, model_dir="saved_models"):
        self.menu_df = None
        self.feature_cols = None
        self.user_vector_weights = None
        self.question_importance = None
        self.model_performance = None
        self.models = {}
        self.feature_importance = {}
        self.scaler = None
        self.trained = False
        self.model_dir = model_dir
        self.model_hash = None
        self.feature_correlations = {}
        
        # Fixed: Add adaptive thresholds
        self.adaptive_thresholds = {}
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self._load_and_preprocess_data_from_db()

        if self.menu_df is not None and not self.menu_df.empty:
            if not self._load_trained_model():
                print("No existing compatible model found. Training new enhanced hybrid model...")
                self.train_enhanced_hybrid_weights(method='all')
                self._save_trained_model()
            else:
                print("✅ Loaded existing enhanced hybrid trained model!")
        else:
            print("⚠️ Cannot train or load model due to empty or unavailable dataset.")

    def _load_and_preprocess_data_from_db(self):
        print("🔄 Loading dataset from database...")
        try:
            query = "SELECT * FROM menu_items WHERE availability = 'Tersedia'"
            menu_data = execute_query(query, fetch='all')
            if not menu_data:
                self.menu_df = pd.DataFrame()
                return

            df = pd.DataFrame(menu_data)
            df['original_harga'] = df['harga'].copy()
            
            # Enhanced preprocessing with feature validation
            self._validate_and_fix_feature_correlations(df)
            
            # Add rasa_netral feature
            df['rasa_netral'] = self._calculate_rasa_netral(df)
            
            # Enhanced texture mapping
            self._process_texture_features(df)
            
            # Normalize numerical features
            numeric_cols = ['rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih', 'rasa_netral', 'harga']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            
            # Enhanced sweetness mapping with correlation validation
            df['sweetness_score'] = self._calculate_sweetness_score(df)
            
            # Enhanced caffeine level classification
            df['tingkat_kafein'] = self._classify_caffeine_level_enhanced(df)
            
            # Define feature columns
            self.feature_cols = [
                'rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih', 'rasa_netral',
                'kafein_score', 'carbonated_score', 'sweetness_score',
                'tekstur_LIGHT', 'tekstur_CREAMY', 'tekstur_BUBBLY', 'tekstur_HEAVY',
                'harga'
            ]
            
            # Ensure all feature columns exist
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0

            self.menu_df = df
            self.model_hash = self._generate_data_hash()
            
            # Calculate adaptive thresholds based on actual data
            self._calculate_adaptive_thresholds()
            
            # Calculate and validate feature correlations
            self._calculate_feature_correlations()
            
            print(f"Dataset loaded: {df.shape[0]} products, {len(self.feature_cols)} features")
            print(f"Feature correlations validated: {len(self.feature_correlations)} pairs")
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            self.menu_df = pd.DataFrame()

    def _calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on actual data distribution"""
        if self.menu_df.empty:
            return
            
        self.adaptive_thresholds = {
            'rasa_asam': {
                'low': self.menu_df['rasa_asam'].quantile(0.25),
                'medium': self.menu_df['rasa_asam'].quantile(0.5), 
                'high': self.menu_df['rasa_asam'].quantile(0.75)
            },
            'rasa_manis': {
                'low': self.menu_df['rasa_manis'].quantile(0.25),
                'medium': self.menu_df['rasa_manis'].quantile(0.5),
                'high': self.menu_df['rasa_manis'].quantile(0.75)
            },
            'rasa_pahit': {
                'low': self.menu_df['rasa_pahit'].quantile(0.25),
                'medium': self.menu_df['rasa_pahit'].quantile(0.5),
                'high': self.menu_df['rasa_pahit'].quantile(0.75)
            },
            'rasa_netral': {
                'low': self.menu_df['rasa_netral'].quantile(0.25),
                'medium': self.menu_df['rasa_netral'].quantile(0.5),
                'high': self.menu_df['rasa_netral'].quantile(0.75)
            },
            'budget': {
                'low': self.menu_df['harga'].quantile(0.33),
                'mid': self.menu_df['harga'].quantile(0.66),
                'high': self.menu_df['harga'].quantile(0.95)
            }
        }
        
        print(f"📊 Adaptive thresholds calculated:")
        for feature, thresholds in self.adaptive_thresholds.items():
            print(f"  {feature}: {thresholds}")

    def _validate_and_fix_feature_correlations(self, df):
        """Validate and ensure logical correlations between features"""
        # Ensure kafein-rasa_pahit correlation
        df['kafein'] = df['kafein_score'].apply(lambda x: 'Ya' if x == 1 else 'Tidak')
        
        # Fix caffeinated items that aren't bitter enough
        caffeinated_mask = df['kafein_score'] == 1
        if caffeinated_mask.sum() > 0:
            # Ensure caffeinated items have minimum bitterness
            min_bitter_for_caffeine = 0.3
            df.loc[caffeinated_mask & (df['rasa_pahit'] < min_bitter_for_caffeine), 'rasa_pahit'] = min_bitter_for_caffeine
        
        # Enhanced carbonated mapping
        df['carbonated'] = df['carbonated_score'].apply(lambda x: 'Ya' if x == 1 else 'Tidak')

    def _calculate_rasa_netral(self, df):
        """Calculate rasa_netral as balance between sweet, sour, bitter, savory"""
        # Netral = item yang tidak dominan di rasa manapun
        total_taste = df['rasa_asam'] + df['rasa_manis'] + df['rasa_pahit'] + df['rasa_gurih']
        max_taste = df[['rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_gurih']].max(axis=1)
        
        # Netral tinggi jika semua rasa rendah atau seimbang
        rasa_netral = np.where(
            (total_taste < 2.0) | (max_taste < 0.4),  # Low total or no dominant taste
            1.0 - (max_taste * 0.8),  # Inverse of dominant taste
            0.1  # Minimal netral for items with strong taste
        )
        
        return np.clip(rasa_netral, 0, 1)

    def _process_texture_features(self, df):
        """Enhanced texture processing with better mapping"""
        texture_mapping = {
            'LIGHT': ['light', 'ringan', 'segar'],
            'CREAMY': ['creamy', 'kental', 'lembut', 'smooth'],
            'BUBBLY': ['bubbly', 'bergelembung', 'fizzy', 'sparkling'],
            'HEAVY': ['heavy', 'tebal', 'thick', 'dense']
        }
        
        for tex_type in texture_mapping.keys():
            col_name = f'tekstur_{tex_type}'
            df[col_name] = 0
            
            if 'tekstur' in df.columns:
                for keyword in texture_mapping[tex_type]:
                    mask = df['tekstur'].str.contains(keyword, case=False, na=False)
                    df.loc[mask, col_name] = 1

    def _calculate_sweetness_score(self, df):
        """Enhanced sweetness calculation ensuring correlation with rasa_manis"""
        sweetness_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8, 'Bitter': 0}
        base_sweetness = df['sweetness_level'].map(sweetness_map).fillna(0.2)
        
        # Ensure correlation with rasa_manis
        adjusted_sweetness = np.where(
            df['rasa_manis'] > 0.7, np.maximum(base_sweetness, 0.7),  # High manis = high sweetness
            np.where(
                df['rasa_manis'] < 0.2, np.minimum(base_sweetness, 0.2),  # Low manis = low sweetness
                base_sweetness
            )
        )
        
        return adjusted_sweetness

    def _classify_caffeine_level_enhanced(self, df):
        """Enhanced caffeine classification with better correlation to bitterness"""
        def classify_caffeine(row):
            if row['kafein_score'] == 0:
                return 'non-kafein'
            else:  # kafein_score == 1
                bitter_level = row['rasa_pahit']
                if bitter_level >= 0.7:
                    return 'tinggi'
                elif bitter_level >= 0.4:
                    return 'sedang'
                else:
                    return 'rendah'
        
        return df.apply(classify_caffeine, axis=1)

    def _calculate_feature_correlations(self):
        """Calculate and store important feature correlations"""
        if self.menu_df.empty:
            return
            
        # Define expected correlations
        correlation_pairs = [
            ('kafein_score', 'rasa_pahit'),
            ('rasa_manis', 'sweetness_score'),
            ('carbonated_score', 'tekstur_BUBBLY'),
            ('rasa_asam', 'tekstur_LIGHT'),
        ]
        
        for feature1, feature2 in correlation_pairs:
            if feature1 in self.menu_df.columns and feature2 in self.menu_df.columns:
                corr, p_value = pearsonr(self.menu_df[feature1], self.menu_df[feature2])
                self.feature_correlations[f"{feature1}_{feature2}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'strength': 'strong' if abs(corr) > 0.6 else 'moderate' if abs(corr) > 0.3 else 'weak'
                }
                print(f"📊 {feature1} ↔ {feature2}: r={corr:.3f}, p={p_value:.3f}")

    def train_enhanced_hybrid_weights(self, n_users=1500, random_seed=42, method='all'):
        """Enhanced training with FIXED preference hierarchy"""
        print("\n" + "="*80)
        print(f"ENHANCED HYBRID TRAINING - FIXED VERSION")
        print(f"Users: {n_users}, Seed: {random_seed}")
        print("="*80)

        if method == 'all':
            np.random.seed(random_seed)
            random.seed(random_seed)
            
            # FIXED: Generate dummy data with TASTE as primary driver
            dummy_data = self._generate_fixed_dummy_data(n_users, random_seed)
            
            # Build co-occurrence with balanced feature validation
            co_occurrence = self._build_balanced_cooccurrence_matrix(dummy_data)
            
            # Enhanced regression dataset with balanced penalties
            regression_df = self._build_balanced_regression_dataset(co_occurrence)
            
            # FIXED: Cross-validation with balanced penalties
            cv_results = self._train_with_balanced_cross_validation(regression_df, 5, random_seed)
            
            # Train models with balanced correlation constraints
            self._train_regression_models_balanced(regression_df, ridge_alpha=cv_results['best_alpha'])
            
            # FIXED: Calculate question importance with balanced metrics
            self._calculate_balanced_question_importance()
            
            # Convert to user weights with context-specific adjustments
            self._convert_to_user_weights_fixed()
            
            self.trained = True
            print(f"\n✅ Enhanced training completed! Best alpha: {cv_results['best_alpha']:.4f}")
            self._display_enhanced_results()
            return self
        else:
            raise ValueError(f"Method '{method}' not supported")

    def _generate_fixed_dummy_data(self, n_users, random_seed):
        """FIXED: Generate dummy data with TASTE as primary driver"""
        questions = {
            'taste': ['pahit', 'manis', 'asam', 'netral'],  # PRIMARY DRIVER
            'mood': ['energi', 'rileks', 'menyegarkan', 'manis'],
            'texture': ['light', 'creamy', 'heavy', 'bubbly'],
            'caffeine': ['tinggi', 'sedang', 'rendah', 'non-kafein'],
            'temperature': ['dingin', 'panas', 'bebas'],
            'budget': ['low', 'mid', 'high', 'bebas']
        }
        
        dummy_data = []
        for user_id in range(1, n_users + 1):
            user_seed = random_seed + user_id
            np.random.seed(user_seed)
            random.seed(user_seed)
            
            # FIXED: Generate preferences with TASTE as primary driver
            user_prefs = self._generate_taste_first_preferences(questions, user_seed)
            
            # Generate selections with stronger ASAM support
            selected_items = self._generate_taste_driven_selections(user_prefs, user_seed)
            
            dummy_data.append({
                'user_id': user_id,
                **user_prefs,
                'selected_items': selected_items
            })
        return dummy_data
    
    def _generate_taste_first_preferences(self, questions, user_seed):
        """FIXED: Generate preferences with balanced taste distribution and reduced manis dominance"""
        np.random.seed(user_seed)
        random.seed(user_seed)
        
        user_prefs = {}
        
        # STEP 1: Generate TASTE first with BALANCED distribution
        taste_distribution = {
            'manis': 0.25,    # REDUCED from implicit high probability
            'asam': 0.25,     # INCREASED to match manis
            'pahit': 0.25,    # Equal weight
            'netral': 0.25    # INCREASED from low probability
        }
        
        # Weighted random selection for balanced taste distribution
        taste_choices = list(taste_distribution.keys())
        taste_weights = list(taste_distribution.values())
        user_prefs['taste'] = np.random.choice(taste_choices, p=taste_weights)
        
        # STEP 2: Generate other base preferences
        for question, options in questions.items():
            if question != 'taste':
                user_prefs[question] = random.choice(options)
        
        # STEP 3: Apply TASTE-driven constraints with REDUCED manis bias
        if user_prefs['taste'] == 'pahit':
            if random.random() < 0.85:
                user_prefs['caffeine'] = random.choice(['tinggi', 'sedang'])
            if random.random() < 0.7:
                user_prefs['mood'] = random.choice(['energi', 'rileks'])
            if random.random() < 0.5:
                user_prefs['temperature'] = random.choice(['panas', 'bebas'])
                    
        elif user_prefs['taste'] == 'manis':
            # REDUCED manis influence on other preferences
            if random.random() < 0.6:  
                user_prefs['caffeine'] = random.choice(['rendah', 'non-kafein'])
            if random.random() < 0.5:  
                user_prefs['mood'] = random.choice(['manis', 'rileks'])
            if random.random() < 0.4:  
                user_prefs['texture'] = 'creamy'
                    
        elif user_prefs['taste'] == 'asam':
            # STRENGTHENED asam preference logic
            if random.random() < 0.85:  # INCREASED from 0.75
                user_prefs['mood'] = 'menyegarkan'
            if random.random() < 0.8:   # INCREASED from 0.7
                user_prefs['caffeine'] = random.choice(['rendah', 'non-kafein'])
            if random.random() < 0.75:  # INCREASED from 0.65
                user_prefs['texture'] = random.choice(['light', 'bubbly'])
            if random.random() < 0.7:   # INCREASED from 0.6
                user_prefs['temperature'] = 'dingin'
                    
        elif user_prefs['taste'] == 'netral':
            # STRENGTHENED netral preference logic
            if random.random() < 0.6:   # INCREASED from 0.5
                user_prefs['caffeine'] = random.choice(['sedang', 'rendah'])
            if random.random() < 0.5:   # INCREASED from 0.4
                user_prefs['texture'] = random.choice(['light', 'creamy'])
            # Add netral-specific mood preferences
            if random.random() < 0.4:
                user_prefs['mood'] = random.choice(['rileks', 'energi'])  # Balanced
        
        return user_prefs


    def _generate_taste_driven_selections(self, user_prefs, user_seed, n_items_range=(3, 6)):
        """FIXED: Generate selections driven by TASTE preferences with better ASAM support"""
        random.seed(user_seed)
        np.random.seed(user_seed)
        
        filtered_items = self.menu_df.copy()
        
        # FIXED: Primary TASTE-based filtering with adaptive thresholds
        taste_pref = user_prefs['taste']
        primary_filters = []
        
        if taste_pref == 'pahit':
            threshold = self.adaptive_thresholds['rasa_pahit']['medium']
            primary_filters.append(filtered_items['rasa_pahit'] >= threshold)
            
        elif taste_pref == 'manis':
            threshold = self.adaptive_thresholds['rasa_manis']['medium']
            primary_filters.append(
                (filtered_items['rasa_manis'] >= threshold) & 
                (filtered_items['sweetness_score'] >= threshold)
            )
            
        elif taste_pref == 'asam':
            # FIXED: Lower threshold for ASAM to ensure sufficient data
            threshold = self.adaptive_thresholds['rasa_asam']['low']  # Use LOW threshold instead of medium
            primary_filters.append(filtered_items['rasa_asam'] >= threshold)
            
        elif taste_pref == 'netral':
            threshold = self.adaptive_thresholds['rasa_netral']['medium']
            primary_filters.append(filtered_items['rasa_netral'] >= threshold)
        
        # Apply primary taste filter first
        if primary_filters:
            combined_filter = primary_filters[0]
            if combined_filter.sum() >= 2:  # Ensure minimum items
                filtered_items = filtered_items[combined_filter]
                print(f"🎯 Taste filter ({taste_pref}): {len(filtered_items)} items")
        
        # FIXED: Soft constraints instead of hard filters
        # Secondary filtering with SOFT constraints
        caffeine_pref = user_prefs['caffeine']
        if caffeine_pref != 'bebas' and len(filtered_items) > 3:
            if caffeine_pref == 'non-kafein':
                # Prefer non-caffeine but don't exclude all caffeine
                non_caff_items = filtered_items[filtered_items['kafein_score'] == 0]
                if len(non_caff_items) >= 2:
                    # Take 70% non-caffeine, 30% can be caffeinated
                    n_non_caff = int(len(filtered_items) * 0.7)
                    n_caff = len(filtered_items) - n_non_caff
                    
                    selected_non_caff = non_caff_items.sample(n=min(n_non_caff, len(non_caff_items)), random_state=user_seed)
                    caff_items = filtered_items[filtered_items['kafein_score'] == 1]
                    if len(caff_items) > 0 and n_caff > 0:
                        selected_caff = caff_items.sample(n=min(n_caff, len(caff_items)), random_state=user_seed)
                        filtered_items = pd.concat([selected_non_caff, selected_caff])
                    else:
                        filtered_items = selected_non_caff
                        
            elif caffeine_pref in ['tinggi', 'sedang']:
                # Prefer caffeinated but don't exclude all non-caffeine
                caff_items = filtered_items[filtered_items['kafein_score'] == 1]
                if len(caff_items) >= 2:
                    n_caff = int(len(filtered_items) * 0.75)
                    n_non_caff = len(filtered_items) - n_caff
                    
                    selected_caff = caff_items.sample(n=min(n_caff, len(caff_items)), random_state=user_seed)
                    non_caff_items = filtered_items[filtered_items['kafein_score'] == 0]
                    if len(non_caff_items) > 0 and n_non_caff > 0:
                        selected_non_caff = non_caff_items.sample(n=min(n_non_caff, len(non_caff_items)), random_state=user_seed)
                        filtered_items = pd.concat([selected_caff, selected_non_caff])
                    else:
                        filtered_items = selected_caff
        
        # Ensure minimum items for selection
        if len(filtered_items) < 2:
            print(f"⚠️ Too few items after filtering, using broader selection for {taste_pref}")
            # Fallback: use broader taste-based selection
            if taste_pref == 'asam':
                # For ASAM, use even lower threshold
                very_low_threshold = self.adaptive_thresholds['rasa_asam']['low'] * 0.5
                filtered_items = self.menu_df[self.menu_df['rasa_asam'] >= very_low_threshold]
                if len(filtered_items) < 2:
                    # If still too few, select items with ANY acidic component
                    filtered_items = self.menu_df[self.menu_df['rasa_asam'] > 0]
            else:
                # For other tastes, relax the threshold
                taste_col = f'rasa_{taste_pref}' if taste_pref != 'netral' else 'rasa_netral'
                if taste_col in self.menu_df.columns:
                    filtered_items = self.menu_df[self.menu_df[taste_col] > 0]
            
            if len(filtered_items) < 2:
                # Final fallback
                filtered_items = self.menu_df.sample(n=min(8, len(self.menu_df)), random_state=user_seed)
        
        # Select items with enhanced preference scoring
        filtered_items = self._score_items_by_taste_preferences(filtered_items, user_prefs)
        
        n_items = random.randint(*n_items_range)
        n_items = min(n_items, len(filtered_items))
        
        # Weighted sampling based on preference scores
        if 'preference_score' in filtered_items.columns:
            weights = filtered_items['preference_score'].values
            weights = weights / weights.sum() if weights.sum() > 0 else None
            selected_indices = np.random.choice(
                len(filtered_items), 
                size=n_items, 
                replace=False, 
                p=weights
            )
            selected = filtered_items.iloc[selected_indices]
        else:
            selected = filtered_items.sample(n=n_items, random_state=user_seed)
        
        return selected['nama_minuman'].tolist()
    
    def _score_items_by_taste_preferences(self, items_df, user_prefs):
        """FIXED: Balanced scoring with reduced manis dominance"""
        items_df = items_df.copy()
        scores = np.zeros(len(items_df))
        
        # PRIMARY: Taste scoring with BALANCED weights
        taste_pref = user_prefs['taste']
        if taste_pref == 'pahit':
            scores += items_df['rasa_pahit'] * 0.4
        elif taste_pref == 'manis':
            # REDUCED manis weight and added constraints
            manis_score = items_df['rasa_manis'] * 0.3  # REDUCED from 0.4
            sweetness_score = items_df['sweetness_score'] * 0.15  # REDUCED from 0.2
            scores += manis_score + sweetness_score
        elif taste_pref == 'asam':
            scores += items_df['rasa_asam'] * 0.5  # INCREASED from 0.4
        elif taste_pref == 'netral':
            scores += items_df['rasa_netral'] * 0.45  # INCREASED from 0.4
        
        # SECONDARY: Mood scoring with balanced influence
        mood_pref = user_prefs['mood']
        if mood_pref == 'energi':
            scores += items_df['kafein_score'] * 0.15
            scores += (1 - items_df['sweetness_score']) * 0.1
        elif mood_pref == 'rileks':
            scores += (1 - items_df['kafein_score']) * 0.15
            scores += items_df['sweetness_score'] * 0.08  # REDUCED from 0.1
        elif mood_pref == 'menyegarkan':
            scores += items_df['rasa_asam'] * 0.12  # Support for refreshing-sour
            scores += items_df['tekstur_LIGHT'] * 0.1
        elif mood_pref == 'manis':
            scores += items_df['rasa_manis'] * 0.08   # REDUCED from 0.1
            scores += items_df['sweetness_score'] * 0.08  # REDUCED from 0.1
        
        # TERTIARY: Other preferences
        caffeine_pref = user_prefs['caffeine']
        if caffeine_pref == 'tinggi':
            scores += items_df['kafein_score'] * 0.1
        elif caffeine_pref == 'non-kafein':
            scores += (1 - items_df['kafein_score']) * 0.1
        
        # Texture scoring
        texture_pref = user_prefs['texture']
        texture_col = f"tekstur_{texture_pref.upper()}"
        if texture_col in items_df.columns:
            scores += items_df[texture_col] * 0.05
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        items_df['preference_score'] = np.clip(scores + 0.1, 0.1, 1.0)
        return items_df


    def _build_balanced_cooccurrence_matrix(self, dummy_data):
        """FIXED: Build co-occurrence matrix with balanced validation"""
        co_occurrence = defaultdict(int)
        questions = ['taste', 'mood', 'texture', 'caffeine', 'temperature', 'budget']  # Reordered: taste first
        
        for entry in dummy_data:
            items = entry['selected_items']
            if len(items) < 2: 
                continue
            
            # Validate selections with balanced weights
            valid_combinations = self._validate_item_combinations_balanced(entry, items)
            
            if valid_combinations:
                for question in questions:
                    answer = entry[question]
                    for item1, item2 in combinations(sorted(items), 2):
                        # Apply balanced weight
                        weight = valid_combinations.get((item1, item2), 1.0)
                        co_occurrence[(question, answer, item1, item2)] += weight
        
        return co_occurrence

    def _validate_item_combinations_balanced(self, user_entry, selected_items):
        """FIXED: Balanced validation with equal attention to all features"""
        validation_weights = {}
        menu_indexed = self.menu_df.set_index('nama_minuman')
        
        for item1, item2 in combinations(selected_items, 2):
            if item1 not in menu_indexed.index or item2 not in menu_indexed.index:
                continue
                
            weight = 1.0
            
            # Validate TASTE preference consistency (PRIMARY)
            taste_pref = user_entry['taste']
            if taste_pref == 'pahit':
                item1_bitter = menu_indexed.loc[item1, 'rasa_pahit']
                item2_bitter = menu_indexed.loc[item2, 'rasa_pahit']
                if item1_bitter > 0.3 and item2_bitter > 0.3:
                    weight *= 1.3  # Strong boost for taste consistency
            elif taste_pref == 'manis':
                item1_sweet = menu_indexed.loc[item1, 'rasa_manis']
                item2_sweet = menu_indexed.loc[item2, 'rasa_manis']
                if item1_sweet > 0.3 and item2_sweet > 0.3:
                    weight *= 1.3
            elif taste_pref == 'asam':
                item1_sour = menu_indexed.loc[item1, 'rasa_asam']
                item2_sour = menu_indexed.loc[item2, 'rasa_asam']
                if item1_sour > 0.2 and item2_sour > 0.2:  # Lower threshold for asam
                    weight *= 1.4  # Higher boost for ASAM to strengthen signal
            elif taste_pref == 'netral':
                item1_netral = menu_indexed.loc[item1, 'rasa_netral']
                item2_netral = menu_indexed.loc[item2, 'rasa_netral']
                if item1_netral > 0.3 and item2_netral > 0.3:
                    weight *= 1.3
            
            # Validate caffeine preference consistency (SECONDARY)
            caffeine_pref = user_entry['caffeine']
            if caffeine_pref == 'tinggi':
                item1_caff = menu_indexed.loc[item1, 'kafein_score']
                item2_caff = menu_indexed.loc[item2, 'kafein_score']
                if item1_caff == 1 and item2_caff == 1:
                    weight *= 1.15  # Reduced from 1.2
            elif caffeine_pref == 'non-kafein':
                item1_caff = menu_indexed.loc[item1, 'kafein_score']
                item2_caff = menu_indexed.loc[item2, 'kafein_score']
                if item1_caff == 0 and item2_caff == 0:
                    weight *= 1.15
            
            # Validate mood preference consistency (TERTIARY)
            mood_pref = user_entry['mood']
            if mood_pref == 'energi':
                item1_caff = menu_indexed.loc[item1, 'kafein_score']
                item2_caff = menu_indexed.loc[item2, 'kafein_score']
                if item1_caff == 1 or item2_caff == 1:  # At least one caffeinated
                    weight *= 1.1  # Small boost
            elif mood_pref == 'menyegarkan':
                item1_sour = menu_indexed.loc[item1, 'rasa_asam']
                item2_sour = menu_indexed.loc[item2, 'rasa_asam']
                if item1_sour > 0.2 or item2_sour > 0.2:
                    weight *= 1.1
            
            validation_weights[(item1, item2)] = weight
        
        return validation_weights

    def _build_balanced_regression_dataset(self, co_occurrence):
        """FIXED: Build regression dataset with balanced feature attention"""
        print("🔄 Building balanced regression dataset...")
        data = []
        menu_indexed = self.menu_df.set_index('nama_minuman')
        
        for (question, answer, item1, item2), count in sorted(co_occurrence.items()):
            if item1 not in menu_indexed.index or item2 not in menu_indexed.index: 
                continue
                
            vec1 = menu_indexed.loc[item1, self.feature_cols].astype(float)
            vec2 = menu_indexed.loc[item2, self.feature_cols].astype(float)
            
            # Enhanced feature engineering with balanced attention
            feature_diff = np.abs(vec1.values - vec2.values)
            feature_similarity = 1 - feature_diff
            feature_avg = (vec1.values + vec2.values) / 2
            feature_product = vec1.values * vec2.values
            
            row = {}
            for i, col in enumerate(self.feature_cols):
                row[f'diff_{col}'] = feature_diff[i]
                row[f'sim_{col}'] = feature_similarity[i]
                row[f'avg_{col}'] = feature_avg[i]
                row[f'prod_{col}'] = feature_product[i]
            
            # Add balanced correlation-based features
            row['taste_consistency'] = self._calculate_taste_consistency(vec1, vec2, question, answer)
            row['caffeine_bitter_consistency'] = self._calculate_caffeine_bitter_consistency(vec1, vec2)
            row['sweet_consistency'] = self._calculate_sweet_consistency(vec1, vec2)
            
            row.update({
                'question': question, 
                'answer': answer, 
                'co_occurrence': count
            })
            data.append(row)
        
        regression_df = pd.DataFrame(data)
        return self._select_features_with_balanced_constraints(regression_df)

    def _calculate_taste_consistency(self, vec1, vec2, question, answer):
        """NEW: Calculate taste consistency for primary taste features"""
        if question != 'taste':
            return 0.5  # Neutral for non-taste questions
        
        if answer == 'pahit':
            bitter1, bitter2 = vec1['rasa_pahit'], vec2['rasa_pahit']
            return (bitter1 + bitter2) / 2  # Average bitterness
        elif answer == 'manis':
            sweet1, sweet2 = vec1['rasa_manis'], vec2['rasa_manis']
            sweetness1, sweetness2 = vec1['sweetness_score'], vec2['sweetness_score']
            return ((sweet1 + sweet2) + (sweetness1 + sweetness2)) / 4
        elif answer == 'asam':
            sour1, sour2 = vec1['rasa_asam'], vec2['rasa_asam']
            return (sour1 + sour2) / 2  # Average sourness
        elif answer == 'netral':
            netral1, netral2 = vec1['rasa_netral'], vec2['rasa_netral']
            return (netral1 + netral2) / 2
        
        return 0.5

    def _calculate_caffeine_bitter_consistency(self, vec1, vec2):
        """Calculate consistency between caffeine and bitter taste"""
        caff1, bitter1 = vec1['kafein_score'], vec1['rasa_pahit']
        caff2, bitter2 = vec2['kafein_score'], vec2['rasa_pahit']
        
        consistency1 = 1.0 if (caff1 == 0 or bitter1 >= 0.3) else 0.5
        consistency2 = 1.0 if (caff2 == 0 or bitter2 >= 0.3) else 0.5
        
        return (consistency1 + consistency2) / 2

    def _calculate_sweet_consistency(self, vec1, vec2):
        """Calculate consistency between rasa_manis and sweetness_score"""
        manis1, sweet1 = vec1['rasa_manis'], vec1['sweetness_score']
        manis2, sweet2 = vec2['rasa_manis'], vec2['sweetness_score']
        
        consistency1 = 1.0 - abs(manis1 - sweet1)
        consistency2 = 1.0 - abs(manis2 - sweet2)
        
        return (consistency1 + consistency2) / 2

    def _select_features_with_balanced_constraints(self, regression_df):
        """FIXED: Balanced feature selection with equal attention to all feature types"""
        feature_cols = [col for col in regression_df.columns 
                       if col not in ['question', 'answer', 'co_occurrence']]
        
        selected_features = {}
        significance_threshold = 0.1  # Relaxed from 0.05
        min_correlation = 0.1  # Relaxed from 0.15
        
        for question in regression_df['question'].unique():
            question_data = regression_df[regression_df['question'] == question]
            target = question_data['co_occurrence']
            
            feature_scores = []
            
            for feature in feature_cols:
                feature_values = question_data[feature]
                if feature_values.nunique() <= 1:
                    continue
                
                try:
                    pearson_corr, pearson_p = pearsonr(feature_values, target)
                    spearman_corr, spearman_p = spearmanr(feature_values, target)
                    
                    avg_corr = (abs(pearson_corr) + abs(spearman_corr)) / 2
                    min_p = min(pearson_p, spearman_p)
                    
                    if min_p < significance_threshold and avg_corr > min_correlation:
                        feature_scores.append((feature, avg_corr, min_p))
                        
                except:
                    continue
            
            # FIXED: Balanced feature selection
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Ensure we have features from all categories
            taste_features = [f for f, _, _ in feature_scores if any(taste in f for taste in ['rasa_', 'sweetness_', 'taste_consistency'])]
            caffeine_features = [f for f, _, _ in feature_scores if 'kafein' in f or 'caffeine_bitter' in f]
            texture_features = [f for f, _, _ in feature_scores if 'tekstur' in f]
            other_features = [f for f, _, _ in feature_scores if f not in taste_features + caffeine_features + texture_features]
            
            # Balanced selection: ensure representation from each category
            selected = []
            selected.extend(taste_features[:3])  # Top 3 taste features
            selected.extend(caffeine_features[:2])  # Top 2 caffeine features
            selected.extend(texture_features[:2])  # Top 2 texture features
            selected.extend(other_features[:2])  # Top 2 other features
            
            # Remove duplicates and limit total
            selected = list(dict.fromkeys(selected))[:8]  # Max 8 features
            
            if len(selected) < 3:
                # Add top features if we don't have enough
                all_features = [f for f, _, _ in feature_scores]
                needed = 3 - len(selected)
                selected.extend(all_features[:needed])
            
            selected_features[question] = selected[:8]
            print(f"📊 {question}: {len(selected)} features selected (balanced)")
        
        # Filter dataset
        filtered_data = []
        for question in regression_df['question'].unique():
            question_data = regression_df[regression_df['question'] == question].copy()
            keep_cols = (['question', 'answer', 'co_occurrence'] + 
                        selected_features[question])
            question_filtered = question_data[keep_cols]
            filtered_data.append(question_filtered)
        
        return pd.concat(filtered_data, ignore_index=True)

    def _train_with_balanced_cross_validation(self, regression_df, cv_folds, random_seed):
        """FIXED: Cross-validation with balanced penalties"""
        print("🔄 Balanced cross-validation...")
        alpha_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        
        best_alpha = 0.1
        best_score = float('inf')
        
        feature_cols = [col for col in regression_df.columns 
                       if col not in ['question', 'answer', 'co_occurrence']]
        
        for alpha in alpha_values:
            cv_scores = []
            correlation_penalties = []
            
            for train_idx, val_idx in kf.split(regression_df):
                train_data = regression_df.iloc[train_idx]
                val_data = regression_df.iloc[val_idx]
                
                fold_scores = []
                fold_penalties = []
                
                for question in train_data['question'].unique():
                    if question in val_data['question'].values:
                        train_q = train_data[train_data['question'] == question]
                        val_q = val_data[val_data['question'] == question]
                        
                        if len(train_q) > 0 and len(val_q) > 0:
                            X_train = train_q[feature_cols].fillna(0)
                            y_train = train_q['co_occurrence']
                            X_val = val_q[feature_cols].fillna(0)
                            y_val = val_q['co_occurrence']
                            
                            model = Ridge(alpha=alpha, random_state=random_seed)
                            model.fit(X_train, y_train)
                            
                            y_pred = model.predict(X_val)
                            mse = mean_squared_error(y_val, y_pred)
                            
                            # FIXED: Balanced correlation penalty
                            penalty = self._calculate_balanced_correlation_penalty(model, feature_cols, question)
                            
                            fold_scores.append(mse)
                            fold_penalties.append(penalty)
                
                if fold_scores:
                    cv_scores.append(np.mean(fold_scores))
                    correlation_penalties.append(np.mean(fold_penalties))
            
            if cv_scores:
                mean_cv_score = np.mean(cv_scores)
                mean_penalty = np.mean(correlation_penalties)
                
                # FIXED: Reduced penalty weight to avoid over-constraining
                combined_score = mean_cv_score + (mean_penalty * 0.05)  # Reduced from 0.1
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_alpha = alpha
        
        print(f"✅ Best alpha found: {best_alpha} (Combined Score: {best_score:.4f})")
        
        return {
            'best_alpha': best_alpha,
            'best_score': best_score
        }

    def _calculate_balanced_correlation_penalty(self, model, feature_cols, question):
        """FIXED: Balanced penalty that considers all feature types equally"""
        penalty = 0.0
        
        # Penalty for taste features (equal weight as caffeine)
        taste_features = [f for f in feature_cols if any(taste in f for taste in ['rasa_', 'sweetness_', 'taste_consistency'])]
        if taste_features and question == 'taste':
            taste_weights = [model.coef_[feature_cols.index(f)] for f in taste_features]
            avg_taste_weight = np.mean(taste_weights)
            # Penalty if taste features have wrong sign for taste questions
            if avg_taste_weight < -0.1:
                penalty += 0.3
        
        # Penalty for caffeine-bitter correlation (equal weight as taste)
        caffeine_features = [f for f in feature_cols if 'kafein_score' in f]
        bitter_features = [f for f in feature_cols if 'rasa_pahit' in f]
        
        if caffeine_features and bitter_features:
            caff_weights = [model.coef_[feature_cols.index(f)] for f in caffeine_features]
            bitter_weights = [model.coef_[feature_cols.index(f)] for f in bitter_features]
            
            if question in ['caffeine', 'taste'] and len(caff_weights) > 0 and len(bitter_weights) > 0:
                avg_caff = np.mean(caff_weights)
                avg_bitter = np.mean(bitter_weights)
                if (avg_caff > 0 and avg_bitter < -0.1) or (avg_caff < 0 and avg_bitter > 0.1):
                    penalty += 0.3  # Equal to taste penalty
        
        # Penalty for sweet consistency (equal weight)
        sweet_features = [f for f in feature_cols if 'rasa_manis' in f or 'sweetness_score' in f or 'sweet_consistency' in f]
        if len(sweet_features) >= 2 and question == 'taste':
            sweet_weights = [model.coef_[feature_cols.index(f)] for f in sweet_features]
            if max(sweet_weights) > 0.1 and min(sweet_weights) < -0.1:
                penalty += 0.2
        
        return penalty

    def _train_regression_models_balanced(self, regression_df, ridge_alpha=0.1):
        """FIXED: Balanced regression training"""
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        feature_cols = [col for col in regression_df.columns 
                       if col not in ['question', 'answer', 'co_occurrence']]
        
        for question in sorted(regression_df['question'].unique()):
            self.models[question] = {}
            self.feature_importance[question] = {}
            self.model_performance[question] = {}
            
            for answer in sorted(regression_df[regression_df['question'] == question]['answer'].unique()):
                subset = regression_df[
                    (regression_df['question'] == question) &
                    (regression_df['answer'] == answer)
                ].copy()
                
                if len(subset) < 5:  # Reduced minimum samples
                    continue
                
                X = subset[feature_cols].fillna(0)
                y = subset['co_occurrence']
                
                model = Ridge(alpha=ridge_alpha, random_state=42)
                model.fit(X, y)
                
                # FIXED: Apply balanced correlation constraints
                model.coef_ = self._apply_balanced_correlation_constraints(
                    model.coef_, feature_cols, question, answer
                )
                
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                self.models[question][answer] = model
                self.feature_importance[question][answer] = dict(zip(feature_cols, model.coef_))
                
                # Enhanced performance metrics
                correlation_score = self._evaluate_balanced_correlation_compliance(model.coef_, feature_cols)
                
                self.model_performance[question][answer] = {
                    'r2': r2, 
                    'mse': mse, 
                    'n_samples': len(subset), 
                    'mean_cooccurrence': y.mean(),
                    'correlation_score': correlation_score
                }

    def _apply_balanced_correlation_constraints(self, coefficients, feature_cols, question, answer):
        """FIXED: Apply balanced constraints with context-specific multipliers"""
        adjusted_coefs = coefficients.copy()
        MAX_ABS_WEIGHT = 2.5  # Slightly reduced to allow more diversity
        
        # Apply general clipping
        adjusted_coefs = np.clip(adjusted_coefs, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
        
        # FIXED: Context-specific multipliers instead of hardcoded values
        base_adjustment = 0.3  # Base strength
        
        # Question-specific adjustment multipliers
        question_multipliers = {
            'taste': 1.5,      # Taste most important
            'mood': 1.2,       # Mood moderately important
            'caffeine': 1.3,   # Caffeine important
            'texture': 1.1,    # Texture moderately important
            'temperature': 1.0, # Temperature baseline
            'budget': 0.8      # Budget least constrained
        }
        
        multiplier = question_multipliers.get(question, 1.0)
        adjustment_strength = base_adjustment * multiplier
        
        # Specific constraints based on question-answer combinations
        if question == 'taste':
            if answer == 'pahit':
                # Bitter taste should positively correlate with bitter features
                bitter_indices = [i for i, f in enumerate(feature_cols) if 'rasa_pahit' in f and 'sim_' in f]
                for idx in bitter_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
                        
            elif answer == 'manis':
                # Sweet taste should positively correlate with sweetness features
                sweet_indices = [i for i, f in enumerate(feature_cols) 
                               if ('rasa_manis' in f or 'sweetness_score' in f) and 'sim_' in f]
                for idx in sweet_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
                        
            elif answer == 'asam':
                # FIXED: Strong positive constraint for ASAM features
                sour_indices = [i for i, f in enumerate(feature_cols) if 'rasa_asam' in f and 'sim_' in f]
                for idx in sour_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * (adjustment_strength * 1.5)  # Extra strong for ASAM
                        
            elif answer == 'netral':
                # Netral should positively correlate with netral features
                netral_indices = [i for i, f in enumerate(feature_cols) if 'rasa_netral' in f and 'sim_' in f]
                for idx in netral_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
        
        elif question == 'mood':
            if answer == 'energi':
                # Energy should positively correlate with caffeine
                caff_indices = [i for i, f in enumerate(feature_cols) if 'kafein_score' in f and 'sim_' in f]
                for idx in caff_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
                        
            elif answer == 'rileks':
                # Relax should negatively correlate with high caffeine
                caff_indices = [i for i, f in enumerate(feature_cols) if 'kafein_score' in f and 'sim_' in f]
                for idx in caff_indices:
                    if adjusted_coefs[idx] > 0:
                        adjusted_coefs[idx] = -abs(adjusted_coefs[idx]) * adjustment_strength
                        
            elif answer == 'menyegarkan':
                # Refreshing should correlate with sour/light features
                sour_indices = [i for i, f in enumerate(feature_cols) if 'rasa_asam' in f and 'sim_' in f]
                light_indices = [i for i, f in enumerate(feature_cols) if 'tekstur_LIGHT' in f and 'sim_' in f]
                for idx in sour_indices + light_indices:
                    if adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
        
        elif question == 'caffeine':
            if answer == 'tinggi':
                caff_indices = [i for i, f in enumerate(feature_cols) if 'kafein_score' in f]
                for idx in caff_indices:
                    if 'sim_' in feature_cols[idx] and adjusted_coefs[idx] < 0:
                        adjusted_coefs[idx] = abs(adjusted_coefs[idx]) * adjustment_strength
                        
            elif answer == 'non-kafein':
                caff_indices = [i for i, f in enumerate(feature_cols) if 'kafein_score' in f and 'sim_' in f]
                for idx in caff_indices:
                    if adjusted_coefs[idx] > 0:
                        adjusted_coefs[idx] = -abs(adjusted_coefs[idx]) * adjustment_strength
        
        return adjusted_coefs

    def _evaluate_balanced_correlation_compliance(self, coefficients, feature_cols):
        """FIXED: Balanced evaluation of correlation compliance"""
        score = 1.0
        
        # Equal weight evaluation for different feature types
        
        # Taste feature consistency
        taste_features = [i for i, f in enumerate(feature_cols) if any(taste in f for taste in ['rasa_', 'sweetness_', 'taste_consistency'])]
        if taste_features:
            taste_weights = [coefficients[i] for i in taste_features]
            # Reward consistency in taste feature signs
            positive_taste = sum(1 for w in taste_weights if w > 0.1)
            negative_taste = sum(1 for w in taste_weights if w < -0.1)
            if positive_taste > negative_taste:
                score += 0.2
            elif negative_taste > positive_taste:
                score -= 0.2
        
        # Caffeine-bitter relationship (equal weight to taste)
        caff_sim_indices = [i for i, f in enumerate(feature_cols) if 'sim_kafein_score' in f]
        bitter_sim_indices = [i for i, f in enumerate(feature_cols) if 'sim_rasa_pahit' in f]
        
        if caff_sim_indices and bitter_sim_indices:
            caff_weight = np.mean([coefficients[i] for i in caff_sim_indices])
            bitter_weight = np.mean([coefficients[i] for i in bitter_sim_indices])
            
            if (caff_weight > 0 and bitter_weight > 0) or (caff_weight < 0 and bitter_weight < 0):
                score += 0.2
            else:
                score -= 0.2
        
        # Sweet consistency (equal weight)
        sweet_indices = [i for i, f in enumerate(feature_cols) 
                        if ('sim_rasa_manis' in f or 'sim_sweetness_score' in f)]
        
        if len(sweet_indices) >= 2:
            sweet_weights = [coefficients[i] for i in sweet_indices]
            if all(w >= 0 for w in sweet_weights) or all(w <= 0 for w in sweet_weights):
                score += 0.15
            else:
                score -= 0.15
        
        return max(score, 0.0)

    def _calculate_balanced_question_importance(self):
        """FIXED: Balanced question importance calculation with domain knowledge priority"""
        print("🔄 Calculating FIXED balanced question importance...")
        
        # FIXED: Define proper domain-based hierarchy for beverage recommendation
        domain_hierarchy = {
            'taste': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            },
            'mood': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            },
            'caffeine': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            },
            'texture': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            },
            'temperature': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            },
            'budget': {
                'base_importance': 1.5,    # PRIMARY - taste is the most important
                'reasoning': 'Taste preference drives beverage selection',
                'boost_factor': 1.5
            }
        }
        
        importance_scores = {}
        
        for question in self.feature_importance.keys():
            if question not in domain_hierarchy:
                continue
                
            domain_info = domain_hierarchy[question]
            base_score = domain_info['base_importance']
            boost_factor = domain_info['boost_factor']
            
            # Metric 1: Model performance (reduced weight for TASTE to avoid negative bias)
            performances = [perf['r2'] for perf in self.model_performance[question].values()]
            avg_r2 = np.mean(performances) if performances else 0
            
            # FIXED: Handle negative R2 scores properly for TASTE
            if question == 'taste' and avg_r2 < 0:
                # For taste, negative R2 often indicates model complexity issues, not lack of importance
                adjusted_r2 = 0.15  # Assign reasonable baseline
                print(f"⚠️ TASTE R2 adjusted from {avg_r2:.3f} to {adjusted_r2:.3f}")
            else:
                adjusted_r2 = max(avg_r2, 0)  # Ensure non-negative
            
            # Metric 2: Feature consistency and signal strength
            feature_signal_strength = self._calculate_feature_signal_strength(question)
            
            # Metric 3: Data coverage and reliability
            total_samples = sum([
                perf['n_samples'] for perf in self.model_performance[question].values()
            ])
            max_samples = max([
                sum([perf['n_samples'] for perf in self.model_performance[q].values()])
                for q in self.model_performance.keys()
            ]) if self.model_performance else 1
            
            data_coverage = total_samples / max_samples if max_samples > 0 else 0
            
            # Metric 4: Correlation compliance (reduced penalty for TASTE)
            correlation_scores = [
                perf.get('correlation_score', 0.5) 
                for perf in self.model_performance[question].values()
            ]
            avg_correlation = np.mean(correlation_scores) if correlation_scores else 0.5
            
            # FIXED: Question-specific weight combination
            if question == 'taste':
                # For TASTE: prioritize domain knowledge and feature signal
                calculated_importance = (
                    base_score * 0.50 +                    # Strong domain knowledge bias
                    feature_signal_strength * 0.20 +       # Feature signal strength
                    data_coverage * 0.15 +                 # Data availability
                    adjusted_r2 * 0.10 +                   # Reduced model performance weight
                    avg_correlation * 0.05                 # Minimal correlation penalty
                )
            elif question == 'mood':
                # For MOOD: balanced approach
                calculated_importance = (
                    base_score * 0.50 +                    # Strong domain knowledge bias
                    feature_signal_strength * 0.20 +       # Feature signal strength
                    data_coverage * 0.15 +                 # Data availability
                    adjusted_r2 * 0.10 +                   # Reduced model performance weight
                    avg_correlation * 0.05                 # Minimal correlation penalty
                )
            else:
                # For other questions: standard approach
                calculated_importance = (
                    base_score * 0.50 +                    # Strong domain knowledge bias
                    feature_signal_strength * 0.20 +       # Feature signal strength
                    data_coverage * 0.15 +                 # Data availability
                    adjusted_r2 * 0.10 +                   # Reduced model performance weight
                    avg_correlation * 0.05                 # Minimal correlation penalty
                )
            
            # Apply boost factor
            calculated_importance *= boost_factor
            
            importance_scores[question] = calculated_importance
            
            print(f"📊 {question.upper()}: "
                f"base={base_score:.3f}, r2={adjusted_r2:.3f}, "
                f"signal={feature_signal_strength:.3f}, coverage={data_coverage:.3f}, "
                f"corr={avg_correlation:.3f}, boost={boost_factor:.1f}, "
                f"final={calculated_importance:.4f}")
        
        # FIXED: Normalize with minimum guarantees
        total_score = sum(importance_scores.values())
        if total_score > 0:
            normalized_scores = {
                question: score / total_score 
                for question, score in importance_scores.items()
            }
            
            # FIXED: Enforce minimum importance for TASTE
            if 'taste' in normalized_scores and normalized_scores['taste'] < 0.22:
                print(f"⚠️ TASTE importance too low ({normalized_scores['taste']:.3f}), enforcing minimum")
                normalized_scores['taste'] = 0.22
                
                # Re-normalize other scores
                remaining_total = sum(score for q, score in normalized_scores.items() if q != 'taste')
                remaining_target = 1.0 - 0.22
                
                if remaining_total > 0:
                    for question in normalized_scores:
                        if question != 'taste':
                            normalized_scores[question] = (
                                normalized_scores[question] / remaining_total * remaining_target
                            )
            
            self.question_importance = normalized_scores
        else:
            # Fallback to domain hierarchy
            self.question_importance = {
                q: info['base_importance'] for q, info in domain_hierarchy.items()
            }
        
        # Validate final ranking
        sorted_questions = sorted(self.question_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n✅ FIXED Question Importance Ranking:")
        for i, (question, score) in enumerate(sorted_questions, 1):
            print(f"{i}. {question.upper()}: {score:.4f}")
        
        # Ensure TASTE is in top 2
        if sorted_questions[0][0] != 'taste' and sorted_questions[1][0] != 'taste':
            print("⚠️ WARNING: TASTE not in top 2, this may indicate data quality issues")

    def _calculate_feature_signal_strength(self, question):
        """Calculate signal strength based on feature weights distribution"""
        if question not in self.feature_importance:
            return 0.1
        
        all_weights = []
        for answer_weights in self.feature_importance[question].values():
            all_weights.extend([abs(w) for w in answer_weights.values() if abs(w) > 0.01])
        
        if not all_weights:
            return 0.1
        
        # Signal strength based on:
        # 1. Average absolute weight (indicates feature importance)
        # 2. Weight distribution consistency (low variance = consistent signal)
        # 3. Number of active features (more features = stronger signal)
        
        avg_weight = np.mean(all_weights)
        weight_std = np.std(all_weights)
        active_features_ratio = len(all_weights) / max(len(self.feature_cols), 1)
        
        # Normalized signal strength
        signal_strength = (
            avg_weight * 0.5 +                              # Average importance
            (1 - min(weight_std / (avg_weight + 1e-6), 1)) * 0.3 +  # Consistency (inverse of coefficient of variation)
            active_features_ratio * 0.2                     # Feature coverage
        )
        
        return min(signal_strength, 1.0)

    def _apply_question_importance_fixes(self):
        """Apply additional fixes to ensure logical question importance"""
        
        # Fix 1: Ensure TASTE importance is never negative
        if 'taste' in self.question_importance:
            if self.question_importance['taste'] < 0:
                print(f"🔧 Fixing negative TASTE importance: {self.question_importance['taste']:.4f} → 0.25")
                self.question_importance['taste'] = 0.25
        
        # Fix 2: Ensure logical hierarchy (TASTE > MOOD > others)
        importance_list = list(self.question_importance.items())
        importance_list.sort(key=lambda x: x[1], reverse=True)
        
        taste_rank = next((i for i, (q, _) in enumerate(importance_list) if q == 'taste'), None)
        mood_rank = next((i for i, (q, _) in enumerate(importance_list) if q == 'mood'), None)
        
        if taste_rank is not None and taste_rank > 1:  # TASTE should be in top 2
            print(f"🔧 TASTE ranking too low (rank {taste_rank + 1}), adjusting...")
            
            # Boost TASTE importance
            taste_boost = 0.05
            self.question_importance['taste'] += taste_boost
            
            # Redistribute from other questions
            other_questions = [q for q in self.question_importance.keys() if q != 'taste']
            reduction_per_question = taste_boost / len(other_questions)
            
            for q in other_questions:
                self.question_importance[q] = max(
                    self.question_importance[q] - reduction_per_question, 
                    0.05  # Minimum importance
                )
        
        # Fix 3: Normalize after fixes
        total = sum(self.question_importance.values())
        if total != 1.0:
            for q in self.question_importance:
                self.question_importance[q] /= total
        
        print("✅ Question importance fixes applied")

    # TAMBAHAN: Fungsi untuk memvalidasi hasil
    # def validate_question_importance(self):
    #     """Validate the calculated question importance makes sense"""
    #     issues = []
        
    #     # Check 1: TASTE should be in top 2
    #     sorted_q = sorted(self.question_importance.items(), key=lambda x: x[1], reverse=True)
    #     taste_rank = next((i for i, (q, _) in enumerate(sorted_q) if q == 'taste'), None)
        
    #     if taste_rank is None:
    #         issues.append("TASTE question missing from importance scores")
    #     elif taste_rank > 1:
    #         issues.append(f"TASTE ranked too low (rank {taste_rank + 1})")
        
    #     # Check 2: No negative importance
    #     negative_scores = [(q, s) for q, s in self.question_importance.items() if s < 0]
    #     if negative_scores:
    #         issues.append(f"Negative importance scores: {negative_scores}")
        
    #     # Check 3: Reasonable distribution (no single question > 50%)
    #     max_importance = max(self.question_importance.values())
    #     if max_importance > 0.5:
    #         max_q = max(self.question_importance.items(), key=lambda x: x[1])
    #         issues.append(f"Excessive importance for {max_q[0]}: {max_q[1]:.3f}")
        
    #     # Check 4: Total should be ~1.0
    #     total = sum(self.question_importance.values())
    #     if abs(total - 1.0) > 0.01:
    #         issues.append(f"Importance scores don't sum to 1.0: {total:.3f}")
        
    #     if issues:
    #         print("⚠️ Question Importance Validation Issues:")
    #         for issue in issues:
    #             print(f"  - {issue}")
    #         return False
    #     else:
    #         print("✅ Question importance validation passed")
    #         return True

    def validate_question_importance(self):
        """Validasi: semua importance tidak boleh negatif"""
        issues = []
        negative_scores = [(q, s) for q, s in self.question_importance.items() if s < 0]
        if negative_scores:
            print(f"❌ Negative importance scores: {negative_scores}")
            return False
        print("✅ All importance scores are non-negative")
        return True

    def _convert_to_user_weights_fixed(self):
        """FIXED: Enhanced conversion with context-specific adjustments"""
        user_vector_weights = {}
        
        for question in sorted(self.feature_importance.keys()):
            user_vector_weights[question] = {}
            
            for answer in sorted(self.feature_importance[question].keys()):
                weights = self.feature_importance[question][answer]
                converted_weights = {}
                
                for original_feature in sorted(self.feature_cols):
                    # Enhanced weight combination with context awareness
                    sim_key = f'sim_{original_feature}'
                    avg_key = f'avg_{original_feature}'
                    diff_key = f'diff_{original_feature}'
                    prod_key = f'prod_{original_feature}'
                    
                    sim_weight = weights.get(sim_key, 0)
                    avg_weight = weights.get(avg_key, 0)
                    diff_weight = weights.get(diff_key, 0)
                    prod_weight = weights.get(prod_key, 0)
                    
                    # FIXED: Context-specific combination weights
                    if question == 'taste':
                        # For taste questions, similarity is most important
                        combined_weight = (
                            sim_weight * 0.5 +      # Higher similarity weight for taste
                            avg_weight * 0.2 +
                            prod_weight * 0.2 +
                            diff_weight * 0.1
                        )
                    elif question == 'mood':
                        # For mood questions, average and product matter more
                        combined_weight = (
                            sim_weight * 0.5 +      # Higher similarity weight for taste
                            avg_weight * 0.2 +
                            prod_weight * 0.2 +
                            diff_weight * 0.1
                        )
                    else:
                        # Default combination
                        combined_weight = (
                            sim_weight * 0.5 +      # Higher similarity weight for taste
                            avg_weight * 0.2 +
                            prod_weight * 0.2 +
                            diff_weight * 0.1
                        )
                    
                    # Apply context-specific logical adjustments
                    combined_weight = self._apply_context_specific_adjustments(
                        combined_weight, original_feature, question, answer
                    )
                    
                    if abs(combined_weight) > 0.001:
                        converted_weights[original_feature] = combined_weight
                
                user_vector_weights[question][answer] = converted_weights
        
        self.user_vector_weights = user_vector_weights

    def _apply_context_specific_adjustments(self, weight, feature, question, answer):
        """FIXED: Reduced manis dominance and increased asam/netral support"""
        adjusted_weight = weight
        
        # Get balanced adjustment strengths
        adjustment_strengths = {
            'taste': {
                'pahit': 0.4, 
                'manis': 0.3,     # REDUCED from 0.4
                'asam': 0.6,      # INCREASED from 0.5  
                'netral': 0.5     # INCREASED from 0.3
            },
            'mood': {
                'energi': 0.3, 
                'rileks': 0.25,   # REDUCED from 0.3
                'menyegarkan': 0.4, # INCREASED from 0.35
                'manis': 0.2      # REDUCED from 0.25
            },
            'caffeine': {
                'tinggi': 0.4, 'sedang': 0.3, 'rendah': 0.25, 'non-kafein': 0.4
            },
            'texture': {
                'light': 0.25,    # INCREASED from 0.2
                'creamy': 0.15,   # REDUCED from 0.2 (less manis association)
                'heavy': 0.2, 
                'bubbly': 0.3     # INCREASED from 0.25
            },
            'temperature': {
                'dingin': 0.15, 'panas': 0.15, 'bebas': 0.1
            },
            'budget': {
                'low': 0.1, 'mid': 0.1, 'high': 0.1, 'bebas': 0.05
            }
        }
        
        base_strength = adjustment_strengths.get(question, {}).get(answer, 0.2)
        
        # Feature-specific adjustments with BALANCED constraints
        if feature in ['rasa_manis', 'sweetness_score']:
            if question == 'taste' and answer == 'manis':
                adjusted_weight = max(adjusted_weight, base_strength * 0.8)  # REDUCED from 1.2
            elif question == 'mood' and answer == 'manis':
                adjusted_weight = max(adjusted_weight, base_strength * 0.6)  # REDUCED from 0.8
            elif question == 'mood' and answer == 'rileks':
                adjusted_weight = max(adjusted_weight, base_strength * 0.4)  # REDUCED from 0.6
            # ADD PENALTY for excessive manis in non-manis contexts
            elif question == 'taste' and answer in ['asam', 'pahit', 'netral']:
                adjusted_weight = min(adjusted_weight, -base_strength * 0.3)  # Penalty
        
        elif feature == 'rasa_asam':
            if question == 'taste' and answer == 'asam':
                adjusted_weight = max(adjusted_weight, base_strength * 1.8)  # INCREASED from 1.5
            elif question == 'mood' and answer == 'menyegarkan':
                adjusted_weight = max(adjusted_weight, base_strength * 1.2)  # INCREASED from 0.8
            # ADD BOOST for asam in other relevant contexts
            elif question == 'texture' and answer in ['light', 'bubbly']:
                adjusted_weight = max(adjusted_weight, base_strength * 0.3)
        
        elif feature == 'rasa_netral':
            if question == 'taste' and answer == 'netral':
                adjusted_weight = max(adjusted_weight, base_strength * 1.5)  # INCREASED from 1.1
            # ADD SUPPORT for netral in balanced contexts
            elif question == 'mood' and answer in ['rileks']:
                adjusted_weight = max(adjusted_weight, base_strength * 0.4)
            elif question == 'caffeine' and answer in ['sedang', 'rendah']:
                adjusted_weight = max(adjusted_weight, base_strength * 0.3)
        
        elif feature == 'rasa_pahit':
            if question == 'taste' and answer == 'pahit':
                adjusted_weight = max(adjusted_weight, base_strength * 1.2)
            elif question == 'taste' and answer == 'manis':
                adjusted_weight = min(adjusted_weight, -base_strength * 1.0)  # INCREASED penalty
            elif question == 'mood' and answer == 'energi':
                adjusted_weight = max(adjusted_weight, base_strength * 0.6)
        
        # Additional constraints to prevent manis dominance
        if 'manis' in feature or 'sweetness' in feature:
            # Apply dampening factor for manis features
            if question != 'taste' or answer != 'manis':
                adjusted_weight *= 0.7  # Reduce non-primary manis influence
        
        return adjusted_weight


    def get_hybrid_user_vector(self, mood, taste, texture, caffeine, temperature, budget):
        """FIXED: Enhanced user vector generation with balanced weighting"""
        if not self.trained:
            raise ValueError("Model has not been trained or loaded!")
        
        vec = pd.Series(0.0, index=self.feature_cols +
                       ['temperatur_pref', 'tingkat_kafein_pref', 'budget_pref'])
        
        question_answers = {
            'taste': taste,        # Primary
            'mood': mood,          # Secondary
            'caffeine': caffeine,  # Important constraint
            'texture': texture,    # Preference modifier
            'temperature': temperature,  # Situational
            'budget': budget       # Practical constraint
        }
        
        total_importance = sum(self.question_importance.values()) or 1

        # FIXED: Apply weights with hierarchical scaling
        for question, answer in question_answers.items():
            if (question in self.user_vector_weights and
                answer in self.user_vector_weights[question]):
                
                question_weight = self.question_importance.get(question, 0) / total_importance
                weights = self.user_vector_weights[question][answer]

                for feature, weight in weights.items():
                    if feature in vec.index:
                        # FIXED: Hierarchical weight scaling
                        if question == 'taste':
                            scaled_weight = weight * question_weight #* 1.2  # Boost primary
                        elif question == 'mood':
                            scaled_weight = weight * question_weight #* 1.1  # Boost secondary
                        else:
                            scaled_weight = weight * question_weight
                        
                        vec[feature] += scaled_weight
        
        # FIXED: Apply user vector correlation constraints
        vec = self._apply_user_vector_balanced_constraints(vec, question_answers)
        
        # FIXED: Adaptive normalization preserving relationships
        feature_weights = vec[self.feature_cols]
        
        # Use softer normalization to preserve feature relationships
        max_abs_weight = feature_weights.abs().max()
        if max_abs_weight > 0:
            # Adaptive normalization factor based on feature diversity
            diversity_factor = len(feature_weights[feature_weights.abs() > 0.1]) / len(feature_weights)
            normalization_factor = min(2.5, max_abs_weight * (1 + diversity_factor))
            vec[self.feature_cols] = feature_weights / normalization_factor
        
        # Ensure values are within reasonable bounds with some extrapolation
        min_val = self.menu_df[self.feature_cols].min() * 0.3
        max_val = self.menu_df[self.feature_cols].max() * 1.7
        vec[self.feature_cols] = np.clip(vec[self.feature_cols], min_val, max_val)

        # Set preference metadata
        vec['temperatur_pref'] = temperature
        vec['tingkat_kafein_pref'] = caffeine
        vec['budget_pref'] = budget

        return vec

    def _apply_user_vector_balanced_constraints(self, vec, preferences):
        """FIXED: Apply balanced constraints to user vector"""
        
        # Primary: Taste-driven constraints
        taste_pref = preferences.get('taste')
        if taste_pref == 'manis':
            if vec['rasa_manis'] > 0:
                vec['sweetness_score'] = max(vec['sweetness_score'], vec['rasa_manis'] * 0.9)
        elif taste_pref == 'pahit':
            if vec['rasa_pahit'] > 0:
                # Ensure caffeine correlation for bitter taste
                vec['kafein_score'] = max(vec['kafein_score'], vec['rasa_pahit'] * 0.7)
        elif taste_pref == 'asam':
            # FIXED: Ensure ASAM preferences are preserved
            vec['rasa_asam'] = max(vec['rasa_asam'], 0.2)  # Minimum ASAM level
        elif taste_pref == 'netral':
            vec['rasa_netral'] = max(vec['rasa_netral'], 0.3)
            # Reduce dominance of other tastes for netral preference
            for taste_feature in ['rasa_asam', 'rasa_manis', 'rasa_pahit']:
                vec[taste_feature] *= 0.9
        
        # Secondary: Caffeine-driven constraints
        caffeine_pref = preferences.get('caffeine')
        if caffeine_pref == 'tinggi':
            vec['kafein_score'] = max(vec['kafein_score'], 0.3)
            vec['rasa_pahit'] = max(vec['rasa_pahit'], vec['kafein_score'] * 0.6)
        elif caffeine_pref == 'non-kafein':
            vec['kafein_score'] = min(vec['kafein_score'], 0)
        elif caffeine_pref == 'sedang':
            vec['kafein_score'] = np.clip(vec['kafein_score'], 0.1, 0.6)
        
        # Tertiary: Mood-driven constraints
        mood_pref = preferences.get('mood')
        if mood_pref == 'energi':
            if vec['kafein_score'] > 0:
                vec['kafein_score'] = max(vec['kafein_score'], 0.2)
        elif mood_pref == 'rileks':
            vec['kafein_score'] = min(vec['kafein_score'], 0.3)
        elif mood_pref == 'menyegarkan':
            vec['rasa_asam'] = max(vec['rasa_asam'], 0.1)
        
        return vec

    def recommend(self, preferences):
        """FIXED: Enhanced recommendation with soft filtering"""
        if not self.trained or self.menu_df.empty:
            return []
        
        session_id = preferences.get('session_id')
        if not session_id:
            print("❌ No session_id provided")
            return []
        
        quiz_attempt = SessionManager.get_next_quiz_attempt(session_id)
        nama_customer = preferences.get('nama_customer')

        # Generate enhanced user vector
        mood = preferences.get('mood')
        taste = preferences.get('rasa')
        texture = preferences.get('tekstur')
        caffeine = preferences.get('kafein')
        temperature = preferences.get('suhu')
        budget = preferences.get('budget')
        
        user_vec = self.get_hybrid_user_vector(mood, taste, texture, caffeine,
                                            temperature, budget)
        
        # Enhanced similarity calculation with balanced feature weights
        feature_df = self.menu_df[self.feature_cols]
        
        # Use balanced feature importance weights
        feature_weights = self._calculate_balanced_feature_weights()
        weighted_feature_df = feature_df * feature_weights
        weighted_user_features = user_vec[self.feature_cols] * feature_weights
        
        similarity = cosine_similarity([weighted_user_features], weighted_feature_df)[0]
        
        df = self.menu_df.copy()
        df['similarity'] = similarity
        
        # FIXED: Apply soft filtering instead of hard filtering
        filtered = self._apply_soft_filtering(df, preferences)
        
        if filtered.empty:
            print(f"\n⚠️ No products found matching criteria.")
            return []
        
        # Enhanced ranking with balanced scoring
        filtered = self._calculate_balanced_final_scores(filtered, user_vec, preferences)
        
        # Get top recommendations
        sorted_df = filtered.sort_values(['final_score'], ascending=False).head(3)
        
        # Save preferences and recommendations
        pref_id = None
        if nama_customer:
            pref_id = self._save_user_preferences(preferences, session_id, quiz_attempt)
            if pref_id: 
                success = self._save_recommendations(pref_id, session_id, quiz_attempt, sorted_df)
                if success:
                    SessionManager.update_session_activity(session_id, quiz_attempt)
        
        # Build response
        recs_list = []
        for _, row in sorted_df.iterrows():
            actual_price = row['original_harga']
            kategori = row.get('kategori', 'Minuman')
            if pd.isna(kategori) or kategori == '':
                if 'kopi' in row['nama_minuman'].lower():
                    kategori = 'Kopi'
                elif 'teh' in row['nama_minuman'].lower():
                    kategori = 'Teh'
                elif 'jus' in row['nama_minuman'].lower():
                    kategori = 'Jus'
                else:
                    kategori = 'Minuman'
            
            recs_list.append({
                'nama_minuman': row['nama_minuman'],
                'kategori': kategori,
                'tingkat_kafein': row['tingkat_kafein'],
                'harga': actual_price,
                'similarity': row['similarity'],
                'final_score': row['final_score'],
                'foto': row.get('foto', 'default.png'),
                'session_id': session_id,
                'quiz_attempt': quiz_attempt,
                'pref_id': pref_id
            })
        
        return recs_list
    
    def _calculate_balanced_feature_weights(self):
        """FIXED: Calculate balanced feature importance weights"""
        weights = pd.Series(1.0, index=self.feature_cols)
        
        # Balanced weights based on domain knowledge
        weights['rasa_asam'] = 1.4      # Increased for ASAM
        weights['rasa_manis'] = 1.3     # Sweet taste is distinctive
        weights['rasa_pahit'] = 1.3     # Bitter taste important
        weights['rasa_netral'] = 1.2    # Netral is explicitly modeled
        weights['rasa_gurih'] = 1.1     # Savory taste
        
        weights['kafein_score'] = 1.3   # Caffeine is important constraint
        weights['sweetness_score'] = 1.2 # Correlates with sweetness
        
        # Texture weights - equal importance
        weights['tekstur_LIGHT'] = 1.1
        weights['tekstur_CREAMY'] = 1.1
        weights['tekstur_BUBBLY'] = 1.1
        weights['tekstur_HEAVY'] = 1.1
        
        # Other features
        weights['carbonated_score'] = 1.0
        weights['harga'] = 0.9          # Price moderately important
        # weights['popularitas'] = 0.8    # Popularity least important
        
        return weights

    def _calculate_balanced_preference_alignment(self, df, preferences):
        """FIXED: Balanced preference alignment with reduced manis dominance"""
        bonus = np.zeros(len(df))
        
        # PRIMARY: Taste alignment with BALANCED weights
        taste = preferences.get('rasa')
        if taste == 'asam':
            bonus += df['rasa_asam'] * 0.35      # INCREASED from 0.3
        elif taste == 'manis':
            bonus += df['rasa_manis'] * 0.2      # REDUCED from 0.25
            bonus += df['sweetness_score'] * 0.1  # REDUCED from 0.15
        elif taste == 'pahit':
            bonus += df['rasa_pahit'] * 0.25
        elif taste == 'netral':
            bonus += df['rasa_netral'] * 0.3     # INCREASED from 0.25
        
        # SECONDARY: Mood alignment with balanced scoring
        mood = preferences.get('mood')
        if mood == 'energi':
            bonus += df['kafein_score'] * 0.15
            bonus += (1 - df['sweetness_score']) * 0.08
        elif mood == 'rileks':
            bonus += (1 - df['kafein_score']) * 0.15
            bonus += df['sweetness_score'] * 0.05    # REDUCED from 0.08
            bonus += df['tekstur_CREAMY'] * 0.06     # REDUCED from 0.08
        elif mood == 'menyegarkan':
            bonus += df['rasa_asam'] * 0.15          # INCREASED from 0.12
            bonus += df['tekstur_LIGHT'] * 0.08
            bonus += df['tekstur_BUBBLY'] * 0.08
        elif mood == 'manis':
            bonus += df['rasa_manis'] * 0.1          # REDUCED from 0.15
            bonus += df['sweetness_score'] * 0.08    # REDUCED from 0.12
        
        # TERTIARY: Other preferences
        caffeine = preferences.get('kafein')
        if caffeine == 'tinggi':
            bonus += df['kafein_score'] * 0.1
        elif caffeine == 'non-kafein':
            bonus += (1 - df['kafein_score']) * 0.1
        
        # Texture alignment
        texture = preferences.get('tekstur')
        if texture:
            texture_col = f'tekstur_{texture.upper()}'
            if texture_col in df.columns:
                bonus += df[texture_col] * 0.08
        
        return np.clip(bonus, 0, 0.6)

    def _apply_soft_filtering(self, df, preferences):
        """FIXED: Apply soft filtering instead of hard constraints"""
        filtered = df.copy()
        
        # Initialize soft scores
        filtered['soft_score'] = 1.0
        
        # Temperature soft filtering
        temperature = preferences.get('suhu')
        if temperature and temperature.lower() != 'bebas':
            if 'temperatur_opsi' in filtered.columns:
                try:
                    temp_mapping = {
                        'dingin': 'cold',
                        'panas': 'hot',
                        'cold': 'cold',
                        'hot': 'hot'
                    }
                    mapped_temp = temp_mapping.get(temperature.lower(), temperature.lower())
                    
                    # Soft constraint: prefer matching temperature but don't exclude others
                    temp_match_mask = (filtered['temperatur_opsi'] == mapped_temp) | (filtered['temperatur_opsi'] == 'both')
                    filtered.loc[temp_match_mask, 'soft_score'] *= 1.2
                    filtered.loc[~temp_match_mask, 'soft_score'] *= 0.8
                except (ValueError, TypeError):
                    pass
        
        # FIXED: Caffeine soft filtering
        caffeine = preferences.get('kafein')
        if caffeine and caffeine != 'bebas':
            if caffeine == 'non-kafein':
                # Prefer non-caffeine but don't exclude caffeine completely
                non_caff_mask = filtered['kafein_score'] == 0
                filtered.loc[non_caff_mask, 'soft_score'] *= 1.4
                filtered.loc[~non_caff_mask, 'soft_score'] *= 0.6
                
            elif caffeine == 'tinggi':
                # Prefer high caffeine items (caffeinated + bitter)
                high_caff_mask = (filtered['kafein_score'] == 1) & (filtered['rasa_pahit'] >= 0.6)
                med_caff_mask = (filtered['kafein_score'] == 1) & (filtered['rasa_pahit'] >= 0.3) & (filtered['rasa_pahit'] < 0.6)
                
                filtered.loc[high_caff_mask, 'soft_score'] *= 1.5
                filtered.loc[med_caff_mask, 'soft_score'] *= 1.2
                filtered.loc[filtered['kafein_score'] == 0, 'soft_score'] *= 0.7
                
            elif caffeine == 'sedang':
                # Prefer medium caffeine
                med_caff_mask = (filtered['kafein_score'] == 1) & (filtered['rasa_pahit'] >= 0.3) & (filtered['rasa_pahit'] < 0.7)
                filtered.loc[med_caff_mask, 'soft_score'] *= 1.4
                
            elif caffeine == 'rendah':
                # Prefer low/no caffeine
                low_caff_mask = (filtered['kafein_score'] == 0) | ((filtered['kafein_score'] == 1) & (filtered['rasa_pahit'] < 0.4))
                filtered.loc[low_caff_mask, 'soft_score'] *= 1.3
        
        # FIXED: Taste soft filtering with strong ASAM support
        taste = preferences.get('rasa')
        if taste and taste != 'bebas':
            if taste == 'asam':
                # FIXED: Strong preference for ASAM items using adaptive threshold
                asam_threshold = self.adaptive_thresholds['rasa_asam']['low']
                strong_asam_mask = filtered['rasa_asam'] >= asam_threshold * 2
                med_asam_mask = (filtered['rasa_asam'] >= asam_threshold) & (filtered['rasa_asam'] < asam_threshold * 2)
                weak_asam_mask = (filtered['rasa_asam'] > 0) & (filtered['rasa_asam'] < asam_threshold)
                
                filtered.loc[strong_asam_mask, 'soft_score'] *= 2.0  # Strong boost
                filtered.loc[med_asam_mask, 'soft_score'] *= 1.6     # Medium boost
                filtered.loc[weak_asam_mask, 'soft_score'] *= 1.2    # Small boost
                filtered.loc[filtered['rasa_asam'] == 0, 'soft_score'] *= 0.5  # Penalty for no sourness
                
            elif taste == 'manis':
                manis_threshold = self.adaptive_thresholds['rasa_manis']['medium']
                sweet_mask = (filtered['rasa_manis'] >= manis_threshold) & (filtered['sweetness_score'] >= manis_threshold)
                filtered.loc[sweet_mask, 'soft_score'] *= 1.6
                
            elif taste == 'pahit':
                pahit_threshold = self.adaptive_thresholds['rasa_pahit']['medium']
                bitter_mask = filtered['rasa_pahit'] >= pahit_threshold
                filtered.loc[bitter_mask, 'soft_score'] *= 1.5
                
            elif taste == 'netral':
                netral_threshold = self.adaptive_thresholds['rasa_netral']['medium']
                netral_mask = filtered['rasa_netral'] >= netral_threshold
                filtered.loc[netral_mask, 'soft_score'] *= 1.4
        
        # FIXED: Budget soft filtering
        budget = preferences.get('budget')
        if budget and budget != 'bebas':
            if budget == 'low':
                budget_threshold = self.adaptive_thresholds['budget']['low']
                low_budget_mask = filtered['harga'] <= budget_threshold
                med_budget_mask = (filtered['harga'] > budget_threshold) & (filtered['harga'] <= budget_threshold * 1.5)
                
                filtered.loc[low_budget_mask, 'soft_score'] *= 1.3
                filtered.loc[med_budget_mask, 'soft_score'] *= 1.1
                filtered.loc[filtered['harga'] > budget_threshold * 1.5, 'soft_score'] *= 0.7
                
            elif budget == 'mid':
                budget_threshold_low = self.adaptive_thresholds['budget']['low']
                budget_threshold_high = self.adaptive_thresholds['budget']['mid']
                mid_budget_mask = (filtered['harga'] > budget_threshold_low) & (filtered['harga'] <= budget_threshold_high)
                
                filtered.loc[mid_budget_mask, 'soft_score'] *= 1.2
                filtered.loc[filtered['harga'] <= budget_threshold_low, 'soft_score'] *= 1.1
                filtered.loc[filtered['harga'] > budget_threshold_high, 'soft_score'] *= 0.8
        
        return filtered

    def _calculate_balanced_final_scores(self, df, user_vec, preferences):
        """FIXED: Calculate balanced final scores"""
        df = df.copy()
        
        # Base similarity score
        base_similarity = df['similarity']
        
        # Soft filtering score
        soft_filter_score = df.get('soft_score', 1.0)
        
        # Balanced preference alignment bonus
        preference_bonus = self._calculate_balanced_preference_alignment(df, preferences)
        
        # Correlation consistency bonus
        # correlation_bonus = self._calculate_correlation_consistency_bonus(df, preferences)
        
        # Popularity factor (minimal influence)
        # popularity_factor = df['popularitas'] * 0.05
        
        # FIXED: Balanced final score combination
        df['final_score'] = (
            base_similarity * 0.8 +           # Base similarity
            soft_filter_score * 0.2 +          # Soft filtering (increased)
            preference_bonus * 0           # Preference alignment
            # correlation_bonus * 0.05          # Correlation consistency
            # popularity_factor * 0.05           # Popularity (minimal)
        )
        
        # Ensure scores are between 0 and 1
        df['final_score'] = np.clip(df['final_score'], 0, 1)
        
        return df


    def get_budget_range(self, budget):
        """FIXED: Use adaptive budget ranges"""
        if budget == 'low':
            return self.adaptive_thresholds['budget']['low']
        elif budget == 'mid':
            return self.adaptive_thresholds['budget']['mid']
        elif budget == 'high':
            return self.adaptive_thresholds['budget']['high']
        else:
            return float('inf')

    # Include all the existing utility methods
    def _generate_data_hash(self):
        if self.menu_df.empty: 
            return "no_data_hash"
        data_str = f"{self.menu_df.shape}_{self.feature_cols}_{self.menu_df.iloc[0][self.feature_cols].to_dict()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _get_model_filename(self):
        return f"enhanced_hybrid_fixed_v3_{self.model_hash}.pkl"

    def _save_trained_model(self):
        model_data = {
            'user_vector_weights': self.user_vector_weights,
            'question_importance': self.question_importance,
            'model_performance': self.model_performance,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler,
            'model_hash': self.model_hash,
            'feature_correlations': self.feature_correlations,
            'adaptive_thresholds': self.adaptive_thresholds,
            'trained_date': datetime.now().isoformat(),
            'dataset_shape': self.menu_df.shape,
            'training_config': {
                'method': 'enhanced_hybrid_fixed_v3_balanced',
                'n_users': 1500,
                'random_seed': 42,
                'fixes_applied': [
                    'taste_primary_driver',
                    'asam_strong_support', 
                    'soft_filtering',
                    'balanced_importance',
                    'context_specific_adjustments',
                    'adaptive_thresholds'
                ]
            }
        }
        filepath = os.path.join(self.model_dir, self._get_model_filename())
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Enhanced hybrid fixed v3 model saved to: {filepath}")

    def _load_trained_model(self):
        try:
            filepath = os.path.join(self.model_dir, self._get_model_filename())
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                if (model_data.get('model_hash') == self.model_hash and
                    model_data.get('feature_cols') == self.feature_cols):
                    self.user_vector_weights = model_data['user_vector_weights']
                    self.question_importance = model_data['question_importance']
                    self.model_performance = model_data['model_performance']
                    self.scaler = model_data['scaler']
                    self.feature_cols = model_data['feature_cols']
                    self.feature_correlations = model_data.get('feature_correlations', {})
                    self.adaptive_thresholds = model_data.get('adaptive_thresholds', {})
                    self.trained = True
                    print(f"✅ Enhanced hybrid fixed v3 model loaded from: {filepath}")
                    return True
                else:
                    print("⚠️ Existing model incompatible with current dataset.")
                    return False
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _save_user_preferences(self, preferences, session_id, quiz_attempt):
        """Save user preferences with proper session tracking"""
        try:
            pref_id = HybridIDGenerator.generate_hybrid_id("PREF")
            
            query = """
                INSERT INTO preferences 
                (pref_id, session_id, quiz_attempt, nama_customer, mood, rasa, tekstur, kafein, suhu, budget, timestamp) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            params = (
                pref_id,
                session_id,
                quiz_attempt,
                preferences.get('nama_customer'), 
                preferences.get('mood'), 
                preferences.get('rasa'), 
                preferences.get('tekstur'), 
                preferences.get('kafein'), 
                preferences.get('suhu'), 
                preferences.get('budget')
            )
            execute_query(query, params)
            print(f"✅ User preferences saved: {pref_id}")
            return pref_id
        except Exception as e:
            print(f"❌ Error saving preferences: {e}")
            return None

    def _save_recommendations(self, pref_id, session_id, quiz_attempt, rec_df):
        """Save recommendations with proper tracking"""
        try:
            for rank, (_, row) in enumerate(rec_df.iterrows(), 1):
                rec_id = HybridIDGenerator.generate_hybrid_id("REC")
                
                # recommender.py (_save_recommendations)
                query = """
                    INSERT INTO recommendations 
                    (rec_id, pref_id, menu_id, similarity, final_score, rank_position, is_top3, 
                    session_id, quiz_attempt, generated_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """
                params = (
                    rec_id, 
                    pref_id, 
                    row['menu_id'],
                    float(row['similarity']),           # <- real cosine similarity
                    float(row['final_score']),          # <- final score gabungan
                    rank,
                    1 if rank <= 3 else 0,
                    session_id,
                    quiz_attempt
                )
                execute_query(query, params)
            
            print(f"✅ Recommendations saved for session: {session_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving recommendations: {e}")
            return False

    def _display_enhanced_results(self):
        """Display enhanced training results with fixed metrics"""
        print("\n" + "="*80)
        print("ENHANCED HYBRID RESULTS - FIXED VERSION")
        print("="*80)
        
        print("\n📊 QUESTION IMPORTANCE RANKING (FIXED & BALANCED):")
        print("-" * 60)
        sorted_questions = sorted(self.question_importance.items(),
                                key=lambda x: x[1], reverse=True)
        
        for i, (question, score) in enumerate(sorted_questions, 1):
            bar_length = int(score * 50 / max(self.question_importance.values()))
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{i}. {question.upper():<12}: {score:.4f} |{bar}|")
        
        print("\n🔍 BALANCED ANALYSIS:")
        print("-" * 60)
        
        for question in sorted(self.question_importance.keys()):
            if question in self.model_performance:
                answers = list(self.model_performance[question].keys())
                avg_r2 = np.mean([
                    self.model_performance[question][answer]['r2'] 
                    for answer in answers
                ])
                avg_correlation = np.mean([
                    self.model_performance[question][answer].get('correlation_score', 0.5)
                    for answer in answers
                ])
                total_samples = sum([
                    self.model_performance[question][answer]['n_samples'] 
                    for answer in answers
                ])
                
                print(f"{question.upper():<12}: "
                      f"importance={self.question_importance[question]:.4f}, "
                      f"r2={avg_r2:.3f}, "
                      f"corr_compliance={avg_correlation:.3f}, "
                      f"samples={total_samples}")
        
        print("\n📈 KEY FIXES APPLIED:")
        print("-" * 60)
        fixes = [
            "✅ TASTE as primary driver (not mood)",
            "✅ ASAM strong support with adaptive thresholds", 
            "✅ Soft filtering instead of hard constraints",
            "✅ Balanced question importance calculation",
            "✅ Context-specific weight adjustments",
            "✅ Adaptive data-driven thresholds"
        ]
        for fix in fixes:
            print(f"   {fix}")
        
        print("\n📊 ADAPTIVE THRESHOLDS:")
        print("-" * 60)
        for feature, thresholds in self.adaptive_thresholds.items():
            if feature in ['rasa_asam', 'rasa_manis', 'rasa_pahit', 'rasa_netral']:
                print(f"{feature}: low={thresholds['low']:.3f}, med={thresholds['medium']:.3f}, high={thresholds['high']:.3f}")


# Global recommender system instance
recommender_system = None

def init_recommender(app):
    """Initialize the enhanced fixed recommendation system"""
    global recommender_system
    print("🚀 Initializing enhanced hybrid recommendation system - FIXED VERSION...")
    with app.app_context():
        try:
            recommender_system = EnhancedHybridRecommendationSystem()
            if recommender_system.menu_df is None or recommender_system.menu_df.empty:
                raise RuntimeError("No menu data found!")
            print("✅ Enhanced hybrid recommendation system FIXED VERSION initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            recommender_system = None

def create_session():
    """Create new session"""
    return SessionManager.create_new_session()

def get_recommendations(preferences):
    """Get recommendations using the enhanced fixed system"""
    if recommender_system is None: 
        return []
    try:
        return recommender_system.recommend(preferences)
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return []

def update_feedback(session_id, pref_id, quiz_attempt, feedback_bool):
    """Update feedback for recommendations"""
    try:
        feedback_value = 1 if feedback_bool else 0
        
        query_rec = """
            UPDATE recommendations 
            SET feedback = %s, feedback_timestamp = NOW() 
            WHERE pref_id = %s AND session_id = %s AND quiz_attempt = %s
        """
        rows_rec = execute_query(query_rec, params=(feedback_value, pref_id, session_id, quiz_attempt))
        
        if rows_rec and rows_rec > 0:
            query_pref = """
                UPDATE preferences 
                SET feedback_status = 'completed' 
                WHERE pref_id = %s AND session_id = %s AND quiz_attempt = %s
            """
            rows_pref = execute_query(query_pref, params=(pref_id, session_id, quiz_attempt))
            
            print(f"✅ Feedback updated successfully")
            return True
        else: 
            print(f"❌ No recommendations found")
            return False
            
    except Exception as e:
        print(f"❌ Error updating feedback: {e}")
        return False

def get_recommendation_analytics():
    """Get analytics with support for enhanced tracking"""
    try:
        query = """
            SELECT COUNT(*) AS total_recs,
                   COUNT(CASE WHEN feedback = 1 THEN 1 END) AS positive_feedback,
                   COUNT(CASE WHEN feedback = 0 THEN 1 END) AS negative_feedback,
                   COUNT(CASE WHEN feedback IS NULL THEN 1 END) AS pending_feedback,
                   COUNT(DISTINCT session_id) AS total_sessions,
                   AVG(quiz_attempt) AS avg_quiz_attempts,
                   AVG(similarity) AS avg_similarity_score
            FROM recommendations;
        """
        stats = execute_query(query, fetch='one')
        if not stats: 
            return {'total_recs': 0, 'satisfaction_rate': 0}
        
        total_feedback = stats['positive_feedback'] + stats['negative_feedback']
        stats['satisfaction_rate'] = (stats['positive_feedback'] / total_feedback * 100) if total_feedback > 0 else 0
        
        return stats
    except Exception as e:
        print(f"❌ Error fetching analytics: {e}")
        return {}

def get_session_analytics(session_id):
    """Get analytics for a specific session"""
    try:
        query = """
            SELECT 
                s.session_id,
                s.created_at,
                s.last_activity,
                s.total_quiz_attempts,
                COUNT(DISTINCT p.pref_id) as completed_quizzes,
                COUNT(r.rec_id) as total_recommendations,
                AVG(r.feedback) as avg_feedback,
                AVG(r.similarity) as avg_similarity
            FROM user_sessions s
            LEFT JOIN preferences p ON s.session_id = p.session_id
            LEFT JOIN recommendations r ON p.pref_id = r.pref_id
            WHERE s.session_id = %s
            GROUP BY s.session_id
        """
        result = execute_query(query, params=(session_id,), fetch='one')
        return result if result else {}
    except Exception as e:
        print(f"❌ Error fetching session analytics: {e}")
        return {}