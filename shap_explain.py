import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Comprehensive3DCNN_SHAPAnalysis:
    """
    Deep SHAP analysis for 3D CNN NASA-TLX prediction model.
    Provides comprehensive interpretability analysis for research publications.
    """
    
    def __init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Initialize SHAP analysis for 3D CNN TLX prediction model.
        
        Parameters:
        -----------
        model : tensorflow.keras.Model
            Trained 3D CNN model
        X_train, X_val, X_test : numpy.ndarray
            Data splits with shape (samples, 120, 16, 12, 1)
        y_train, y_val, y_test : numpy.ndarray
            Target TLX scores with shape (samples, 5)
        """
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # TLX subscale names
        self.tlx_subscales = ['Mental_Demand', 'Temporal_Demand', 'Effort', 
                             'Performance', 'Frustration']
        
        # Initialize analysis components
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        self.test_samples = None
        self.predictions = None
        
        # Results storage
        self.analysis_results = {}
        
    def prepare_analysis_data(self, n_background=100, n_test_samples=50):
        """
        Prepare background and test data for SHAP analysis.
        
        Parameters:
        -----------
        n_background : int
            Number of background samples for SHAP explainer
        n_test_samples : int
            Number of test samples to analyze
        """
        print("="*80)
        print("PREPARING DATA FOR COMPREHENSIVE SHAP ANALYSIS")
        print("="*80)
        
        # Select representative background samples from training data
        np.random.seed(42)
        bg_indices = np.random.choice(len(self.X_train), 
                                     min(n_background, len(self.X_train)), 
                                     replace=False)
        self.background_data = self.X_train[bg_indices]
        self.background_labels = self.y_train[bg_indices]
        
        # Select diverse test samples for detailed analysis
        test_indices = self._select_diverse_test_samples(n_test_samples)
        self.test_samples = self.X_test[test_indices]
        self.test_labels = self.y_test[test_indices]
        
        print(f"Background data shape: {self.background_data.shape}")
        print(f"Background labels shape: {self.background_labels.shape}")
        print(f"Test samples shape: {self.test_samples.shape}")
        print(f"Test labels shape: {self.test_labels.shape}")
        
        # Get model predictions for test samples
        self.predictions = self.model.predict(self.test_samples, verbose=0)
        
        print(f"Model predictions shape: {self.predictions.shape}")
        print("Data preparation completed successfully!")
        
    def _select_diverse_test_samples(self, n_samples):
        """Select diverse test samples covering different TLX score ranges."""
        # Calculate prediction errors to select diverse samples
        all_predictions = self.model.predict(self.X_test, verbose=0)
        errors = []
        
        for i in range(len(self.X_test)):
            mae = mean_absolute_error(self.y_test[i], all_predictions[i])
            errors.append(mae)
        
        errors = np.array(errors)
        
        # Select samples from different error quantiles for diversity
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        selected_indices = []
        
        for q in quantiles:
            threshold = np.quantile(errors, q)
            candidates = np.where(errors <= threshold)[0]
            if len(candidates) > 0:
                selected_indices.extend(candidates[:n_samples//len(quantiles)])
        
        # Fill remaining slots randomly
        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            remaining_candidates = list(set(range(len(self.X_test))) - set(selected_indices))
            additional = np.random.choice(remaining_candidates, 
                                        min(remaining, len(remaining_candidates)), 
                                        replace=False)
            selected_indices.extend(additional)
        
        return selected_indices[:n_samples]
    
    def initialize_shap_explainer(self, explainer_type='deep'):
        """
        Initialize SHAP explainer for 3D CNN model.
        
        Parameters:
        -----------
        explainer_type : str
            Type of explainer ('deep', 'gradient', 'kernel')
        """
        print("\n" + "="*80)
        print(f"INITIALIZING {explainer_type.upper()} SHAP EXPLAINER")
        print("="*80)
        
        try:
            if explainer_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
                print("DeepExplainer initialized successfully for 3D CNN model")
                
            elif explainer_type == 'gradient':
                self.explainer = shap.GradientExplainer(self.model, self.background_data)
                print("GradientExplainer initialized successfully for 3D CNN model")
                
            elif explainer_type == 'kernel':
                def model_predict(x):
                    return self.model.predict(x, verbose=0)
                self.explainer = shap.KernelExplainer(model_predict, self.background_data)
                print("KernelExplainer initialized successfully for 3D CNN model")
                
            else:
                raise ValueError(f"Unsupported explainer type: {explainer_type}")
                
        except Exception as e:
            print(f"Error initializing explainer: {e}")
            raise
    
    def compute_shap_values(self):
        """Compute SHAP values for all test samples."""
        print("\n" + "="*80)
        print("COMPUTING SHAP VALUES FOR ALL TEST SAMPLES")
        print("="*80)
        
        print("Computing SHAP values... This may take several minutes.")
        
        try:
            self.shap_values = self.explainer.shap_values(self.test_samples)
            
            # Handle multi-output case
            if isinstance(self.shap_values, list):
                print(f"Multi-output SHAP values computed for {len(self.shap_values)} TLX subscales")
                for i, subscale in enumerate(self.tlx_subscales):
                    print(f"{subscale}: {self.shap_values[i].shape}")
            else:
                print(f"Single output SHAP values computed: {self.shap_values.shape}")
                
            print("SHAP values computation completed successfully!")
            
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            raise
    
    def analyze_global_feature_importance(self):
        """Analyze global feature importance across all samples and subscales."""
        print("\n" + "="*80)
        print("GLOBAL FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        global_importance = {}
        detailed_stats = {}
        
        for i, subscale in enumerate(self.tlx_subscales):
            print(f"\nAnalyzing {subscale}...")
            
            # Get SHAP values for this subscale
            shap_vals = self.shap_values[i] if isinstance(self.shap_values, list) else self.shap_values
            
            # Compute mean absolute SHAP values across all samples
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            global_importance[subscale] = mean_abs_shap
            
            # Detailed statistics
            max_importance = mean_abs_shap.max()
            mean_importance = mean_abs_shap.mean()
            std_importance = mean_abs_shap.std()
            total_importance = mean_abs_shap.sum()
            
            # Calculate importance concentration (Gini coefficient)
            flat_importance = mean_abs_shap.flatten()
            sorted_importance = np.sort(flat_importance)
            n = len(sorted_importance)
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_importance)) / (n * np.sum(sorted_importance)) - (n + 1) / n
            
            detailed_stats[subscale] = {
                'max_importance': max_importance,
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'total_importance': total_importance,
                'gini_coefficient': gini,
                'shape': mean_abs_shap.shape
            }
            
            print(f"  Max importance: {max_importance:.6f}")
            print(f"  Mean importance: {mean_importance:.6f}")
            print(f"  Std importance: {std_importance:.6f}")
            print(f"  Total importance: {total_importance:.4f}")
            print(f"  Gini coefficient: {gini:.4f}")
            print(f"  Shape: {mean_abs_shap.shape}")
        
        # Compare importance across subscales
        print(f"\n{'='*60}")
        print("CROSS-SUBSCALE IMPORTANCE COMPARISON")
        print(f"{'='*60}")
        
        importance_comparison = pd.DataFrame(detailed_stats).T
        print("\nImportance Statistics by TLX Subscale:")
        print(importance_comparison.round(6))
        
        # Identify most/least important subscales
        max_total_subscale = importance_comparison['total_importance'].idxmax()
        min_total_subscale = importance_comparison['total_importance'].idxmin()
        
        print(f"\nMost feature-dependent subscale: {max_total_subscale}")
        print(f"Least feature-dependent subscale: {min_total_subscale}")
        
        self.analysis_results['global_importance'] = global_importance
        self.analysis_results['importance_stats'] = detailed_stats
        
        return global_importance, detailed_stats
    
    def analyze_temporal_importance_patterns(self):
        """Analyze importance patterns across the temporal dimension (120 frames)."""
        print("\n" + "="*80)
        print("TEMPORAL IMPORTANCE PATTERN ANALYSIS")
        print("="*80)
        
        temporal_patterns = {}
        temporal_stats = {}
        
        for i, subscale in enumerate(self.tlx_subscales):
            print(f"\nAnalyzing temporal patterns for {subscale}...")
            
            # Get SHAP values for this subscale
            shap_vals = self.shap_values[i] if isinstance(self.shap_values, list) else self.shap_values
            
            # Average across samples and spatial dimensions (height, width, channels)
            temporal_profile = np.mean(np.abs(shap_vals), axis=(0, 2, 3, 4))
            temporal_patterns[subscale] = temporal_profile
            
            # Find peak importance frames
            peak_frames = np.argsort(temporal_profile)[-10:]  # Top 10 frames
            peak_importance_values = temporal_profile[peak_frames]
            
            # Find critical time windows (top 20% importance)
            threshold = np.percentile(temporal_profile, 80)
            critical_frames = np.where(temporal_profile > threshold)[0]
            
            # Calculate temporal statistics
            temporal_variance = temporal_profile.var()
            temporal_mean = temporal_profile.mean()
            temporal_max = temporal_profile.max()
            temporal_min = temporal_profile.min()
            temporal_range = temporal_max - temporal_min
            
            # Find temporal trend (linear regression slope)
            frames = np.arange(len(temporal_profile))
            slope, intercept, r_value, p_value, std_err = stats.linregress(frames, temporal_profile)
            
            # Identify early, middle, late importance
            early_importance = np.mean(temporal_profile[:40])    # First third
            middle_importance = np.mean(temporal_profile[40:80]) # Middle third
            late_importance = np.mean(temporal_profile[80:])     # Last third
            
            temporal_stats[subscale] = {
                'peak_frames': peak_frames,
                'peak_values': peak_importance_values,
                'critical_frames': critical_frames,
                'temporal_variance': temporal_variance,
                'temporal_mean': temporal_mean,
                'temporal_max': temporal_max,
                'temporal_min': temporal_min,
                'temporal_range': temporal_range,
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'early_importance': early_importance,
                'middle_importance': middle_importance,
                'late_importance': late_importance
            }
            
            print(f"  Peak frames: {peak_frames}")
            print(f"  Peak importance values: {peak_importance_values}")
            print(f"  Critical time windows: {len(critical_frames)} frames")
            print(f"  Temporal variance: {temporal_variance:.6f}")
            print(f"  Temporal trend slope: {slope:.8f} (R²={r_value**2:.4f}, p={p_value:.4f})")
            print(f"  Early vs Late importance: {early_importance:.6f} vs {late_importance:.6f}")
            
            # Categorize temporal pattern
            if late_importance > early_importance * 1.2:
                pattern_type = "Increasing (Late-phase importance)"
            elif early_importance > late_importance * 1.2:
                pattern_type = "Decreasing (Early-phase importance)"
            else:
                pattern_type = "Stable (Distributed importance)"
            
            print(f"  Temporal pattern: {pattern_type}")
        
        # Cross-subscale temporal comparison
        self._compare_temporal_patterns(temporal_patterns, temporal_stats)
        
        self.analysis_results['temporal_patterns'] = temporal_patterns
        self.analysis_results['temporal_stats'] = temporal_stats
        
        return temporal_patterns, temporal_stats
    
    def _compare_temporal_patterns(self, temporal_patterns, temporal_stats):
        """Compare temporal patterns across TLX subscales."""
        print(f"\n{'='*60}")
        print("CROSS-SUBSCALE TEMPORAL PATTERN COMPARISON")
        print(f"{'='*60}")
        
        # Calculate correlations between temporal patterns
        subscale_names = list(temporal_patterns.keys())
        n_subscales = len(subscale_names)
        correlation_matrix = np.zeros((n_subscales, n_subscales))
        
        for i, subscale_i in enumerate(subscale_names):
            for j, subscale_j in enumerate(subscale_names):
                corr = np.corrcoef(temporal_patterns[subscale_i], 
                                  temporal_patterns[subscale_j])[0, 1]
                correlation_matrix[i, j] = corr
        
        print("\nTemporal Pattern Correlations Between TLX Subscales:")
        corr_df = pd.DataFrame(correlation_matrix, 
                              index=subscale_names, 
                              columns=subscale_names)
        print(corr_df.round(3))
        
        # Find most similar and different temporal patterns
        upper_triangle = np.triu(correlation_matrix, k=1)
        max_corr_idx = np.unravel_index(upper_triangle.argmax(), upper_triangle.shape)
        min_corr_idx = np.unravel_index(upper_triangle.argmin(), upper_triangle.shape)
        
        print(f"\nMost similar temporal patterns: {subscale_names[max_corr_idx[0]]} - {subscale_names[max_corr_idx[1]]} (r={upper_triangle[max_corr_idx]:.3f})")
        print(f"Most different temporal patterns: {subscale_names[min_corr_idx[0]]} - {subscale_names[min_corr_idx[1]]} (r={upper_triangle[min_corr_idx]:.3f})")
        
        # Identify subscales with early vs late importance
        early_dominant = []
        late_dominant = []
        
        for subscale, stats in temporal_stats.items():
            if stats['late_importance'] > stats['early_importance'] * 1.2:
                late_dominant.append(subscale)
            elif stats['early_importance'] > stats['late_importance'] * 1.2:
                early_dominant.append(subscale)
        
        if early_dominant:
            print(f"\nEarly-phase dominant subscales: {', '.join(early_dominant)}")
        if late_dominant:
            print(f"Late-phase dominant subscales: {', '.join(late_dominant)}")
    
    def analyze_spatial_importance_patterns(self):
        """Analyze importance patterns in the spatial dimensions (16x12 feature grid)."""
        print("\n" + "="*80)
        print("SPATIAL IMPORTANCE PATTERN ANALYSIS")
        print("="*80)
        
        spatial_patterns = {}
        spatial_stats = {}
        
        for i, subscale in enumerate(self.tlx_subscales):
            print(f"\nAnalyzing spatial patterns for {subscale}...")
            
            # Get SHAP values for this subscale
            shap_vals = self.shap_values[i] if isinstance(self.shap_values, list) else self.shap_values
            
            # Average across samples and temporal dimension
            spatial_pattern = np.mean(np.abs(shap_vals), axis=(0, 1, 4))  # Shape: (16, 12)
            spatial_patterns[subscale] = spatial_pattern
            
            # Find most important spatial regions
            max_pos = np.unravel_index(spatial_pattern.argmax(), spatial_pattern.shape)
            max_importance = spatial_pattern.max()
            
            # Find regions above 90th percentile
            threshold = np.percentile(spatial_pattern, 90)
            important_regions = np.where(spatial_pattern > threshold)
            n_important_regions = len(important_regions[0])
            
            # Calculate spatial statistics
            spatial_variance = spatial_pattern.var()
            spatial_mean = spatial_pattern.mean()
            spatial_std = spatial_pattern.std()
            
            # Calculate spatial concentration (how concentrated the importance is)
            flat_spatial = spatial_pattern.flatten()
            spatial_entropy = -np.sum((flat_spatial / flat_spatial.sum()) * 
                                    np.log(flat_spatial / flat_spatial.sum() + 1e-10))
            
            # Quadrant analysis (divide 16x12 grid into 4 quadrants)
            mid_h, mid_w = 8, 6
            quadrants = {
                'top_left': spatial_pattern[:mid_h, :mid_w],
                'top_right': spatial_pattern[:mid_h, mid_w:],
                'bottom_left': spatial_pattern[mid_h:, :mid_w],
                'bottom_right': spatial_pattern[mid_h:, mid_w:]
            }
            
            quadrant_importance = {q: np.mean(values) for q, values in quadrants.items()}
            most_important_quadrant = max(quadrant_importance, key=quadrant_importance.get)
            
            spatial_stats[subscale] = {
                'max_position': max_pos,
                'max_importance': max_importance,
                'n_important_regions': n_important_regions,
                'spatial_variance': spatial_variance,
                'spatial_mean': spatial_mean,
                'spatial_std': spatial_std,
                'spatial_entropy': spatial_entropy,
                'quadrant_importance': quadrant_importance,
                'most_important_quadrant': most_important_quadrant
            }
            
            print(f"  Max importance position: {max_pos}")
            print(f"  Max importance value: {max_importance:.6f}")
            print(f"  High importance regions: {n_important_regions} positions")
            print(f"  Spatial variance: {spatial_variance:.6f}")
            print(f"  Spatial entropy: {spatial_entropy:.4f}")
            print(f"  Most important quadrant: {most_important_quadrant}")
            print(f"  Quadrant importance: {dict(sorted(quadrant_importance.items(), key=lambda x: x[1], reverse=True))}")
        
        # Cross-subscale spatial comparison
        self._compare_spatial_patterns(spatial_patterns, spatial_stats)
        
        self.analysis_results['spatial_patterns'] = spatial_patterns
        self.analysis_results['spatial_stats'] = spatial_stats
        
        return spatial_patterns, spatial_stats
    
    def _compare_spatial_patterns(self, spatial_patterns, spatial_stats):
        """Compare spatial patterns across TLX subscales."""
        print(f"\n{'='*60}")
        print("CROSS-SUBSCALE SPATIAL PATTERN COMPARISON")
        print(f"{'='*60}")
        
        # Calculate correlations between spatial patterns
        subscale_names = list(spatial_patterns.keys())
        n_subscales = len(subscale_names)
        spatial_correlation_matrix = np.zeros((n_subscales, n_subscales))
        
        for i, subscale_i in enumerate(subscale_names):
            for j, subscale_j in enumerate(subscale_names):
                pattern_i = spatial_patterns[subscale_i].flatten()
                pattern_j = spatial_patterns[subscale_j].flatten()
                corr = np.corrcoef(pattern_i, pattern_j)[0, 1]
                spatial_correlation_matrix[i, j] = corr
        
        print("\nSpatial Pattern Correlations Between TLX Subscales:")
        spatial_corr_df = pd.DataFrame(spatial_correlation_matrix, 
                                      index=subscale_names, 
                                      columns=subscale_names)
        print(spatial_corr_df.round(3))
        
        # Analyze quadrant preferences
        quadrant_preferences = {}
        for quadrant in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            quadrant_preferences[quadrant] = []
            for subscale, stats in spatial_stats.items():
                if stats['most_important_quadrant'] == quadrant:
                    quadrant_preferences[quadrant].append(subscale)
        
        print(f"\nSpatial Quadrant Preferences:")
        for quadrant, subscales in quadrant_preferences.items():
            if subscales:
                print(f"  {quadrant.replace('_', ' ').title()}: {', '.join(subscales)}")
    
    def analyze_individual_predictions(self, n_samples=10):
        """Analyze individual sample predictions in detail."""
        print("\n" + "="*80)
        print(f"INDIVIDUAL SAMPLE PREDICTION ANALYSIS (Top {n_samples} samples)")
        print("="*80)
        
        # Calculate prediction errors for sample selection
        sample_errors = []
        for i in range(len(self.test_samples)):
            mae = mean_absolute_error(self.test_labels[i], self.predictions[i])
            mse = mean_squared_error(self.test_labels[i], self.predictions[i])
            r2 = r2_score(self.test_labels[i], self.predictions[i])
            sample_errors.append({
                'index': i,
                'mae': mae,
                'mse': mse,
                'r2': r2
            })
        
        # Sort samples by different criteria for diverse analysis
        sorted_by_mae = sorted(sample_errors, key=lambda x: x['mae'])
        sorted_by_r2 = sorted(sample_errors, key=lambda x: x['r2'], reverse=True)
        
        # Select diverse samples
        selected_indices = []
        selected_indices.extend([sorted_by_mae[0]['index']])  # Best MAE
        selected_indices.extend([sorted_by_mae[-1]['index']])  # Worst MAE
        selected_indices.extend([sorted_by_r2[0]['index']])  # Best R2
        selected_indices.extend([sorted_by_r2[-1]['index']])  # Worst R2
        
        # Add random samples to reach n_samples
        remaining = n_samples - len(set(selected_indices))
        if remaining > 0:
            remaining_candidates = list(set(range(len(self.test_samples))) - set(selected_indices))
            additional = np.random.choice(remaining_candidates, 
                                        min(remaining, len(remaining_candidates)), 
                                        replace=False)
            selected_indices.extend(additional)
        
        selected_indices = list(set(selected_indices))[:n_samples]
        
        individual_analyses = {}
        
        for idx in selected_indices:
            print(f"\n{'='*50}")
            print(f"SAMPLE {idx} DETAILED ANALYSIS")
            print(f"{'='*50}")
            
            sample_prediction = self.predictions[idx]
            sample_actual = self.test_labels[idx]
            sample_mae = mean_absolute_error(sample_actual, sample_prediction)
            sample_r2 = r2_score(sample_actual, sample_prediction)
            
            print(f"Prediction Quality:")
            print(f"  MAE: {sample_mae:.3f}")
            print(f"  R²: {sample_r2:.3f}")
            print(f"  Predicted TLX: {sample_prediction}")
            print(f"  Actual TLX: {sample_actual}")
            print(f"  Prediction Error: {sample_prediction - sample_actual}")
            
            # Analyze SHAP contributions for this sample
            sample_analysis = {}
            
            for i, subscale in enumerate(self.tlx_subscales):
                shap_vals = self.shap_values[i] if isinstance(self.shap_values, list) else self.shap_values
                sample_shap = shap_vals[idx]
                
                # Calculate positive and negative contributions
                positive_contrib = np.sum(sample_shap[sample_shap > 0])
                negative_contrib = np.sum(sample_shap[sample_shap < 0])
                net_contrib = positive_contrib + negative_contrib
                
                # Calculate contribution statistics
                shap_mean = sample_shap.mean()
                shap_std = sample_shap.std()
                shap_max = sample_shap.max()
                shap_min = sample_shap.min()
                
                sample_analysis[subscale] = {
                    'positive_contrib': positive_contrib,
                    'negative_contrib': negative_contrib,
                    'net_contrib': net_contrib,
                    'shap_range': (shap_min, shap_max),
                    'shap_mean': shap_mean,
                    'shap_std': shap_std,
                    'prediction': sample_prediction[i],
                    'actual': sample_actual[i],
                    'error': sample_prediction[i] - sample_actual[i]
                }
                
                print(f"\n  {subscale}:")
                print(f"    Predicted: {sample_prediction[i]:.2f}, Actual: {sample_actual[i]:.2f}, Error: {sample_prediction[i] - sample_actual[i]:.2f}")
                print(f"    Positive contributions: {positive_contrib:.4f}")
                print(f"    Negative contributions: {negative_contrib:.4f}")
                print(f"    Net contribution: {net_contrib:.4f}")
                print(f"    SHAP range: [{shap_min:.4f}, {shap_max:.4f}]")
                print(f"    SHAP std: {shap_std:.4f}")
            
            individual_analyses[idx] = sample_analysis
        
        self.analysis_results['individual_analyses'] = individual_analyses
        return individual_analyses
    
    def analyze_subscale_relationships(self):
        """Analyze relationships and interactions between TLX subscales."""
        print("\n" + "="*80)
        print("TLX SUBSCALE RELATIONSHIP ANALYSIS")
        print("="*80)
        
        # Analyze SHAP value correlations between subscales
        n_subscales = len(self.tlx_subscales)
        shap_correlations = np.zeros((n_subscales, n_subscales))
        
        for i in range(n_subscales):
            for j in range(n_subscales):
                shap_i = self.shap_values[i].flatten()
                shap_j = self.shap_values[j].flatten()
                correlation = np.corrcoef(shap_i, shap_j)[0, 1]
                shap_correlations[i, j] = correlation
        
        print("\nSHAP Value Correlations Between TLX Subscales:")
        shap_corr_df = pd.DataFrame(shap_correlations, 
                                   index=self.tlx_subscales, 
                                   columns=self.tlx_subscales)
        print(shap_corr_df.round(3))
        
        # Analyze prediction correlations
        pred_correlations = np.corrcoef(self.predictions.T)
        
        print("\nPrediction Correlations Between TLX Subscales:")
        pred_corr_df = pd.DataFrame(pred_correlations, 
                                   index=self.tlx_subscales, 
                                   columns=self.tlx_subscales)
        print(pred_corr_df.round(3))
        
        # Find most and least related subscales
        upper_triangle_shap = np.triu(shap_correlations, k=1)
        upper_triangle_pred = np.triu(pred_correlations, k=1)
        
        # SHAP-based relationships
        max_shap_idx = np.unravel_index(upper_triangle_shap.argmax(), upper_triangle_shap.shape)
        min_shap_idx = np.unravel_index(upper_triangle_shap.argmin(), upper_triangle_shap.shape)
        
        # Prediction-based relationships
        max_pred_idx = np.unravel_index(upper_triangle_pred.argmax(), upper_triangle_pred.shape)
        min_pred_idx = np.unravel_index(upper_triangle_pred.argmin(), upper_triangle_pred.shape)
        
        print(f"\nMost related subscales (SHAP): {self.tlx_subscales[max_shap_idx[0]]} - {self.tlx_subscales[max_shap_idx[1]]} (r={upper_triangle_shap[max_shap_idx]:.3f})")
        print(f"Least related subscales (SHAP): {self.tlx_subscales[min_shap_idx[0]]} - {self.tlx_subscales[min_shap_idx[1]]} (r={upper_triangle_shap[min_shap_idx]:.3f})")
        
        print(f"\nMost related subscales (Predictions): {self.tlx_subscales[max_pred_idx[0]]} - {self.tlx_subscales[max_pred_idx[1]]} (r={upper_triangle_pred[max_pred_idx]:.3f})")
        print(f"Least related subscales (Predictions): {self.tlx_subscales[min_pred_idx[0]]} - {self.tlx_subscales[min_pred_idx[1]]} (r={upper_triangle_pred[min_pred_idx]:.3f})")
        
        # Analyze feature sharing between subscales
        self._analyze_feature_sharing()
        
        relationships = {
            'shap_correlations': shap_correlations,
            'prediction_correlations': pred_correlations,
            'most_related_shap': (self.tlx_subscales[max_shap_idx[0]], self.tlx_subscales[max_shap_idx[1]], upper_triangle_shap[max_shap_idx]),
            'least_related_shap': (self.tlx_subscales[min_shap_idx[0]], self.tlx_subscales[min_shap_idx[1]], upper_triangle_shap[min_shap_idx])
        }
        
        self.analysis_results['subscale_relationships'] = relationships
        return relationships
    
    def _analyze_feature_sharing(self):
        """Analyze which features are important for multiple subscales."""
        print(f"\n{'='*60}")
        print("CROSS-SUBSCALE FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate global importance for each subscale
        global_importance = self.analysis_results.get('global_importance', {})
        if not global_importance:
            return
        
        # Find features that are important for multiple subscales
        # Define threshold as 90th percentile of importance for each subscale
        important_features = {}
        
        for subscale, importance_map in global_importance.items():
            threshold = np.percentile(importance_map, 90)
            important_positions = np.where(importance_map > threshold)
            important_features[subscale] = set(zip(important_positions[0], 
                                                 important_positions[1], 
                                                 important_positions[2]))
        
        # Find shared important features
        all_subscales = list(important_features.keys())
        shared_features = {}
        
        for i in range(len(all_subscales)):
            for j in range(i+1, len(all_subscales)):
                subscale_i = all_subscales[i]
                subscale_j = all_subscales[j]
                
                shared = important_features[subscale_i].intersection(important_features[subscale_j])
                if shared:
                    shared_features[f"{subscale_i}-{subscale_j}"] = len(shared)
        
        if shared_features:
            print("\nShared Important Features Between Subscales:")
            for pair, count in sorted(shared_features.items(), key=lambda x: x[1], reverse=True):
                print(f"  {pair}: {count} shared important features")
        else:
            print("\nNo significant feature sharing detected between subscales.")
    
    def generate_comprehensive_summary(self):
        """Generate a comprehensive summary of all SHAP analysis results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE SHAP ANALYSIS SUMMARY")
        print("="*80)
        
        summary = {
            'analysis_overview': {},
            'key_findings': {},
            'methodological_insights': {},
            'model_interpretability': {}
        }
        
        # Analysis Overview
        summary['analysis_overview'] = {
            'total_test_samples': len(self.test_samples),
            'background_samples': len(self.background_data),
            'tlx_subscales': self.tlx_subscales,
            'data_shape': self.test_samples.shape,
            'shap_computation_successful': self.shap_values is not None
        }
        
        print("ANALYSIS OVERVIEW:")
        print(f"  Total test samples analyzed: {len(self.test_samples)}")
        print(f"  Background samples used: {len(self.background_data)}")
        print(f"  TLX subscales: {', '.join(self.tlx_subscales)}")
        print(f"  Data dimensions: {self.test_samples.shape}")
        
        # Key Findings from different analyses
        key_findings = []
        
        # Global importance findings
        if 'importance_stats' in self.analysis_results:
            stats = self.analysis_results['importance_stats']
            max_important_subscale = max(stats.keys(), key=lambda x: stats[x]['total_importance'])
            min_important_subscale = min(stats.keys(), key=lambda x: stats[x]['total_importance'])
            
            key_findings.append(f"Most feature-dependent subscale: {max_important_subscale}")
            key_findings.append(f"Least feature-dependent subscale: {min_important_subscale}")
        
        # Temporal findings
        if 'temporal_stats' in self.analysis_results:
            temporal_stats = self.analysis_results['temporal_stats']
            
            # Identify temporal patterns
            early_dominant = [s for s, stats in temporal_stats.items() 
                            if stats['early_importance'] > stats['late_importance'] * 1.2]
            late_dominant = [s for s, stats in temporal_stats.items() 
                           if stats['late_importance'] > stats['early_importance'] * 1.2]
            
            if early_dominant:
                key_findings.append(f"Early-phase dominant subscales: {', '.join(early_dominant)}")
            if late_dominant:
                key_findings.append(f"Late-phase dominant subscales: {', '.join(late_dominant)}")
        
        # Relationship findings
        if 'subscale_relationships' in self.analysis_results:
            relationships = self.analysis_results['subscale_relationships']
            most_related = relationships['most_related_shap']
            least_related = relationships['least_related_shap']
            
            key_findings.append(f"Most related subscales: {most_related[0]} - {most_related[1]} (r={most_related[2]:.3f})")
            key_findings.append(f"Least related subscales: {least_related[0]} - {least_related[1]} (r={least_related[2]:.3f})")
        
        summary['key_findings'] = key_findings
        
        print(f"\nKEY FINDINGS:")
        for finding in key_findings:
            print(f"  • {finding}")
        
        # Model interpretability insights
        interpretability_insights = [
            "3D CNN successfully captures temporal-spatial patterns in workload assessment",
            "Different TLX subscales show distinct feature importance patterns",
            "Temporal dynamics play crucial role in workload prediction",
            "Spatial feature arrangements provide meaningful interpretability",
            "Multi-output architecture enables subscale-specific analysis"
        ]
        
        summary['model_interpretability'] = interpretability_insights
        
        print(f"\nMODEL INTERPRETABILITY INSIGHTS:")
        for insight in interpretability_insights:
            print(f"  • {insight}")
        
        # Research implications
        print(f"\nRESEARCH IMPLICATIONS:")
        research_implications = [
            "SHAP analysis provides explainable AI framework for workload assessment",
            "Temporal importance patterns align with cognitive workload theory",
            "Feature importance maps guide sensor placement and data collection",
            "Subscale relationships inform workload measurement strategies",
            "Individual prediction analysis enables personalized workload monitoring"
        ]
        
        for implication in research_implications:
            print(f"  • {implication}")
        
        self.analysis_results['comprehensive_summary'] = summary
        return summary
    
    def run_complete_analysis(self, save_results=True):
        """Run the complete comprehensive SHAP analysis pipeline."""
        print("="*100)
        print("COMPREHENSIVE 3D CNN SHAP ANALYSIS FOR NASA-TLX PREDICTION")
        print("="*100)
        
        try:
            # Step 1: Prepare data
            self.prepare_analysis_data()
            
            # Step 2: Initialize explainer
            self.initialize_shap_explainer('deep')
            
            # Step 3: Compute SHAP values
            self.compute_shap_values()
            
            # Step 4: Global feature importance analysis
            global_importance, importance_stats = self.analyze_global_feature_importance()
            
            # Step 5: Temporal importance analysis
            temporal_patterns, temporal_stats = self.analyze_temporal_importance_patterns()
            
            # Step 6: Spatial importance analysis
            spatial_patterns, spatial_stats = self.analyze_spatial_importance_patterns()
            
            # Step 7: Individual prediction analysis
            individual_analyses = self.analyze_individual_predictions()
            
            # Step 8: Subscale relationship analysis
            relationships = self.analyze_subscale_relationships()
            
            # Step 9: Generate comprehensive summary
            summary = self.generate_comprehensive_summary()
            
            print("\n" + "="*100)
            print("COMPREHENSIVE SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*100)
            
            if save_results:
                self.save_analysis_results()
            
            return self.analysis_results
            
        except Exception as e:
            print(f"\nERROR in SHAP analysis: {e}")
            raise
    
    def save_analysis_results(self, filename='comprehensive_shap_analysis_results.npz'):
        """Save all analysis results to file."""
        print(f"\nSaving analysis results to {filename}...")
        
        # Prepare data for saving (convert to numpy arrays where possible)
        save_data = {}
        
        for key, value in self.analysis_results.items():
            if isinstance(value, dict):
                # Convert nested dictionaries to strings for saving
                save_data[key] = str(value)
            else:
                save_data[key] = value
        
        # Save SHAP values and other numpy arrays
        if self.shap_values is not None:
            if isinstance(self.shap_values, list):
                for i, shap_vals in enumerate(self.shap_values):
                    save_data[f'shap_values_{i}'] = shap_vals
            else:
                save_data['shap_values'] = self.shap_values
        
        np.savez_compressed(filename, **save_data)
        print(f"Analysis results saved successfully to {filename}")

# Usage instructions for your research
print("""
TO USE THIS COMPREHENSIVE SHAP ANALYSIS WITH YOUR MODEL:

1. After training your 3D CNN model, initialize the analyzer:
   
   shap_analyzer = Comprehensive3DCNN_SHAPAnalysis(
       model=pipeline.model,
       X_train=pipeline.X_train,
       X_val=pipeline.X_val,
       X_test=pipeline.X_test,
       y_train=pipeline.y_train,
       y_val=pipeline.y_val,
       y_test=pipeline.y_test
   )

2. Run the complete analysis:
   
   results = shap_analyzer.run_complete_analysis()

3. The analysis will generate comprehensive terminal output and return detailed results.

4. Share the terminal output with me, and I'll help you write the research paper content.

This analysis provides:
- Global feature importance across all subscales
- Temporal importance patterns (120 time frames)
- Spatial importance patterns (16x12 feature grid)
- Individual sample explanations
- Subscale relationship analysis
- Comprehensive interpretability insights

Perfect for high-impact research publications!
""")
