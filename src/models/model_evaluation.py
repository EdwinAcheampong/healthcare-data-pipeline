"""
Model Evaluation and Comparison for Healthcare Workload Prediction.

This module provides comprehensive evaluation and comparison of baseline and advanced models,
including performance metrics, statistical tests, and detailed reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# ML Libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import our model classes
from .baseline_models import HealthcareBaselineModels
from .advanced_models import AdvancedHealthcareModels
from .feature_engineering import FeatureEngineer

import warnings
warnings.filterwarnings('ignore')


class HealthcareModelEvaluator:
    """Comprehensive model evaluation and comparison for healthcare workload prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_models = HealthcareBaselineModels()
        self.advanced_models = AdvancedHealthcareModels()
        self.feature_engineer = FeatureEngineer()
        
        self.all_results = {}
        self.comparison_results = {}
        
    def run_complete_evaluation(self, 
                              encounters_df: pd.DataFrame,
                              patients_df: pd.DataFrame,
                              conditions_df: pd.DataFrame,
                              medications_df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete evaluation of all models."""
        self.logger.info("Starting complete model evaluation")
        
        # Step 1: Feature Engineering
        self.logger.info("Step 1: Feature Engineering")
        feature_df = self.feature_engineer.engineer_features(
            encounters_df, patients_df, conditions_df, medications_df
        )
        
        # Step 2: Train Baseline Models
        self.logger.info("Step 2: Training Baseline Models")
        baseline_results = self.baseline_models.train_all_baseline_models(feature_df)
        baseline_evaluation = self.baseline_models.evaluation_results
        
        # Step 3: Train Advanced Models
        self.logger.info("Step 3: Training Advanced Models")
        advanced_results = self.advanced_models.train_all_advanced_models(feature_df)
        advanced_evaluation = self.advanced_models.evaluation_results
        
        # Step 4: Compare Models
        self.logger.info("Step 4: Model Comparison")
        comparison = self.compare_models(baseline_evaluation, advanced_evaluation)
        
        # Step 5: Generate Comprehensive Report
        self.logger.info("Step 5: Generating Report")
        report = self.generate_comprehensive_report(
            baseline_evaluation, advanced_evaluation, comparison
        )
        
        # Store results
        self.all_results = {
            'baseline': baseline_evaluation,
            'advanced': advanced_evaluation,
            'comparison': comparison,
            'report': report
        }
        
        self.logger.info("Complete model evaluation finished")
        return self.all_results
    
    def compare_models(self, 
                      baseline_results: Dict[str, Dict[str, float]],
                      advanced_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare baseline and advanced models."""
        self.logger.info("Comparing baseline and advanced models")
        
        comparison = {
            'summary': {},
            'detailed_comparison': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Combine all results
        all_models = {**baseline_results, **advanced_results}
        
        # Find best models in each category
        baseline_models = list(baseline_results.keys())
        advanced_models = list(advanced_results.keys())
        
        if baseline_models:
            best_baseline = min(baseline_models, 
                              key=lambda x: baseline_results[x]['mape'])
            best_baseline_mape = baseline_results[best_baseline]['mape']
        else:
            best_baseline = None
            best_baseline_mape = float('inf')
        
        if advanced_models:
            best_advanced = min(advanced_models, 
                              key=lambda x: advanced_results[x]['mape'])
            best_advanced_mape = advanced_results[best_advanced]['mape']
        else:
            best_advanced = None
            best_advanced_mape = float('inf')
        
        # Overall best model
        if best_baseline_mape < best_advanced_mape:
            overall_best = best_baseline
            overall_best_mape = best_baseline_mape
            best_category = 'baseline'
        else:
            overall_best = best_advanced
            overall_best_mape = best_advanced_mape
            best_category = 'advanced'
        
        # Summary statistics
        comparison['summary'] = {
            'total_models': len(all_models),
            'baseline_models': len(baseline_models),
            'advanced_models': len(advanced_models),
            'best_baseline': best_baseline,
            'best_baseline_mape': best_baseline_mape,
            'best_advanced': best_advanced,
            'best_advanced_mape': best_advanced_mape,
            'overall_best': overall_best,
            'overall_best_mape': overall_best_mape,
            'best_category': best_category
        }
        
        # Detailed comparison
        comparison['detailed_comparison'] = {}
        for metric in ['mape', 'mae', 'rmse', 'r2']:
            metric_values = {model: results[metric] for model, results in all_models.items()}
            
            comparison['detailed_comparison'][metric] = {
                'values': metric_values,
                'best_model': min(metric_values.keys(), key=lambda x: metric_values[x]),
                'best_value': min(metric_values.values()),
                'average_value': np.mean(list(metric_values.values())),
                'std_value': np.std(list(metric_values.values()))
            }
        
        # Statistical significance tests
        if len(baseline_models) > 0 and len(advanced_models) > 0:
            baseline_mape = [baseline_results[model]['mape'] for model in baseline_models]
            advanced_mape = [advanced_results[model]['mape'] for model in advanced_models]
            
            # T-test for difference in means
            t_stat, p_value = stats.ttest_ind(baseline_mape, advanced_mape)
            
            comparison['statistical_tests'] = {
                't_test': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'baseline_mean': np.mean(baseline_mape),
                    'advanced_mean': np.mean(advanced_mape)
                }
            }
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_comparison_recommendations(
            comparison['summary'], comparison['detailed_comparison']
        )
        
        return comparison
    
    def _generate_comparison_recommendations(self, 
                                           summary: Dict[str, Any],
                                           detailed_comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        # Overall performance assessment
        if summary['overall_best_mape'] < 8:
            recommendations.append("🎯 EXCELLENT: Target MAPE <8% achieved! Ready for production deployment.")
        elif summary['overall_best_mape'] < 12:
            recommendations.append("✅ GOOD: Performance close to target. Consider fine-tuning for production.")
        elif summary['overall_best_mape'] < 20:
            recommendations.append("⚠️ FAIR: Performance needs improvement. Consider additional feature engineering.")
        else:
            recommendations.append("❌ POOR: Performance significantly below target. Review data quality and model selection.")
        
        # Category comparison
        if summary['best_category'] == 'advanced':
            if summary['best_advanced_mape'] < summary['best_baseline_mape'] * 0.9:
                recommendations.append("🚀 Advanced models significantly outperform baselines. Proceed with advanced approach.")
            else:
                recommendations.append("📊 Advanced models show marginal improvement. Consider cost-benefit analysis.")
        else:
            recommendations.append("💡 Baseline models perform well. Advanced models may not be necessary for current requirements.")
        
        # Model-specific recommendations
        best_model = summary['overall_best']
        if 'tcn_lstm' in best_model.lower():
            recommendations.append("🧠 TCN-LSTM shows strong performance. Consider attention mechanism optimization.")
        elif 'stacking' in best_model.lower():
            recommendations.append("🎯 Stacking ensemble effective. Consider adding more diverse base models.")
        elif 'xgboost' in best_model.lower() or 'lightgbm' in best_model.lower():
            recommendations.append("🌳 Gradient boosting performs well. Consider hyperparameter optimization.")
        elif 'arima' in best_model.lower():
            recommendations.append("📈 ARIMA captures temporal patterns. Consider hybrid approaches.")
        
        # Statistical significance
        if 'statistical_tests' in self.comparison_results:
            t_test = self.comparison_results['statistical_tests'].get('t_test', {})
            if t_test.get('significant', False):
                if t_test['advanced_mean'] < t_test['baseline_mean']:
                    recommendations.append("📊 Statistically significant improvement with advanced models.")
                else:
                    recommendations.append("📊 Advanced models show statistically significant difference (not necessarily better).")
        
        return recommendations
    
    def generate_comprehensive_report(self,
                                    baseline_results: Dict[str, Dict[str, float]],
                                    advanced_results: Dict[str, Dict[str, float]],
                                    comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        self.logger.info("Generating comprehensive report")
        
        report = {
            'executive_summary': self._generate_executive_summary(comparison),
            'detailed_analysis': self._generate_detailed_analysis(baseline_results, advanced_results),
            'performance_metrics': self._generate_performance_metrics(baseline_results, advanced_results),
            'model_rankings': self._generate_model_rankings(baseline_results, advanced_results),
            'recommendations': comparison.get('recommendations', []),
            'next_steps': self._generate_next_steps(comparison),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_executive_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = comparison['summary']
        
        return {
            'overall_performance': {
                'best_model': summary['overall_best'],
                'best_mape': summary['overall_best_mape'],
                'target_achieved': summary['overall_best_mape'] < 8,
                'performance_category': self._categorize_performance(summary['overall_best_mape'])
            },
            'model_comparison': {
                'baseline_best': summary['best_baseline'],
                'baseline_mape': summary['best_baseline_mape'],
                'advanced_best': summary['best_advanced'],
                'advanced_mape': summary['best_advanced_mape'],
                'improvement': summary['best_baseline_mape'] - summary['best_advanced_mape'] if summary['best_advanced_mape'] < summary['best_baseline_mape'] else 0
            },
            'deployment_readiness': {
                'ready_for_production': summary['overall_best_mape'] < 8,
                'recommended_model': summary['overall_best'],
                'confidence_level': self._calculate_confidence_level(summary['overall_best_mape'])
            }
        }
    
    def _categorize_performance(self, mape: float) -> str:
        """Categorize performance based on MAPE."""
        if mape < 5:
            return "Exceptional"
        elif mape < 8:
            return "Excellent"
        elif mape < 12:
            return "Good"
        elif mape < 20:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_confidence_level(self, mape: float) -> str:
        """Calculate confidence level for deployment."""
        if mape < 5:
            return "Very High"
        elif mape < 8:
            return "High"
        elif mape < 12:
            return "Medium"
        else:
            return "Low"
    
    def _generate_detailed_analysis(self,
                                  baseline_results: Dict[str, Dict[str, float]],
                                  advanced_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate detailed analysis."""
        return {
            'baseline_models': {
                'count': len(baseline_results),
                'models': list(baseline_results.keys()),
                'average_mape': np.mean([results['mape'] for results in baseline_results.values()]),
                'best_mape': min([results['mape'] for results in baseline_results.values()]) if baseline_results else float('inf')
            },
            'advanced_models': {
                'count': len(advanced_results),
                'models': list(advanced_results.keys()),
                'average_mape': np.mean([results['mape'] for results in advanced_results.values()]),
                'best_mape': min([results['mape'] for results in advanced_results.values()]) if advanced_results else float('inf')
            },
            'performance_gaps': {
                'baseline_to_target': max(0, min([results['mape'] for results in baseline_results.values()]) - 8) if baseline_results else float('inf'),
                'advanced_to_target': max(0, min([results['mape'] for results in advanced_results.values()]) - 8) if advanced_results else float('inf')
            }
        }
    
    def _generate_performance_metrics(self,
                                    baseline_results: Dict[str, Dict[str, float]],
                                    advanced_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate detailed performance metrics."""
        all_results = {**baseline_results, **advanced_results}
        
        metrics = {}
        for metric in ['mape', 'mae', 'rmse', 'r2']:
            values = [results[metric] for results in all_results.values()]
            metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return metrics
    
    def _generate_model_rankings(self,
                               baseline_results: Dict[str, Dict[str, float]],
                               advanced_results: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Generate model rankings by performance."""
        all_results = {**baseline_results, **advanced_results}
        
        # Rank by MAPE (lower is better)
        ranked_by_mape = sorted(all_results.keys(), 
                              key=lambda x: all_results[x]['mape'])
        
        # Rank by R² (higher is better)
        ranked_by_r2 = sorted(all_results.keys(), 
                            key=lambda x: all_results[x]['r2'], reverse=True)
        
        return {
            'by_mape': ranked_by_mape,
            'by_r2': ranked_by_r2,
            'top_3_mape': ranked_by_mape[:3],
            'top_3_r2': ranked_by_r2[:3]
        }
    
    def _generate_next_steps(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate next steps based on evaluation results."""
        next_steps = []
        summary = comparison['summary']
        
        if summary['overall_best_mape'] < 8:
            next_steps.extend([
                "✅ Proceed to Phase 2B (RL System Development)",
                "🔧 Implement model deployment pipeline",
                "📊 Set up monitoring and alerting systems",
                "🧪 Conduct A/B testing in production environment"
            ])
        elif summary['overall_best_mape'] < 12:
            next_steps.extend([
                "🔧 Fine-tune best performing model",
                "📈 Implement additional feature engineering",
                "🧪 Conduct hyperparameter optimization",
                "📊 Validate on additional datasets"
            ])
        else:
            next_steps.extend([
                "🔍 Investigate data quality issues",
                "📊 Conduct exploratory data analysis",
                "🧪 Try different model architectures",
                "📈 Improve feature engineering pipeline"
            ])
        
        # Model-specific next steps
        best_model = summary['overall_best']
        if 'tcn_lstm' in best_model.lower():
            next_steps.append("🧠 Optimize TCN-LSTM architecture and hyperparameters")
        elif 'ensemble' in best_model.lower():
            next_steps.append("🎯 Explore additional ensemble methods and base models")
        
        return next_steps
    
    def save_evaluation_results(self, output_path: str = "models/evaluation_results.json"):
        """Save evaluation results to file."""
        import json
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results
        serializable_results = {}
        for key, value in self.all_results.items():
            if isinstance(value, dict):
                serializable_results[key] = json.loads(
                    json.dumps(value, default=convert_numpy)
                )
            else:
                serializable_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Evaluation results saved to {output_file}")
    
    def generate_visualization_report(self, output_path: str = "models/visualization_report.html"):
        """Generate HTML visualization report."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # Create visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance Comparison', 'MAPE Distribution', 
                              'R² Score Comparison', 'Model Rankings'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Combine results for visualization
            all_results = {**self.all_results.get('baseline', {}), 
                          **self.all_results.get('advanced', {})}
            
            if all_results:
                models = list(all_results.keys())
                mape_values = [all_results[model]['mape'] for model in models]
                r2_values = [all_results[model]['r2'] for model in models]
                
                # Model Performance Comparison
                fig.add_trace(
                    go.Bar(x=models, y=mape_values, name='MAPE', marker_color='lightcoral'),
                    row=1, col=1
                )
                
                # MAPE Distribution
                fig.add_trace(
                    go.Histogram(x=mape_values, name='MAPE Distribution', nbinsx=10),
                    row=1, col=2
                )
                
                # R² Score Comparison
                fig.add_trace(
                    go.Bar(x=models, y=r2_values, name='R²', marker_color='lightblue'),
                    row=2, col=1
                )
                
                # Model Rankings (top 5)
                top_5_models = sorted(all_results.keys(), key=lambda x: all_results[x]['mape'])[:5]
                top_5_mape = [all_results[model]['mape'] for model in top_5_models]
                
                fig.add_trace(
                    go.Bar(x=top_5_models, y=top_5_mape, name='Top 5 Models', marker_color='lightgreen'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Healthcare Workload Prediction Model Evaluation",
                showlegend=False,
                height=800
            )
            
            # Save as HTML
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            fig.write_html(str(output_file))
            self.logger.info(f"Visualization report saved to {output_file}")
            
        except ImportError:
            self.logger.warning("Plotly not available. Skipping visualization report.")
        except Exception as e:
            self.logger.error(f"Error generating visualization report: {str(e)}")
