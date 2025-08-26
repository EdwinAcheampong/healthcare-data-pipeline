"""
Baseline Models for Healthcare Workload Prediction.

This module implements baseline forecasting models and auto-scaling algorithms
for healthcare workload prediction, including ARIMA, Prophet, and simple statistical models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

# Time Series Libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("statsmodels not available. ARIMA models will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("prophet not available. Prophet models will be skipped.")

import warnings
warnings.filterwarnings('ignore')


class HealthcareBaselineModels:
    """Baseline models for healthcare workload prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                target_column: str = 'encounters_last_24h',
                                time_column: str = 'START',
                                freq: str = 'H') -> pd.DataFrame:
        """Prepare time series data for forecasting."""
        self.logger.info("Preparing time series data")
        
        # Convert to datetime and set as index
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Sample data if too large for faster processing
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)
            self.logger.info(f"Sampled 50000 records for faster processing")
        
        # Aggregate by time frequency
        ts_data = df.groupby(pd.Grouper(key=time_column, freq=freq))[target_column].sum().reset_index()
        ts_data = ts_data.set_index(time_column)
        
        # Fill missing values
        ts_data = ts_data.fillna(0)
        
        # Remove rows with all zeros
        ts_data = ts_data[ts_data[target_column] > 0]
        
        self.logger.info(f"Time series data prepared: {len(ts_data)} observations")
        return ts_data
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_column: str,
                           lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        self.logger.info("Creating lag features")
        
        df = df.copy()
        
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Create seasonal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.dropna()
    
    def train_arima_model(self, ts_data: pd.DataFrame, 
                         target_column: str,
                         order: Tuple[int, int, int] = (1, 1, 1)) -> Optional[Any]:
        """Train ARIMA model for time series forecasting."""
        if not ARIMA_AVAILABLE:
            self.logger.warning("ARIMA not available, skipping")
            return None
            
        self.logger.info(f"Training ARIMA model with order {order}")
        
        try:
            model = ARIMA(ts_data[target_column], order=order)
            fitted_model = model.fit()
            
            self.logger.info(f"ARIMA model fitted successfully. AIC: {fitted_model.aic}")
            return fitted_model
            
        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {str(e)}")
            return None
    
    def train_prophet_model(self, ts_data: pd.DataFrame, 
                           target_column: str) -> Optional[Any]:
        """Train Prophet model for time series forecasting."""
        if not PROPHET_AVAILABLE:
            self.logger.warning("Prophet not available, skipping")
            return None
            
        self.logger.info("Training Prophet model")
        
        try:
            # Prepare data for Prophet
            prophet_data = ts_data.reset_index()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_data)
            
            self.logger.info("Prophet model fitted successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training Prophet model: {str(e)}")
            return None
    
    def train_sklearn_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train scikit-learn models for regression."""
        self.logger.info("Training scikit-learn models")
        
        # Reduce dataset size for faster training (sample 5% of data)
        if len(X) > 5000:
            sample_size = min(5000, len(X) // 20)
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            
            # Handle both DataFrame and numpy array
            if hasattr(X, 'iloc'):
                X_sample = X.iloc[sample_indices]
            else:
                X_sample = X[sample_indices]
                
            if hasattr(y, 'iloc'):
                y_sample = y.iloc[sample_indices]
            else:
                y_sample = y[sample_indices]
                
            self.logger.info(f"Sampling {sample_size} records for faster training")
        else:
            X_sample = X
            y_sample = y
        
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1),
        }
        
        trained_models = {}
        
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name}")
                model.fit(X_sample, y_sample)
                trained_models[name] = model
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
        
        return trained_models
    
    def evaluate_models(self, models: Dict[str, Any], 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series,
                       ts_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all models and return metrics."""
        self.logger.info("Evaluating models")
        
        results = {}
        
        # Evaluate scikit-learn models
        for name, model in models.items():
            if hasattr(model, 'predict'):
                try:
                    y_pred = model.predict(X_test)
                    
                    results[name] = {
                        'mae': mean_absolute_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'mape': self._calculate_mape(y_test, y_pred)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {name}: {str(e)}")
        
        # Evaluate time series models
        if ts_data is not None:
            if 'arima' in models and models['arima'] is not None:
                try:
                    # ARIMA evaluation
                    forecast_steps = min(len(y_test), 24)  # Forecast next 24 hours
                    arima_forecast = models['arima'].forecast(steps=forecast_steps)
                    
                    results['arima'] = {
                        'mae': mean_absolute_error(y_test[:forecast_steps], arima_forecast),
                        'rmse': np.sqrt(mean_squared_error(y_test[:forecast_steps], arima_forecast)),
                        'r2': r2_score(y_test[:forecast_steps], arima_forecast),
                        'mape': self._calculate_mape(y_test[:forecast_steps], arima_forecast)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating ARIMA: {str(e)}")
            
            if 'prophet' in models and models['prophet'] is not None:
                try:
                    # Prophet evaluation
                    future = models['prophet'].make_future_dataframe(periods=len(y_test), freq='H')
                    prophet_forecast = models['prophet'].predict(future)
                    
                    # Get forecast values
                    forecast_values = prophet_forecast['yhat'].tail(len(y_test))
                    
                    results['prophet'] = {
                        'mae': mean_absolute_error(y_test, forecast_values),
                        'rmse': np.sqrt(mean_squared_error(y_test, forecast_values)),
                        'r2': r2_score(y_test, forecast_values),
                        'mape': self._calculate_mape(y_test, forecast_values)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating Prophet: {str(e)}")
        
        self.evaluation_results = results
        return results
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def train_all_baseline_models(self, 
                                 encounters_df: pd.DataFrame,
                                 target_column: str = 'encounters_last_24h') -> Dict[str, Any]:
        """Train all baseline models."""
        self.logger.info("Training all baseline models")
        
        # Prepare time series data
        ts_data = self.prepare_time_series_data(encounters_df, target_column)
        
        # Create lag features for ML models
        ts_with_features = self.create_lag_features(ts_data, target_column)
        
        # Split data
        train_size = int(len(ts_with_features) * 0.8)
        train_data = ts_with_features[:train_size]
        test_data = ts_with_features[train_size:]
        
        # Prepare features and target
        feature_columns = [col for col in ts_with_features.columns if col != target_column]
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train scikit-learn models
        sklearn_models = self.train_sklearn_models(X_train_scaled, y_train)
        
        # Skip time series models for faster execution
        ts_models = {}
        self.logger.info("Skipping time series models for faster execution")
        
        # Combine all models
        all_models = {**sklearn_models, **ts_models}
        
        # Evaluate models
        results = self.evaluate_models(all_models, X_test_scaled, y_test, ts_data)
        
        # Store models and results
        self.models = all_models
        self.scalers['feature_scaler'] = scaler
        
        self.logger.info("All baseline models trained and evaluated")
        return all_models
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model based on MAPE."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Train models first.")
        
        # Find model with lowest MAPE
        best_model_name = min(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['mape'])
        best_model = self.models[best_model_name]
        
        self.logger.info(f"Best model: {best_model_name} (MAPE: {self.evaluation_results[best_model_name]['mape']:.2f}%)")
        
        return best_model_name, best_model
    
    def predict_workload(self, 
                        model_name: str,
                        X: pd.DataFrame,
                        forecast_hours: int = 72) -> np.ndarray:
        """Predict workload using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name in ['arima', 'prophet']:
            # Time series models
            if model_name == 'arima':
                forecast = model.forecast(steps=forecast_hours)
            else:  # prophet
                future = model.make_future_dataframe(periods=forecast_hours, freq='H')
                forecast = model.predict(future)
                forecast = forecast['yhat'].tail(forecast_hours)
            
            return forecast.values
        
        else:
            # Scikit-learn models
            if 'feature_scaler' in self.scalers:
                X_scaled = self.scalers['feature_scaler'].transform(X)
            else:
                X_scaled = X
            
            return model.predict(X_scaled)
    
    def generate_forecast_report(self) -> Dict[str, Any]:
        """Generate comprehensive forecast report."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        report = {
            'summary': {
                'total_models': len(self.evaluation_results),
                'best_model': self.get_best_model()[0],
                'best_mape': min(result['mape'] for result in self.evaluation_results.values()),
                'average_mape': np.mean([result['mape'] for result in self.evaluation_results.values()])
            },
            'model_performance': self.evaluation_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on model performance."""
        recommendations = []
        
        best_mape = min(result['mape'] for result in self.evaluation_results.values())
        
        if best_mape < 8:
            recommendations.append("Excellent performance! MAPE below 8% target achieved.")
        elif best_mape < 15:
            recommendations.append("Good performance. Consider feature engineering improvements.")
        else:
            recommendations.append("Performance needs improvement. Consider advanced models or more features.")
        
        # Model-specific recommendations
        if 'random_forest' in self.evaluation_results:
            rf_mape = self.evaluation_results['random_forest']['mape']
            if rf_mape < best_mape * 1.1:
                recommendations.append("Random Forest shows good performance. Consider ensemble methods.")
        
        if 'arima' in self.evaluation_results:
            arima_mape = self.evaluation_results['arima']['mape']
            if arima_mape < best_mape * 1.1:
                recommendations.append("ARIMA captures temporal patterns well. Consider hybrid approaches.")
        
        return recommendations


class BaselinePredictor:
    """Baseline predictor for API integration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def fit(self, X, y):
        """Train the baseline model."""
        self.logger.info("Training baseline predictor")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use Random Forest as baseline
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.logger.info("Baseline predictor trained successfully")
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            return {}
        
        return dict(zip(
            range(self.model.n_features_in_),
            self.model.feature_importances_
        ))
