"""
Advanced ML Models for Healthcare Workload Prediction.

This module implements advanced machine learning models including TCN-LSTM hybrid architecture,
attention mechanisms, and ensemble methods for healthcare workload prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Deep learning models will be skipped.")

# ML Libraries
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time series processing."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, 
                 kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Temporal convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            self.conv_layers.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size//2)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch_size, sequence_length, hidden_size)
        return x


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence modeling."""
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super(AttentionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_size)
        return attended_output, attention_weights


class TCNLSTMModel(nn.Module):
    """TCN-LSTM hybrid model with attention mechanism."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 sequence_length: int = 24, dropout: float = 0.2):
        super(TCNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # TCN layers
        self.tcn = TemporalConvNet(input_size, hidden_size, num_layers=3, dropout=dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # TCN processing
        tcn_output = self.tcn(x)  # (batch_size, sequence_length, hidden_size)
        
        # LSTM processing
        lstm_output, _ = self.lstm(tcn_output)  # (batch_size, sequence_length, hidden_size)
        
        # Attention mechanism
        attended_output, attention_weights = self.attention(lstm_output)
        
        # Final prediction
        x = self.relu(self.fc1(attended_output))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attention_weights


class HealthcareDataset(Dataset):
    """Custom dataset for healthcare time series data."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, sequence_length: int = 24):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get sequence
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


class AdvancedHealthcareModels:
    """Advanced ML models for healthcare workload prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        
    def prepare_sequence_data(self, df: pd.DataFrame, 
                            feature_columns: List[str],
                            target_column: str,
                            sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sequence models."""
        self.logger.info("Preparing sequence data")
        
        # Select features and target
        features = df[feature_columns].values
        targets = df[target_column].values
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        self.scalers['sequence_scaler'] = scaler
        
        return features_scaled, targets
    
    def train_tcn_lstm_model(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           input_size: int,
                           hidden_size: int = 64,
                           num_epochs: int = 100,
                           batch_size: int = 32,
                           learning_rate: float = 0.001) -> Optional[Any]:
        """Train TCN-LSTM hybrid model."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, skipping TCN-LSTM")
            return None
        
        self.logger.info("Training TCN-LSTM model")
        
        try:
            # Create datasets
            train_dataset = HealthcareDataset(X_train, y_train, sequence_length=24)
            val_dataset = HealthcareDataset(X_val, y_val, sequence_length=24)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TCNLSTMModel(input_size, hidden_size).to(device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 20
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    predictions, _ = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        predictions, _ = model(batch_X)
                        val_loss += criterion(predictions, batch_y).item()
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                                   f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()
            
            self.logger.info("TCN-LSTM model training completed")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training TCN-LSTM model: {str(e)}")
            return None
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train ensemble models including XGBoost, LightGBM, and stacking."""
        self.logger.info("Training ensemble models")
        
        models = {}
        
        # XGBoost
        try:
            self.logger.info("Training XGBoost model")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {str(e)}")
        
        # LightGBM
        try:
            self.logger.info("Training LightGBM model")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            lgb_model.fit(X_train, y_train)
            models['lightgbm'] = lgb_model
        except Exception as e:
            self.logger.error(f"Error training LightGBM: {str(e)}")
        
        # Voting Regressor
        try:
            self.logger.info("Training Voting Regressor")
            estimators = [(name, model) for name, model in models.items()]
            if len(estimators) >= 2:
                voting_regressor = VotingRegressor(estimators=estimators)
                voting_regressor.fit(X_train, y_train)
                models['voting_regressor'] = voting_regressor
        except Exception as e:
            self.logger.error(f"Error training Voting Regressor: {str(e)}")
        
        return models
    
    def train_stacking_model(self, base_models: Dict[str, Any],
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Optional[Any]:
        """Train stacking ensemble model."""
        self.logger.info("Training Stacking Regressor")
        
        try:
            # Create stacking regressor
            estimators = [(name, model) for name, model in base_models.items()]
            meta_model = Ridge(alpha=1.0)
            
            stacking_regressor = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=3,
                n_jobs=-1
            )
            
            stacking_regressor.fit(X_train, y_train)
            
            self.logger.info("Stacking Regressor training completed")
            return stacking_regressor
            
        except Exception as e:
            self.logger.error(f"Error training Stacking Regressor: {str(e)}")
            return None
    
    def evaluate_advanced_models(self, models: Dict[str, Any],
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate advanced models."""
        self.logger.info("Evaluating advanced models")
        
        results = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                    
                    results[name] = {
                        'mae': mean_absolute_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'mape': self._calculate_mape(y_test, y_pred)
                    }
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
        
        self.evaluation_results = results
        return results
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if mask.sum() == 0:
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def train_all_advanced_models(self, 
                                feature_df: pd.DataFrame,
                                target_column: str = 'encounters_last_24h') -> Dict[str, Any]:
        """Train all advanced models."""
        self.logger.info("Training all advanced models")
        
        # Prepare data
        feature_columns = [col for col in feature_df.columns 
                         if col not in ['ENCOUNTER', 'PATIENT', 'START', target_column]]
        
        # Split data
        train_size = int(len(feature_df) * 0.7)
        val_size = int(len(feature_df) * 0.15)
        
        train_data = feature_df[:train_size]
        val_data = feature_df[train_size:train_size + val_size]
        test_data = feature_df[train_size + val_size:]
        
        # Prepare features and targets
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_val = val_data[feature_columns]
        y_val = val_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['advanced_scaler'] = scaler
        
        # Train ensemble models
        ensemble_models = self.train_ensemble_models(
            pd.DataFrame(X_train_scaled, columns=feature_columns),
            y_train,
            pd.DataFrame(X_val_scaled, columns=feature_columns),
            y_val
        )
        
        # Train stacking model
        stacking_model = self.train_stacking_model(
            ensemble_models,
            pd.DataFrame(X_train_scaled, columns=feature_columns),
            y_train,
            pd.DataFrame(X_val_scaled, columns=feature_columns),
            y_val
        )
        
        if stacking_model:
            ensemble_models['stacking_regressor'] = stacking_model
        
        # Train TCN-LSTM model
        if TORCH_AVAILABLE:
            X_train_seq, y_train_seq = self.prepare_sequence_data(
                train_data, feature_columns, target_column
            )
            X_val_seq, y_val_seq = self.prepare_sequence_data(
                val_data, feature_columns, target_column
            )
            
            tcn_lstm_model = self.train_tcn_lstm_model(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                input_size=len(feature_columns)
            )
            
            if tcn_lstm_model:
                ensemble_models['tcn_lstm'] = tcn_lstm_model
        
        # Evaluate models
        results = self.evaluate_advanced_models(
            ensemble_models,
            pd.DataFrame(X_test_scaled, columns=feature_columns),
            y_test
        )
        
        # Store models and results
        self.models = ensemble_models
        
        self.logger.info("All advanced models trained and evaluated")
        return ensemble_models
    
    def predict_with_advanced_models(self, 
                                   model_name: str,
                                   X: pd.DataFrame,
                                   forecast_hours: int = 72) -> np.ndarray:
        """Predict using advanced models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'tcn_lstm' and TORCH_AVAILABLE:
            # Handle TCN-LSTM predictions
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # Prepare sequence data
            if 'sequence_scaler' in self.scalers:
                X_scaled = self.scalers['sequence_scaler'].transform(X)
            else:
                X_scaled = X
            
            # Create sequences for prediction
            sequence_length = 24
            predictions = []
            
            with torch.no_grad():
                for i in range(len(X_scaled) - sequence_length):
                    sequence = torch.FloatTensor(X_scaled[i:i+sequence_length]).unsqueeze(0).to(device)
                    pred, _ = model(sequence)
                    predictions.append(pred.cpu().numpy()[0, 0])
            
            return np.array(predictions)
        
        else:
            # Handle other models
            if 'advanced_scaler' in self.scalers:
                X_scaled = self.scalers['advanced_scaler'].transform(X)
            else:
                X_scaled = X
            
            return model.predict(X_scaled)
    
    def get_best_advanced_model(self) -> Tuple[str, Any]:
        """Get the best performing advanced model."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Train models first.")
        
        best_model_name = min(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['mape'])
        best_model = self.models[best_model_name]
        
        self.logger.info(f"Best advanced model: {best_model_name} "
                        f"(MAPE: {self.evaluation_results[best_model_name]['mape']:.2f}%)")
        
        return best_model_name, best_model
    
    def generate_advanced_model_report(self) -> Dict[str, Any]:
        """Generate comprehensive report for advanced models."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        report = {
            'summary': {
                'total_models': len(self.evaluation_results),
                'best_model': self.get_best_advanced_model()[0],
                'best_mape': min(result['mape'] for result in self.evaluation_results.values()),
                'average_mape': np.mean([result['mape'] for result in self.evaluation_results.values()])
            },
            'model_performance': self.evaluation_results,
            'model_types': {
                'ensemble': [name for name in self.models.keys() if 'regressor' in name.lower()],
                'deep_learning': [name for name in self.models.keys() if 'tcn' in name.lower() or 'lstm' in name.lower()]
            },
            'recommendations': self._generate_advanced_recommendations()
        }
        
        return report
    
    def _generate_advanced_recommendations(self) -> List[str]:
        """Generate recommendations for advanced models."""
        recommendations = []
        
        best_mape = min(result['mape'] for result in self.evaluation_results.values())
        
        if best_mape < 5:
            recommendations.append("Exceptional performance! Advanced models achieving <5% MAPE.")
        elif best_mape < 8:
            recommendations.append("Excellent performance! Target MAPE <8% achieved with advanced models.")
        elif best_mape < 12:
            recommendations.append("Good performance. Consider hyperparameter tuning and feature selection.")
        else:
            recommendations.append("Performance needs improvement. Consider data quality and model architecture.")
        
        # Model-specific recommendations
        if 'tcn_lstm' in self.evaluation_results:
            tcn_mape = self.evaluation_results['tcn_lstm']['mape']
            if tcn_mape < best_mape * 1.1:
                recommendations.append("TCN-LSTM shows strong performance. Consider attention mechanism tuning.")
        
        if 'stacking_regressor' in self.evaluation_results:
            stack_mape = self.evaluation_results['stacking_regressor']['mape']
            if stack_mape < best_mape * 1.1:
                recommendations.append("Stacking ensemble effective. Consider adding more base models.")
        
        return recommendations
