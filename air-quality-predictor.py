import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import logging
import joblib
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityPredictor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'temperature', 'humidity', 'wind_speed', 'wind_direction']
        self.target = 'AQI'
        self.models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'NeuralNetwork': MLPRegressor(random_state=42)
        }
        self.best_model = None

    def load_and_preprocess_data(self) -> None:
        logger.info("Loading and preprocessing data...")
        self.data = pd.read_csv(self.data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime', inplace=True)
        self._handle_missing_values()
        self._engineer_features()

    def _handle_missing_values(self) -> None:
        logger.info("Handling missing values...")
        imputer = KNNImputer(n_neighbors=5)
        self.data[self.features] = imputer.fit_transform(self.data[self.features])

    def _engineer_features(self) -> None:
        logger.info("Engineering features...")
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['season'] = pd.cut(self.data.index.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate rolling averages
        for feature in self.features:
            self.data[f'{feature}_rolling_24h'] = self.data[feature].rolling(window=24).mean()
        
        # Interaction terms
        self.data['PM_interaction'] = self.data['PM2.5'] * self.data['PM10']
        self.data['temp_humidity_interaction'] = self.data['temperature'] * self.data['humidity']

        self.features.extend(['hour', 'day_of_week', 'month', 'is_weekend', 'PM_interaction', 'temp_humidity_interaction'])
        self.features.extend([f'{feature}_rolling_24h' for feature in self.features])

    def perform_pca(self) -> None:
        logger.info("Performing PCA...")
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        pca_result = pca.fit_transform(self.data[self.features])
        
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        self.data = pd.concat([self.data, pca_df], axis=1)
        self.features.extend(pca_df.columns)

    def train_and_evaluate_models(self) -> None:
        logger.info("Training and evaluating models...")
        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        for name, model in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            param_grid = {
                'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
                'GradientBoosting': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]},
                'NeuralNetwork': {'model__hidden_layer_sizes': [(100,), (100, 50)], 'model__alpha': [0.0001, 0.001]}
            }

            grid_search = GridSearchCV(pipeline, param_grid[name], cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2, 'Best Params': grid_search.best_params_}
            logger.info(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        self.best_model = min(results, key=lambda x: results[x]['MSE'])
        logger.info(f"Best model: {self.best_model}")
        
        # Save the best model
        joblib.dump(self.models[self.best_model], f'{self.best_model}_model.joblib')

    def visualize_results(self) -> None:
        logger.info("Visualizing results...")
        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model = joblib.load(f'{self.best_model}_model.joblib')
        y_pred = best_model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Air Quality Index')
        plt.ylabel('Predicted Air Quality Index')
        plt.title('Actual vs Predicted Air Quality Index')
        plt.tight_layout()
        plt.savefig('prediction_scatter_plot.png')
        plt.close()

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()

    def perform_time_series_analysis(self) -> None:
        logger.info("Performing time series analysis...")
        # Decompose the time series
        result = seasonal_decompose(self.data[self.target], model='additive', period=24*7)
        
        plt.figure(figsize=(12, 10))
        plt.subplot(411)
        plt.plot(result.observed)
        plt.title('Observed')
        plt.subplot(412)
        plt.plot(result.trend)
        plt.title('Trend')
        plt.subplot(413)
        plt.plot(result.seasonal)
        plt.title('Seasonal')
        plt.subplot(414)
        plt.plot(result.resid)
        plt.title('Residual')
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png')
        plt.close()

    def forecast_future(self, periods: int = 30) -> None:
        logger.info(f"Forecasting future for {periods} periods...")
        # Prepare data for Prophet
        prophet_data = self.data.reset_index()[['datetime', self.target]].rename(columns={'datetime': 'ds', self.target: 'y'})
        
        model = Prophet()
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        plt.figure(figsize=(12, 6))
        model.plot(forecast)
        plt.title(f'Air Quality Index Forecast for Next {periods} Days')
        plt.tight_layout()
        plt.savefig('aqi_forecast.png')
        plt.close()

    def run_analysis(self) -> None:
        self.load_and_preprocess_data()
        self.perform_pca()
        self.train_and_evaluate_models()
        self.visualize_results()
        self.perform_time_series_analysis()
        self.forecast_future()

if __name__ == "__main__":
    predictor = AirQualityPredictor('air_quality_data.csv')
    predictor.run_analysis()
