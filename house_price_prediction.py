import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns = ['square_footage', 'bedrooms', 'bathrooms']
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic house price data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        square_footage = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(2, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        
        # Generate prices with some noise
        base_price = 50000
        price_per_sqft = 100
        price_per_bedroom = 15000
        price_per_bathroom = 10000
        
        noise = np.random.normal(0, 20000, n_samples)
        prices = (base_price + 
                 square_footage * price_per_sqft + 
                 bedrooms * price_per_bedroom + 
                 bathrooms * price_per_bathroom + 
                 noise)
        
        # Ensure positive prices
        prices = np.abs(prices)
        
        data = pd.DataFrame({
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'price': prices
        })
        
        return data
    
    def load_data(self, file_path=None):
        """Load data from CSV file or generate sample data"""
        if file_path and file_path.endswith('.csv'):
            try:
                data = pd.read_csv(file_path)
                print(f"Loaded data from {file_path}")
                return data
            except FileNotFoundError:
                print(f"File {file_path} not found. Using sample data instead.")
                return self.generate_sample_data()
        else:
            print("No file provided. Using sample data.")
            return self.generate_sample_data()
    
    def preprocess_data(self, data):
        """Preprocess the data for training"""
        # Handle missing values
        data = data.dropna()
        
        # Ensure all required columns exist
        for col in self.feature_columns + ['price']:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Separate features and target
        X = data[self.feature_columns]
        y = data['price']
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the linear regression model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate the model performance"""
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R²': train_r2,
            'Test R²': test_r2
        }
        
        return metrics, y_test_pred
    
    def visualize_results(self, y_test, y_test_pred, data):
        """Create visualizations for the results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('House Price Prediction Analysis', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Prices')
        axes[0, 0].set_ylabel('Predicted Prices')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        
        # 2. Residuals plot
        residuals = y_test - y_test_pred
        axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Prices')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Feature correlation matrix (text-based)
        corr_data = data[self.feature_columns + ['price']].copy()
        corr_matrix = corr_data.corr()
        axes[1, 0].text(0.5, 0.5, str(corr_matrix.round(2)), 
                        transform=axes[1, 0].transAxes, fontsize=10,
                        ha='center', va='center')
        axes[1, 0].set_title('Feature Correlation Matrix')
        axes[1, 0].axis('off')
        
        # 4. Distribution of prices
        axes[1, 1].hist(data['price'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Price')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of House Prices')
        
        plt.tight_layout()
        plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return feature_importance
    
    def predict_price(self, square_footage, bedrooms, bathrooms):
        """Predict house price for new data"""
        new_data = pd.DataFrame({
            'square_footage': [square_footage],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms]
        })
        
        new_data_scaled = self.scaler.transform(new_data)
        predicted_price = self.model.predict(new_data_scaled)[0]
        
        return predicted_price
    
    def run_complete_analysis(self, file_path=None):
        """Run the complete analysis pipeline"""
        print("=" * 50)
        print("HOUSE PRICE PREDICTION ANALYSIS")
        print("=" * 50)
        
        # Load data
        data = self.load_data(file_path)
        print(f"\nDataset shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Preprocess data
        X, y = self.preprocess_data(data)
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train model
        X_train, X_test, y_train, y_test = self.train_model(X, y)
        print("\nModel training completed!")
        
        # Evaluate model
        metrics, y_test_pred = self.evaluate_model(X_train, X_test, y_train, y_test)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        
        # Feature importance
        importance = self.get_feature_importance()
        print("\nFeature Importance:")
        print(importance)
        
        # Visualize results
        self.visualize_results(y_test, y_test_pred, data)
        
        return metrics

def main():
    """Main function to run the house price prediction"""
    predictor = HousePricePredictor()
    
    # Run complete analysis with sample data
    metrics = predictor.run_complete_analysis()
    
    # Example predictions
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)
    
    test_cases = [
        (1500, 3, 2),
        (2500, 4, 3),
        (1800, 2, 2),
        (3000, 5, 4)
    ]
    
    for sqft, beds, baths in test_cases:
        price = predictor.predict_price(sqft, beds, baths)
        print(f"House: {sqft} sqft, {beds} bed, {baths} bath → Predicted Price: ${price:,.2f}")

if __name__ == "__main__":
    main()
