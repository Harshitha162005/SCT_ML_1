# House Price Prediction with Linear Regression

This project implements a linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms.

## Features

- **Linear Regression Model**: Uses scikit-learn's LinearRegression for price prediction
- **Data Preprocessing**: Handles missing values and scales features
- **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and R²
- **Visualization**: Multiple plots for analysis and insights
- **Sample Data**: Generates synthetic data for demonstration purposes

## Files Structure

```
├── house_price_prediction.py    # Main prediction script
├── sample_data_generator.py     # Generate sample CSV data
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── house_prices.csv          # Sample data (generated)
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample data (optional):
```bash
python sample_data_generator.py
```

## Usage

### Basic Usage
```python
from house_price_prediction import HousePricePredictor

predictor = HousePricePredictor()
metrics = predictor.run_complete_analysis('house_prices.csv')
```

### Predict New House Prices
```python
# Predict price for a 2000 sqft house with 3 bedrooms and 2 bathrooms
price = predictor.predict_price(2000, 3, 2)
print(f"Predicted price: ${price:,.2f}")
```

### Run Complete Analysis
```bash
python house_price_prediction.py
```

## Model Features

The model uses the following features to predict house prices:
- **square_footage**: Total living area in square feet
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms

## Model Performance

The model provides the following evaluation metrics:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## Visualization

The script generates several visualizations:
1. Actual vs Predicted prices scatter plot
2. Residuals plot
3. Feature correlation heatmap
4. Price distribution histogram

## Custom Data

To use your own data:
1. Create a CSV file with columns: `square_footage`, `bedrooms`, `bathrooms`, `price`
2. Run the analysis with your file path:
```python
predictor.run_complete_analysis('your_data.csv')
```

## Example Output

```
==================================================
HOUSE PRICE PREDICTION ANALYSIS
==================================================

Dataset shape: (1000, 4)

First 5 rows:
   square_footage  bedrooms  bathrooms   price
0            2055         3          2  275500
1            1744         2          1  224400
2            2170         3          2  287000
3            1956         3          2  265600
4            1996         3          2  269600

Model Performance Metrics:
Train RMSE: 19845.67
Test RMSE: 20123.45
Train MAE: 15890.23
Test MAE: 16123.78
Train R²: 0.85
Test R²: 0.83
```

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
