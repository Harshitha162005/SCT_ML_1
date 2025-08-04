import pandas as pd
import numpy as np

def generate_sample_csv(filename='house_prices.csv', n_samples=1000):
    """Generate a sample CSV file with house price data"""
    np.random.seed(42)
    
    # Generate features
    square_footage = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(2, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    
    # Generate prices with realistic relationships
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
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_footage': square_footage.astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': prices.astype(int)
    })
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 10 rows:")
    print(data.head(10))
    
    return data

if __name__ == "__main__":
    generate_sample_csv()
