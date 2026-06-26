import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataCollector:
    """Collects and generates e-commerce data"""
    
    def __init__(self, data_dir="data/raw"):
        """Initialize data collector"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Data collector initialized. Output: {self.data_dir}")
    
    def generate_synthetic_data(self, num_users=1000, num_products=500, num_interactions=5000):
        """
        Generate synthetic e-commerce data
        This is for testing. In real project, you'd scrape actual data.
        """
        print(f"\n📊 Generating synthetic data...")
        print(f"  - Users: {num_users}")
        print(f"  - Products: {num_products}")
        print(f"  - Interactions: {num_interactions}")
        
        # Generate products
        print("  ⏳ Creating products...")
        products_data = {
            'product_id': range(1, num_products + 1),
            'product_name': [f'Product_{i}' for i in range(1, num_products + 1)],
            'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports'], 
                                        num_products),
            'price': np.random.uniform(10, 500, num_products),
            'description': [f'High quality product {i}' for i in range(1, num_products + 1)]
        }
        df_products = pd.DataFrame(products_data)
        
        # Generate users
        print("  ⏳ Creating users...")
        users_data = {
            'user_id': range(1, num_users + 1),
            'username': [f'user_{i}' for i in range(1, num_users + 1)],
            'signup_date': pd.date_range('2023-01-01', periods=num_users, freq='h'),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'India', 'Germany'], 
                                       num_users)
        }
        df_users = pd.DataFrame(users_data)
        
        # Generate interactions (implicit feedback: views, purchases, ratings)
        print("  ⏳ Creating user-product interactions...")
        interactions_data = {
            'user_id': np.random.randint(1, num_users + 1, num_interactions),
            'product_id': np.random.randint(1, num_products + 1, num_interactions),
            'interaction_type': np.random.choice(['view', 'click', 'purchase', 'wishlist'], 
                                                num_interactions, p=[0.5, 0.2, 0.2, 0.1]),
            'timestamp': pd.date_range('2023-06-01', periods=num_interactions, freq='min'),
            'rating': np.random.choice([1, 2, 3, 4, 5], num_interactions, p=[0.05, 0.1, 0.2, 0.3, 0.35])
        }
        df_interactions = pd.DataFrame(interactions_data)
        
        # Remove duplicate user-product pairs (keep latest interaction)
        df_interactions = df_interactions.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'product_id'], keep='last'
        )
        
        return df_products, df_users, df_interactions
    
    def save_data(self, df_products, df_users, df_interactions):
        """Save data to CSV files"""
        print("\n💾 Saving data to files...")
        
        products_path = self.data_dir / 'products.csv'
        users_path = self.data_dir / 'users.csv'
        interactions_path = self.data_dir / 'interactions.csv'
        
        df_products.to_csv(products_path, index=False)
        print(f"  ✓ Saved {len(df_products)} products to {products_path}")
        
        df_users.to_csv(users_path, index=False)
        print(f"  ✓ Saved {len(df_users)} users to {users_path}")
        
        df_interactions.to_csv(interactions_path, index=False)
        print(f"  ✓ Saved {len(df_interactions)} interactions to {interactions_path}")
        
        return {
            'products_path': products_path,
            'users_path': users_path,
            'interactions_path': interactions_path
        }
    
    def load_data(self, data_dir=None):
        """Load existing data from CSV files"""
        if data_dir is None:
            data_dir = self.data_dir
        
        products_path = Path(data_dir) / 'products.csv'
        users_path = Path(data_dir) / 'users.csv'
        interactions_path = Path(data_dir) / 'interactions.csv'
        
        print("\n📂 Loading data from files...")
        
        if all([products_path.exists(), users_path.exists(), interactions_path.exists()]):
            df_products = pd.read_csv(products_path)
            df_users = pd.read_csv(users_path)
            df_interactions = pd.read_csv(interactions_path)
            
            print(f"  ✓ Loaded {len(df_products)} products")
            print(f"  ✓ Loaded {len(df_users)} users")
            print(f"  ✓ Loaded {len(df_interactions)} interactions")
            
            return df_products, df_users, df_interactions
        else:
            print("  ✗ Data files not found!")
            return None, None, None
    
    def print_data_summary(self, df_products, df_users, df_interactions):
        """Print summary statistics"""
        print("\n📈 Data Summary:")
        print(f"\n📦 PRODUCTS:")
        print(f"  - Total: {len(df_products)}")
        print(f"  - Categories: {df_products['category'].nunique()}")
        print(f"  - Price range: ${df_products['price'].min():.2f} - ${df_products['price'].max():.2f}")
        print(f"\n👥 USERS:")
        print(f"  - Total: {len(df_users)}")
        print(f"  - Countries: {df_users['country'].nunique()}")
        print(f"\n🔗 INTERACTIONS:")
        print(f"  - Total: {len(df_interactions)}")
        print(f"  - Sparsity: {1 - (len(df_interactions) / (len(df_products) * len(df_users))):.4f}")
        print(f"  - Types: {df_interactions['interaction_type'].unique()}")


def main():
    """Main execution"""
    print("=" * 60)
    print("SmartRecommendation: Data Collection")
    print("=" * 60)
    
    # Initialize collector
    collector = DataCollector(data_dir="data/raw")
    
    # Generate synthetic data
    df_products, df_users, df_interactions = collector.generate_synthetic_data(
        num_users=1000,
        num_products=500,
        num_interactions=5000
    )
    
    # Save to files
    paths = collector.save_data(df_products, df_users, df_interactions)
    
    # Print summary
    collector.print_data_summary(df_products, df_users, df_interactions)
    
    print("\n" + "=" * 60)
    print("✅ Day 1 Task Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
