from gpupriori import *
import pandas as pd

np.random.seed(42)
num_instances = 500
num_items = 20
# Random binary matrix (simulate one-hot encoded data)
dummy_data = np.random.randint(0, 2, size=(num_instances, num_items))
item_names = [f"Item_{i}" for i in range(num_items)]
data_df = pd.DataFrame(dummy_data, columns=item_names)

# Initialize the GPUpriori object
engine = GPUpriori(data=data_df, use_gpu=False)  # Set use_gpu=True if GPU is available and desired

# Compute intersections and Pearson correlation matrix
engine.compute_intersections()
engine.compute_pearson_correlation()

# Save matrices to CSV files
engine.save_to_csv()

# Get items correlated with a specific item (e.g., "Item_5")
correlated = engine.get_correlated_items("Item_5", threshold=0.2)
print(f"Items correlated with 'Item_5' above threshold 0.2:\n{correlated}")

# Get top 5 correlations for all items
top_correlations = engine.get_top_correlations(top_k=5)
print("\nTop correlations for each item:")
for key, value in top_correlations.items():
    print(f"{key}: {value}")