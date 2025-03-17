import pandas as pd
import numpy as np
import torch


class GPUpriori:
    """
    A GPU-accelerated, Apriori-like engine for computing intersections and Pearson correlations
    from a binary (or one-hot encoded) dataset.
    
    Attributes:
        data (pd.DataFrame or np.ndarray): Input dataset. Assumes rows are instances and columns are items.
        item_names (list): List of item names.
        use_gpu (bool): Whether to use GPU acceleration (if available).
        data_tensor (torch.Tensor): Data converted to a tensor.
        intersections (np.ndarray): Intersection matrix.
        pearson_corr (pd.DataFrame): Pearson correlation matrix.
    """

    def __init__(self, data, item_names=None, use_gpu=False):
        """
        Initialize the GPUpriori object.

        Args:
            data (pd.DataFrame or np.ndarray): Input dataset.
            item_names (list, optional): List of item names. If not provided and data is a DataFrame,
                                         the DataFrame's column names will be used.
            use_gpu (bool): If True, the computations are moved to the GPU (if available).
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.item_names = item_names or list(self.data.columns)
        else:
            self.data = data
            if item_names is None:
                raise ValueError("item_names must be provided when data is not a DataFrame")
            self.item_names = item_names

        self.use_gpu = use_gpu and torch.cuda.is_available()
        # Convert data to a tensor
        if isinstance(self.data, pd.DataFrame):
            data_values = self.data.values
        else:
            data_values = self.data
        self.data_tensor = torch.tensor(data_values, dtype=torch.float)
        if self.use_gpu:
            self.data_tensor = self.data_tensor.cuda()

        self.intersections = None
        self.pearson_corr = None

    def compute_intersections(self):
        """
        Compute the intersections matrix, which is essentially the co-occurrence count
        between each pair of items (columns) based on the one-hot/binary encoded data.
        
        Returns:
            np.ndarray: A matrix with shape (num_items, num_items) containing intersection counts.
        """
        # Compute intersections: (data.T dot data) gives co-occurrence counts.
        intersections = torch.matmul(self.data_tensor.t(), self.data_tensor)
        if self.use_gpu:
            intersections = intersections.cpu()
        self.intersections = intersections.numpy()
        return self.intersections

    def compute_pearson_correlation(self):
        """
        Compute the Pearson correlation matrix between each pair of items.
        
        Returns:
            pd.DataFrame: A DataFrame of Pearson correlation coefficients indexed by item names.
        """
        # Using pandas for a straightforward implementation
        if isinstance(self.data, pd.DataFrame):
            self.pearson_corr = self.data.corr(method='pearson')
        else:
            # Convert to DataFrame if not already one
            df = pd.DataFrame(self.data, columns=self.item_names)
            self.pearson_corr = df.corr(method='pearson')
        return self.pearson_corr

    def save_to_csv(self, intersections_file='intersections.csv', pearson_file='pearson_corr.csv'):
        """
        Save the intersections and Pearson correlation matrices to CSV files.

        Args:
            intersections_file (str): Filename for saving the intersections matrix.
            pearson_file (str): Filename for saving the Pearson correlation matrix.
        """
        if self.intersections is not None:
            df_inter = pd.DataFrame(self.intersections, index=self.item_names, columns=self.item_names)
            df_inter.to_csv(intersections_file, index_label='Item')
        else:
            print("Intersections not computed yet. Call compute_intersections() first.")

        if self.pearson_corr is not None:
            self.pearson_corr.to_csv(pearson_file, index_label='Item')
        else:
            print("Pearson correlation matrix not computed yet. Call compute_pearson_correlation() first.")

    def get_correlated_items(self, item, threshold=0.5):
        """
        Retrieve a dictionary of items with Pearson correlation greater than the given threshold
        for the specified item.

        Args:
            item (str): The item to query.
            threshold (float): Minimum correlation value to consider.

        Returns:
            dict: A dictionary where keys are item names and values are correlation scores.
        """
        if self.pearson_corr is None:
            raise ValueError("Pearson correlation matrix not computed. Call compute_pearson_correlation() first.")

        if item not in self.pearson_corr.index:
            raise ValueError(f"Item '{item}' not found in the correlation matrix.")

        series = self.pearson_corr.loc[item].drop(item)
        correlated = series[series > threshold]
        return correlated.sort_values(ascending=False).to_dict()

    def get_top_correlations(self, top_k=10):
        """
        For each item, get the top K correlated items.

        Args:
            top_k (int): Number of top correlations to return for each item.

        Returns:
            dict: A dictionary with item names as keys and dictionaries of top correlated items as values.
        """
        if self.pearson_corr is None:
            raise ValueError("Pearson correlation matrix not computed. Call compute_pearson_correlation() first.")

        top_corr = {}
        for item in self.pearson_corr.index:
            series = self.pearson_corr.loc[item].drop(item)
            top_corr[item] = series.sort_values(ascending=False).head(top_k).to_dict()
        return top_corr
