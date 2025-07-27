import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch


def normalize_column(df, column_list, col_min, col_max, condition):
    """
    Normalizes a column of a dataframe to the range [-1, 1].

    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): The name of the column to normalize.
        
    Returns:
        pd.DataFrame: DataFrame with normalized column in the range [-1, 1].
    """
    for column_name in column_list:
        if condition is True:
            # Normalize the column to the range [0, 1]
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
        elif condition is False:
            # Normalize the column to the range [0, 1]
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
        
            # Scale to the new range [-1, 1]
            df[column_name] = df[column_name] * 2 - 1
        
    return df


def denormalize(normalized_array, original_min, original_max):
    """
    Denormalizes a multidimensional array (list or numpy array) from the range [-1, 1] or [0, 1]
    back to their original range [original_min, original_max].

    Parameters:
        normalized_array (list or np.ndarray): Multidimensional array of normalized values.
        original_min (float): The original minimum value before normalization.
        original_max (float): The original maximum value before normalization.
        
    Returns:
        np.ndarray: A multidimensional array of denormalized values.
    """
    # Convert to a numpy array (if not already) for easy element-wise operations
    normalized_array = np.array(normalized_array)
    
    # Apply the denormalization formula element-wise
    denormalized_array = ((normalized_array + 1) / 2) * (original_max - original_min) + original_min
    
    return denormalized_array


def dataset_dynamic_model(df, input_cols, output_cols, nsteps, batch_size):
    """
    Prepare dataset and DataLoader from DataFrame.

    :param df: DataFrame containing the input and output columns
    :param input_cols: List of column names for input features
    :param output_cols: List of column names for output targets
    :param nsteps: Number of time steps for each sequence
    :param batch_size: Batch size for DataLoader
    :return: DataLoader instance with data in tensor format
    """
    # Extract input and output data from DataFrame
    inputs = torch.tensor(df[input_cols].values, dtype=torch.float32)
    outputs = torch.tensor(df[output_cols].values, dtype=torch.float32)

    # Calculate the number of sequences
    num_sequences = len(df) // nsteps

    # Truncate data to fit exact number of sequences
    inputs = inputs[:num_sequences * nsteps]
    outputs = outputs[:num_sequences * nsteps]

    # Reshape data into (batch, number of simulation steps, nx)
    inputs = inputs.view(num_sequences, nsteps, -1)
    outputs = outputs.view(num_sequences, nsteps, -1)

    # Create dataset and DataLoader
    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader