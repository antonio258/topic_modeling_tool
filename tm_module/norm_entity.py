import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

def find_null(data):
    """Replaces all NaN values in the input data with 0.

    Args:
        data (numpy.ndarray): The input data array.

    Returns:
        numpy.ndarray: The modified data array with NaN values replaced by 0.
    """
    data[np.isnan(data)] = 0
    return data

def norm_col(data):
    """Normalizes the input data column-wise.

    Args:
        data (pandas.Series): The input data column.

    Returns:
        numpy.ndarray: The normalized data array.
    """
    l = data.values
    l = [arr.tolist() for arr in l]
    l = np.array(l)
    x_normed = l / l.sum(axis=1, keepdims=1)
    x_normed = [element * 100 for element in x_normed]
    return x_normed

def normalize(data):
    """Normalizes the input data.

    Args:
        data (numpy.ndarray): The input data array.

    Returns:
        numpy.ndarray: The normalized data array.
    """
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data / sum(data)
    data = 100 * data
    data = np.round(data, 2)
    return data

def sum_array(data):
    """Calculates the sum of elements in the input array.

    Args:
        data (numpy.ndarray): The input data array.

    Returns:
        float: The sum of elements in the array.
    """
    return sum(data)

def norm_entity(df_topics, data, id_column, entities, save_path):
    """Normalizes the topics and entities in the input data.

    Args:
        df_topics (pandas.DataFrame): The DataFrame containing topics data.
        data (pandas.DataFrame): The DataFrame containing entity data.
        id_column (str): The name of the column containing the unique identifier.
        entities (list): The list of entity columns to normalize.
        save_path (str): The path to save the normalized data.

    Returns:
        None
    """
    cols = list(df_topics.columns)
    cols.remove('dominant_topic')
    cols.remove(id_column)
    cols_array = df_topics[cols].values.tolist()
    df_topics['array'] = cols_array
    df_topics.rename(columns={'dominant_topic': 'dmnt'}, inplace=True)
    df = df_topics[[id_column, 'array', 'dmnt']].copy().reset_index(drop=True)
    data = data.reset_index(drop=True)

    df['topics'] = df['array'].apply(normalize)
    df['sum'] = df['topics'].apply(sum_array)
    df[df['sum'] > 0].to_parquet(f"{save_path}/normalized_topics.parquet")

    df = pd.merge(data, df, left_index=True, right_index=True)

    for entity in entities:
        df_filter = df[df['topics'].apply(max) >= 10]
        df_entity = df_filter.groupby(entity).agg({'topics':sum})
        df_entity.rename(columns={'topics': 'ranks'}, inplace=True)

        df_entity['ranks'] = df_entity['ranks'].apply(find_null)

        df_entity['ranks_norm'] = norm_col(df_entity.ranks)
        df_entity['sum'] = df_entity['ranks_norm'].apply(sum_array)

        df_entity.reset_index(inplace=True)
        df_entity[df_entity['sum'] > 0].to_parquet(f"{save_path}/normalized_topics_{entity}.parquet")
