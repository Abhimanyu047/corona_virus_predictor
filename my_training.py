import pandas as pd
import numpy as np

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data)) # Produces random shuffled numbers
    test_set_size = int(len(data) * ratio) #Multiply by ratio to make the divide
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]