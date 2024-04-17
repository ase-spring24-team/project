import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataframe(data_set):
    data_file = f'../data/{data_set}/{data_set}.csv'

    #createing a dataframe
    df = pd.read_csv(data_file)
    print(df.columns.to_numpy())
    count = 0
    for column in df.columns.to_numpy():
        if column.endswith('-') or column.endswith('+'):
            break
        count += 1

    #splitting the dataframe into train and test
    X = df.iloc[:, :count]  
    Y = df.iloc[:, count:]  
    return train_test_split(X, Y, test_size=0.2, random_state=100)

def wrtie_to_csv(data_set, algorithm_name, column_names, data):

    ## five by five
    #randomly selecting 10000 rows from the generated data
    selected_data = random.sample(data, min(10000, len(data)))
    #dividing it into 5
    final_data = np.array_split(selected_data, 5)

    # Create the directory (handles non-existent parent directories)
    directory = os.path.dirname(f'../data/{data_set}/{algorithm_name}/')
    os.makedirs(directory, exist_ok=True)

    for i, data in enumerate(final_data):
        file_path = f'../data/{data_set}/{algorithm_name}/{algorithm_name}_hyperparameters_{i + 1}.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            for row in data:
                writer.writerow(row)