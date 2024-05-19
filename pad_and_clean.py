import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
"""
# Load the .arff file
data = arff.loadarff('/Users/willem/Downloads/php3isjYz.arff')
df = pd.DataFrame(data[0])

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())
"""
def clean_and_scale(df):
    # Handle missing values
    df.dropna(inplace=True)

    # Convert categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8'))  # Decode byte strings
        if df[col].nunique() == 2:  # Binary categorical variable
            df[col] = pd.get_dummies(df[col], drop_first=True)
        else:  # Multi-class categorical variable
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('Class', axis=1))
    df_scaled = pd.DataFrame(scaled_features, columns=df.drop('Class', axis=1).columns)
    df_scaled['Class'] = df['Class'].values

    # Split data into features and target
    X = df_scaled.drop('Class', axis=1)
    y = df_scaled['Class']
    y = np.where(y == y[0], -1, 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def pad(X, y, desired_divisor):
    X = X.to_numpy()
    # Calculate rows to add
    current_rows = X.shape[0]
    rows_to_add = (desired_divisor - (current_rows % desired_divisor)) % desired_divisor

    # Calculate columns to add
    current_columns = X.shape[1]
    columns_to_add = (desired_divisor - (current_columns % desired_divisor)) % desired_divisor

    # Padding rows
    if rows_to_add > 0:
        # Create padding array with zeros for rows
        padding_rows = np.zeros((rows_to_add, X.shape[1]))
        X_padded = np.vstack([X, padding_rows])
        y_padded = np.concatenate([y, np.zeros(rows_to_add)])
    else:
        X_padded = X
        y_padded = y

    # Padding columns
    if columns_to_add > 0:
        # Create padding array with zeros for columns
        padding_columns = np.zeros((X_padded.shape[0], columns_to_add))
        X_padded = np.hstack([X_padded, padding_columns])

    return X_padded, y_padded
"""

data = arff.loadarff('/Users/willem/Downloads/php3isjYz.arff')
df = pd.DataFrame(data[0])
hill_train_x, hill_test_x, hill_train_y, hill_test_y = clean_and_scale(df)

print(hill_train_x.shape, hill_train_y.shape, hill_test_x.shape, hill_test_y.shape)
hill_train_x, hill_train_y = pad(hill_train_x, hill_train_y, 7)

hill_test_x, hill_test_y = pad(hill_test_x, hill_test_y, 7)
print(hill_train_x.shape, hill_train_y.shape, hill_test_x.shape, hill_test_y.shape)

"""
