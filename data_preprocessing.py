
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example dataset
data = {
    'feature1': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
    'feature2': [2, 3, 4, np.nan, 6, 7, 8, 9, 10, 11],
    'feature3': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 1. Handle Missing Values
# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# 2. Handle Outliers
# For simplicity, using Z-score to identify outliers
from scipy import stats

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).all(axis=1)
df = df[~outliers]

# 3. Normalize or Scale Features
# Using StandardScaler for standard normalization (mean=0, std=1)
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('target')  # Exclude target column from scaling
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 4. Split the Data into Training and Testing Sets
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Output the results
print("Training features:\n", X_train.head())
print("Training labels:\n", y_train.head())
print("Testing features:\n", X_test.head())
print("Testing labels:\n", y_test.head())
