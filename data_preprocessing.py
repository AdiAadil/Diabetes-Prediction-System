import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# 1️⃣ Dataset load
df = pd.read_csv("data/diabetes.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2️⃣ Missing values check
print("\nMissing values:\n", df.isnull().sum())

# 3️⃣ Basic statistics
print("\nDataset description:\n", df.describe())

# 4️⃣ Features & Target split
X = df.drop('Outcome', axis=1)  # features
y = df['Outcome']               # target
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)