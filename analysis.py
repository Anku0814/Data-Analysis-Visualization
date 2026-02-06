import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

num_col = df.select_dtypes(include='number').columns[0]
print(f"\nAverage of {num_col}: ", df[num_col].mean())

df[num_col].head(10).plot(kind='bar', title="Bar Chart")
plt.show()

plt.scatter(range(len(df[num_col])), df[num_col])
plt.title("Scatter Plot")
plt.xlabel("Index")
plt.ylabel(num_col)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
