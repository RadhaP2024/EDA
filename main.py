import pandas as pd
import matplotlib.pyplot as plt

#LOAD DATASET
df = pd.read_csv("sales_data.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

#DISPLAY FIRST AND LAST ROWS
print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

#DATASET INFORMATION
print("\nDataset Information")
print(df.info())

#SHAPE OF DATASET
print("\nShape of Dataset")
print(df.shape)

print("\nNumber of Rows :", df.shape[0])
print("Number of Columns :", df.shape[1])

#COLUMN NAMES
print("\nColumn Names")
print(df.columns)

#DATA TYPES
print("\nData Types")
print(df.dtypes)

#MISSING VALUES
print("\nMissing Values")
print(df.isnull().sum())

#DUPLICATE RECORDS
print("\nDuplicate Records")
print(df.duplicated().sum())

#STATISTICAL SUMMARY
print("\nStatistical Summary")
print(df.describe())

#CORRELATION MATRIX
print("\nCorrelation Matrix")

numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr())

#HISTOGRAMS
colors = [ 
    "lightblue", "lightgreen", "red", "orange", "purple",
    "brown", "pink", "gray", "olive", "cyan"
]

numeric_columns = numeric_df.columns

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_columns):
    plt.subplot((len(numeric_columns) + 2) // 3, 3, i + 1)
    plt.hist(numeric_df[col].dropna(),
             bins=20,
             color=colors[i % len(colors)],
             edgecolor='black')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#BOX PLOTS
numeric_df.plot(kind='box',
                subplots=True,
                layout=(2,3),
                figsize=(12,8),
                sharex=False,
                sharey=False)

plt.tight_layout()
plt.show()

#CORRELARION HEATMAP
plt.figure(figsize=(8,6))

plt.imshow(numeric_df.corr(),
           cmap="coolwarm",
           interpolation='nearest')

plt.colorbar()

plt.xticks(range(len(numeric_df.columns)),
           numeric_df.columns,
           rotation=90)

plt.yticks(range(len(numeric_df.columns)),
           numeric_df.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

#VALUE COUNTS
print("\nCategorical Columns")

categorical = df.select_dtypes(include='object')

for col in categorical.columns:
    print("\n", col)
    print(df[col].value_counts())

#DATASET SUMMARY
print("\n" + "=" * 50)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nSummary")

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])
print("Missing Values :", df.isnull().sum().sum())
print("Duplicate Rows :", df.duplicated().sum())

print("\nEDA Finished.")