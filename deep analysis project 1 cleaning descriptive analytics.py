import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("coffee sales dataset.csv")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("coffee sales dataset.csv")

# ===== DATA CLEANING =====
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# ===== DESCRIPTIVE ANALYTICS =====
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())
print("\nSkewness:\n", df.skew())
print("\nKurtosis:\n", df.kurtosis())

# ===== ADVANCED ANALYSIS =====

# 1. Correlation Analysis
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix - Coffee Sales')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# 2. Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, col in enumerate(numeric_df.columns[:4]):
    ax = axes[idx // 2, idx % 2]
    ax.hist(numeric_df[col], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions.png', dpi=300)
plt.show()

# 3. Outlier Detection (IQR Method)
def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

print("\n=== OUTLIERS (IQR Method) ===")
for col in numeric_df.columns:
    outlier_count = detect_outliers(numeric_df[col])
    print(f"{col}: {outlier_count} outliers")

# 4. Statistical Testing (if categorical columns exist)
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\n=== CATEGORICAL ANALYSIS ===")
    for col in categorical_cols:
        print(f"\n{col}:\n", df[col].value_counts())

# 5. Advanced Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
numeric_df.boxplot(ax=axes[0])
axes[0].set_title('Box Plot - Numerical Features')

# Pair plot for key relationships
if len(numeric_df.columns) >= 2:
    axes[1].scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], alpha=0.6)
    axes[1].set_xlabel(numeric_df.columns[0])
    axes[1].set_ylabel(numeric_df.columns[1])
    axes[1].set_title('Scatter Plot - Key Relationship')

plt.tight_layout()
plt.savefig('advanced_analysis.png', dpi=300)
plt.show()

print("\n✓ Analysis Complete!")


