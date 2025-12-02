import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print('BEGINNERS DATA EXPLORATION REPORT')
# Load the data
# Replace 'data.csv' with your actual data file path
df = pd.read_csv('coffee sales dataset 2.csv')

# Display basic information about the dataset

coffee_sales_basic_info = df.info()
print(coffee_sales_basic_info)
# print("\n" + "=" * 50)
print("FIRST 50 ROWS")
# print("=" * 50)
print(df.head(50))
# dataset shape
print(f"\nDataset shape: {df.shape}") #to know number of rows and columns
print(f"Number of rows: {df.shape[0]}") # to know number of rows
print(f"Number of columns: {df.shape[1]}") # to know number of columns

#  Display first few rows & column count
print('row_count:', len(df))
print('column_count:', len(df.columns))

# Display summary statistics
# print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
# print("=" * 50)
print(df.describe())
"""ouput and explanation of describe()
The describe() function provides a summary of statistics for numerical columns in the DataFrame.
It includes measures such as count, mean, standard deviation (std), minimum (min),
25th percentile (25%), median (50%), 75th percentile (75%), and maximum (max) values.
This summary helps to understand the distribution and central tendencies of the numerical data."""
"""e.g:
- count: Number of non-missing values in each numerical column.
- mean: Average value of each numerical column.
- std: Standard deviation, indicating the spread of values around the mean.
- min/max: Minimum and maximum values in each numerical column.
- percentiles (25%, 50%, 75%): Values below which a certain percentage of data falls.
- max: Maximum value in each numerical column."""

#----Mssing values(DATA)---
print("MISSING VALUES")
coffee_revenue_missing_values = df.isnull().sum()
missing_percentage = (coffee_revenue_missing_values / len(df)) * 100
# print(coffee_revenue_missing_values[coffee_revenue_missing_values > 0])
print("\nMISSING VALUES PERCENTAGE")
# print(missing_percentage[missing_percentage > 0])
missing_df = pd.DataFrame({coffee_revenue_missing_values.index.name: coffee_revenue_missing_values.index,
                           'Missing Count': coffee_revenue_missing_values.values,
                           'Percentage': missing_percentage.values})
print(missing_df[missing_df['Missing Count'] > 0])

# Check for duplicate rows
coffee_data_duplicates = df.duplicated().sum()
print(f"\nDUPLICATE ROWS: {coffee_data_duplicates}")
# delete duplicate rows if any
coffee_data_duplicates = df.drop_duplicates()
# print(f"\nSHAPE AFTER REMOVING DUPLICATES: {coffee_data_duplicates.shape}")

# ---TYPE CONVERSION & CATEGORICAL DATA EXPLORATION---

# Display coffee_sales_data types
print("DATA TYPES")
coffee_sales_data_types = df.dtypes
print(coffee_sales_data_types)

# Display unique values for categorical columns
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")

coffee_sales_categorical_columns = df.select_dtypes(include=['object']).columns
for col in coffee_sales_categorical_columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    if df[col].nunique() <= 10:
        print(df[col].value_counts())

print('EXPLANATION OUTPUT OF UNIQUE VALUES')
"""The output displays the number of unique values in each categorical column of the DataFrame.
For columns with 10 or fewer unique values, it also shows the frequency of each unique value.
This information helps to understand the diversity of categories present in the dataset
and can guide decisions on data preprocessing, such as encoding strategies for machine learning models."""
"""E.G:
date: 41 unique values → You have data spanning 41 distinct calendar days. This suggests your dataset covers about a month and a half of café operations.
datetime: 254 unique values → There are 254 distinct timestamps (likely transaction-level records). This is more granular than date — it captures the exact time of each sale.
cash_type: 2 unique values → Only two payment methods are recorded (e.g., "cash" vs "card" or "cash" vs "digital"). This is a categorical variable with low cardinality, useful for grouping and comparison.
coffee_name: 30 unique values → You sell 30 different coffee products (e.g., Espresso, Latte, Cappuccino, etc.). This is a high-cardinality categorical variable, perfect for product-level analysis."""

#---Convert data types if necessary----
# Example: Convert 'date' column to datetime if it's not already
if df['date'].dtype == 'object':
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # print("\nConverted 'date' column to datetime.")
# Example: Convert 'cash_type' column to category if it's not already
if df['cash_type'].dtype == 'object':
    df['cash_type'] = df['cash_type'].astype('category')
    # print("Converted 'cash_type' column to category.")
# verify cash_type has consistent labels eg: 'cash', 'Cash', 'CASH'
df['cash_type'] = df['cash_type'].str.lower().str.strip()
# print("Standardized 'cash_type' labels to lowercase and stripped whitespace.")
# look for typos or inconsistent entries in 'coffee_name' column
df['coffee_name'] = df['coffee_name'].str.lower().str.strip()
# print("Standardized 'coffee_name' labels to lowercase and stripped whitespace.")


# ----Visualizations OF DATA EXPLORATION---
print('GENERATING VISUALIZATIONS')

# SET style for better-looking plots
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
# Plot missing values heatmap
if df.isnull().sum().sum() > 0:
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
# Plot correlation matrix for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    plt.subplot(2, 2, 2)
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
# Plot distribution of numerical columns
if len(numerical_cols) > 0:
    plt.subplot(2, 2, 3)
    df[numerical_cols].hist(bins=20, figsize=(12, 8))
    plt.suptitle('Distribution of Numerical Columns')
plt.tight_layout()
plt.savefig('data_exploration_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'data_exploration_results.png'")
plt.show()
# save file after cleaning
df.to_excel('coffee_sales_data_cleaned.xlsx', index=False)
print("Cleaned data saved as 'coffee_sales_data_cleaned.xlsx'")