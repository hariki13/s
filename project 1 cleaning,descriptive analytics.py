# --- Step 1: Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load Your Data ---
df = pd.read_csv('coffee sales dataset.csv')
# print("---Initial Data Overview---")

# step 3.Preview the data first 5 rows
# print("1. 5 Rows of the dataset:")
# print(df.head())


# step 4. get a concise summary the dataframe, including data types and non-null values
# print("\n2. Data Types and Non-Null Values:")
df.info()

# step 5. Data Cleaning ---
# print("\n3.---Starting Data Cleaning:")
# convert 'datetime' column from object to datetime type for time-series analysis
# The errors = 'coerce' will turn any values that can't be converted into Nat (Not a Time)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# check for missing values after conversion
# print("\n1. Missing Values in each columns after conversion:")
# print(df.isnull().sum())
      
# remove rows where 'datetime' or 'money' is missing, as they are critical for analysis
df.dropna(subset=['datetime', 'money','coffee_name'], inplace=True)

# remove duplicate rows to ensure data accuracy
initial_rows = len(df)
df.drop_duplicates(inplace=True)
# print(f"\n2. removed {initial_rows - len(df)} duplicate rows.")

# step 6. ----descriptive analytics ----
# print("\n---performing descriptive analytics---")
# calculate basic statistics for the 'money' column
total_sales = df['money'].sum()
average_sales = df['money'].mean()
max_sales = df['money'].max()
min_sales = df['money'].min()

# print(f"\n1. Total Sales: ${total_sales:.2f}")
# print(f"2. Average Sales: ${average_sales:.2f}")
# print(f"3. Maximum Sales: ${max_sales:.2f}")
# print(f"4. Minimum Sales: ${min_sales:.2f}")

# step 7. analyze product sales
# group by coffee name and calculate total sales and number of item sold for each ()
product_analysis = df.groupby('coffee_name')['money'].agg(['sum', 'count']).reset_index()
product_analysis.rename(columns={'sum': 'total_sales', 'count': 'items_sold'}, inplace=True)

#find the top 5 products by Sales
top_5_products_by_sales = product_analysis.sort_values(by='total_sales', ascending=False).head(5)
# print("\n4. Top 5 Products by Sales:")
# print(top_5_products_by_sales)

# find the top 5 products by items sold
top_5_products_by_items_sold = product_analysis.sort_values(by='items_sold', ascending=False).head(5)
# print("\n5. Top 5 Products by Items Sold:")
# print(top_5_products_by_items_sold)

# --- step 8: visualizations----
# print("\n---performing visualizations---")
#set the style for the plots
sns.set_style('whitegrid')

# a.Bar chart for top 5 products by sales
plt.figure(figsize=(10, 6))
sns.barplot(x='total_sales', y='coffee_name', data=top_5_products_by_sales,color='red')
plt.title('Top 5 Products by Sales', fontsize=16)
plt.xlabel('Total Sales($)', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.xlabel('Product', fontsize=12)

# plt.show()

# save the plot as an image file
# plt.savefig('top_5_products_by_sales.png')
# print("\n1. saved 'top_5_products_by_sales.png'")

# b.Bar chart for top 5 products by items sold
plt.figure(figsize=(10, 6))
sns.barplot(x='items_sold', y='coffee_name', data=top_5_products_by_items_sold,color='green')
plt.title('Top 5 Products by Items Sold', fontsize=16)
plt.xlabel('Items Sold', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# save the plot as an image file
# plt.savefig('top_5_products_by_items_sold.png')
# print("\n2. saved 'top_5_products_by_items_sold.png'")

# c. Daily sales Trend
# Resample data by day and sum the 'money' to get daily sales
daily_sales = df.set_index('datetime').resample('D')['money'].sum()

plt.figure(figsize=(14, 7))
daily_sales.plot(kind='line', marker='o', color='blue',linestyle='--',linewidth=2)
plt.title('daily Sales Trend', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
# plt.show()
df.to_csv("project1 cleaning,descriptive analytics.csv", index=False)
# save the plot as an image file
plt.savefig('daily_sales_trend.png')
print("\n3. saved 'daily_sales_trend.png'")

print("\n---analysis complete---")
## to see the plots, you can either open the saved.png files
## or add 'plt.show() after each plot section.



















