import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print('='*60)#visual of a separator line(==============================)
print('INTERMEDIATE: STATISTICS, GROUPBY & VISUALIZATION')
print('='*60)

# Load the cleaned data
df = pd.read_excel('/workspaces/s/coffee_sales_data_cleaned.xlsx')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# =========================
# 1. ADVANCED STATISTICS
# =========================
print('\n' + '='*60) #This line creates a separator using string repetition. 
# using a constant would make it easier to maintain consistent formatting and adjust the width if needed.
print('1. ADVANCED STATISTICS')


# Revenue statistics by coffee type
print("\n--- Revenue by Coffee Type ---")
coffee_stats = df.groupby('coffee_name')['money'].agg([
    ('transaction_count', 'count'),
    ('total_revenue', 'sum'),
    ('avg_price', 'mean'),
    ('std_price', 'std'),
    ('min_price', 'min'),
    ('max_price', 'max')
]).round(2).sort_values('total_revenue', ascending=False)
print(coffee_stats.head(10))

# Payment method statistics
print("\n--- Revenue by Payment Method ---")
total_len = len(df)
payment_stats = df.groupby('cash_type')['money'].agg([
    ('transactions', 'count'),
    ('total_revenue', 'sum'),
    ('avg_transaction', 'mean'),
    ('percentage', lambda x: (len(x) / total_len * 100))
]).round(2)
print(payment_stats)

# vectorized percentage calculation #this is an alternative to the lambda function above
# payment_stats['percentage'] = (payment_stats['transactions'] / total_len * 100).round(2)
# print(payment_stats)

# Daily statistics
print("\n--- Daily Revenue Statistics ---")
daily_stats = df.groupby('date')['money'].agg([
    ('transactions', 'count'),
    ('daily_revenue', 'sum'),
    ('avg_sale', 'mean')
]).round(2)
# print(daily_stats.describe())

# Day of week performance
print("\n--- Performance by Day of Week ---")
day_performance = df.groupby('day_of_week')['money'].agg([
    ('transactions', 'count'),
    ('total_revenue', 'sum'),
    ('avg_revenue', 'mean')
]).round(2)

# Reorder days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_performance = day_performance.reindex([d for d in day_order if d in day_performance.index])
print(day_performance)

# Hourly patterns
print("\n--- Hourly Sales Pattern ---")
hourly_stats = df.groupby('hour')['money'].agg([
    ('transactions', 'count'),
    ('revenue', 'sum')
]).round(2)
print(hourly_stats)

# =========================
# 2. ADVANCED GROUPBY ANALYSIS
# =========================
print('\n' + '='*60)
print('2. ADVANCED GROUPBY ANALYSIS')
print('='*60)

# Multi-level grouping: Day of week + Payment type
print("\n--- Revenue by Day & Payment Method ---")
day_payment = df.groupby(['day_of_week', 'cash_type'])['money'].agg([
    ('transactions', 'count'),
    ('revenue', 'sum')
]).round(2)
print(day_payment)

# Top products by payment method
print("\n--- Top 5 Products by Payment Method ---")
for payment in df['cash_type'].dropna().unique():
    print(f"\n{payment.upper()}:")
    top_products = df[df['cash_type'] == payment].groupby('coffee_name')['money'].agg([
        ('transactions', 'count'),
        ('revenue', 'sum')
    ]).sort_values('revenue', ascending=False).head(5)
    print(top_products)

# Price range analysis
print("\n--- Price Range Distribution ---")
df['price_range'] = pd.cut(df['money'], bins=[0, 20, 25, 30, np.inf], 
                            labels=['Budget (<$20)', 'Standard ($20-25)', 
                                   'Premium ($25-30)', 'Luxury ($30+)'],
                            include_lowest=True)
price_range_stats = df.groupby('price_range')['money'].agg([
    ('transactions', 'count'),
    ('percentage', lambda x: len(x) / len(df) * 100),
    ('avg_price', 'mean')
]).round(2)
print(price_range_stats)

# =========================
# 3. VISUALIZATIONS
# =========================
print('\n' + '='*60)
print('3. GENERATING VISUALIZATIONS')
print('='*60)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 12))

# Plot 1: Top 10 Products by Revenue
plt.subplot(3, 3, 1)
top10 = coffee_stats.head(10)
plt.barh(top10.index, top10['total_revenue'], color='skyblue')
plt.xlabel('Total Revenue ($)')
plt.title('Top 10 Coffee Products by Revenue')
plt.gca().invert_yaxis()

# Plot 2: Revenue by Payment Method
plt.subplot(3, 3, 2)
plt.pie(payment_stats['total_revenue'], labels=payment_stats.index, 
        autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
plt.title('Revenue Distribution by Payment Method')

# Plot 3: Daily Revenue Trend
plt.subplot(3, 3, 3)
daily_revenue = df.groupby('date')['money'].sum()
plt.plot(daily_revenue.index, daily_revenue.values, marker='o', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.title('Daily Revenue Trend')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 4: Day of Week Performance
plt.subplot(3, 3, 4)
plt.bar(day_performance.index, day_performance['total_revenue'], color='coral')
plt.xlabel('Day of Week')
plt.ylabel('Total Revenue ($)')
plt.title('Revenue by Day of Week')
plt.xticks(rotation=45)

# Plot 5: Hourly Sales Pattern
plt.subplot(3, 3, 5)
plt.plot(hourly_stats.index, hourly_stats['transactions'], marker='s', 
         linewidth=2, color='green')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.title('Sales Activity by Hour')
plt.grid(True, alpha=0.3)

# Plot 6: Price Distribution
plt.subplot(3, 3, 6)
plt.hist(df['money'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.axvline(df['money'].mean(), color='red', linestyle='--', label=f'Mean: ${df["money"].mean():.2f}')
plt.axvline(df['money'].median(), color='orange', linestyle='--', label=f'Median: ${df["money"].median():.2f}')
plt.legend()

# Plot 7: Box Plot by Payment Method
plt.subplot(3, 3, 7)
sns.boxplot(data=df, x='cash_type', y='money', palette='Set2')
plt.xlabel('Payment Method')
plt.ylabel('Price ($)')
plt.title('Price Distribution by Payment Method')

# Plot 8: Correlation Heatmap
plt.subplot(3, 3, 8)
# Use one-hot encoding for categorical variables for correlation analysis
df_encoded = pd.get_dummies(df, columns=['cash_type', 'coffee_name'], drop_first=True)
corr = df_encoded[['money', 'hour'] + [col for col in df_encoded.columns if col.startswith('cash_type_') or col.startswith('coffee_name_')]].corr()
# Exclude coffee_name from correlation analysis to avoid artificial ordinal relationships
corr = df_encoded[['money', 'hour', 'cash_type_encoded']].corr()

# Plot 9: Top Coffee + Payment Method
plt.subplot(3, 3, 9)
top5_coffee = df['coffee_name'].value_counts().head(5).index
df_top5 = df[df['coffee_name'].isin(top5_coffee)]
payment_by_coffee = df_top5.groupby(['coffee_name', 'cash_type']).size().unstack(fill_value=0)
payment_by_coffee.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], ax=plt.gca())
plt.xlabel('Coffee Type')
plt.ylabel('Number of Sales')
plt.title('Top 5 Products: Payment Method Split')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Payment')

plt.tight_layout()
plt.savefig('intermediate_analysis.png', dpi=300, bbox_inches='tight')
# print('\n✅ Visualization saved as: intermediate_analysis.png')
# plt.show()

# =========================
# 4. KEY INSIGHTS SUMMARY
# =========================
print('\n' + '='*60)
print('4. KEY INSIGHTS')
print('='*60)

print(f"\n📊 Total Revenue: ${df['money'].sum():.2f}")
print(f"📈 Average Transaction: ${df['money'].mean():.2f}")
print(f"🔢 Total Transactions: {len(df)}")
print(f"☕ Unique Products: {df['coffee_name'].nunique()}")
print(f"\n🏆 Best Selling Product: {df['coffee_name'].value_counts().index[0]} ({df['coffee_name'].value_counts().iloc[0]} sales)")
print(f"💰 Highest Revenue Product: {coffee_stats.index[0]} (${coffee_stats.iloc[0]['total_revenue']:.2f})")
print(f"📅 Best Day: {day_performance['total_revenue'].idxmax()} (${day_performance['total_revenue'].max():.2f})")
print(f"⏰ Peak Hour: {hourly_stats['transactions'].idxmax()}:00 ({hourly_stats['transactions'].max()} transactions)")
print(f"💳 Most Popular Payment: {payment_stats['transactions'].idxmax()} ({payment_stats.loc[payment_stats['transactions'].idxmax(), 'percentage']:.1f}%)")

print('\n' + '='*60)
print('Analysis Complete!')
print('='*60)

# saved file to excel
df.to_excel('coffee_sales_intermediate_analysis.xlsx', index=False)