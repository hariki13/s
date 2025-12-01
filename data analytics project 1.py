import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("coffee sales dataset.csv")

# ===== DATA CLEANING =====
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
df = df.drop_duplicates()
df = df.dropna()

# ===== DESCRIPTIVE ANALYTICS =====
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())

# ===== TIME-SERIES ANALYSIS =====
print("\n=== TIME-SERIES ANALYSIS ===")
if 'Date' in df.columns or any('date' in col.lower() for col in df.columns):
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        
        # Time-based aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:2]:
            daily_avg = df_sorted.groupby(date_col)[col].mean()
            
            plt.figure(figsize=(12, 5))
            plt.plot(daily_avg.index, daily_avg.values, linewidth=2, color='steelblue')
            plt.title(f'Time-Series: {col}')
            plt.xlabel('Date')
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'timeseries_{col}.png', dpi=300)
            plt.show()

# ===== CLUSTERING ANALYSIS =====
print("\n=== CLUSTERING ANALYSIS ===")
numeric_df = df.select_dtypes(include=[np.number])

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Elbow method to find optimal clusters
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid()
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300)
plt.show()

# Apply K-Means with optimal k
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print(f"Cluster Distribution:\n{df['Cluster'].value_counts()}")

# Visualize clusters
if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], 
                         c=df['Cluster'], cmap='viridis', s=100, alpha=0.6)
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.title('K-Means Clustering')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=300)
    plt.show()

# ===== PREDICTIVE MODELING =====
print("\n=== PREDICTIVE MODELING ===")
if numeric_df.shape[1] > 1:
    # Use first numeric column as target
    X = numeric_df.iloc[:, 1:]
    y = numeric_df.iloc[:, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    # Model Performance
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance:\n{feature_importance}")
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

# ===== CORRELATION & HEATMAP =====
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

print("\n✓ Deep Advanced Analysis Complete!")