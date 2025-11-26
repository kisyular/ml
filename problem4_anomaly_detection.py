"""
Business Problem 4: Anomaly Detection in Transactions

PROBLEM: Identify fraudulent or suspicious transactions
APPROACH: Unsupervised anomaly detection using statistical methods + ML
TECHNIQUES: Isolation Forest, Statistical outliers, Z-score

ANOMALY TYPES TO DETECT:
1. Unusually high order values
2. Abnormal quantities purchased
3. Suspicious customer behavior patterns
4. Unusual product combinations

BUSINESS VALUE:
- Reduce fraud-related losses
- Improve transaction security
- Flag suspicious accounts for review
- Protect customer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("BUSINESS PROBLEM 4: ANOMALY DETECTION IN TRANSACTIONS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n1. Loading data...")

df = pd.read_csv('data/cleaned_retail_data.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"  Total transactions: {len(df):,}")
print(f"  Unique invoices: {df['Invoice'].nunique():,}")
print(f"  Unique customers: {df['Customer ID'].nunique():,}")

# ============================================================================
# FEATURE ENGINEERING FOR ANOMALY DETECTION
# ============================================================================
print("\n2. Engineering features for anomaly detection...")

# TRANSACTION-LEVEL FEATURES
transaction_features = df.groupby('Invoice').agg(
    total_value=('Revenue', 'sum'),
    total_items=('Quantity', 'sum'),
    unique_products=('StockCode', 'nunique'),
    avg_item_price=('Price', 'mean'),
    max_item_price=('Price', 'max'),
    min_item_price=('Price', 'min'),
    std_item_price=('Price', 'std'),
    customer_id=('Customer ID', 'first'),
    country=('Country', 'first'),
    date=('InvoiceDate', 'first')
).reset_index()

# CUSTOMER-LEVEL AGGREGATES (for context)
customer_stats = df.groupby('Customer ID').agg(
    total_orders=('Invoice', 'nunique'),
    avg_order_value=('Revenue', 'mean'),
    total_spent=('Revenue', 'sum'),
    avg_items_per_order=('Quantity', 'mean')
).reset_index()

# Merge customer context
transaction_features = transaction_features.merge(
    customer_stats,
    left_on='customer_id',
    right_on='Customer ID',
    how='left'
)

# DERIVED FEATURES
transaction_features['value_to_avg_ratio'] = (
    transaction_features['total_value'] / transaction_features['avg_order_value']
)
transaction_features['items_to_avg_ratio'] = (
    transaction_features['total_items'] / transaction_features['avg_items_per_order']
)
transaction_features['price_range'] = (
    transaction_features['max_item_price'] - transaction_features['min_item_price']
)

# Handle infinities and NaNs
transaction_features = transaction_features.replace([np.inf, -np.inf], np.nan)
transaction_features = transaction_features.fillna(0)

print(f"  Transaction features created: {len(transaction_features):,}")
print(f"  Number of features: {len(transaction_features.columns)}")

# ============================================================================
# METHOD 1: STATISTICAL OUTLIER DETECTION (Z-SCORE)
# ============================================================================
print("\n3. Method 1: Statistical outlier detection (Z-score)...")

# Calculate z-scores for key metrics
from scipy import stats

transaction_features['z_total_value'] = np.abs(stats.zscore(transaction_features['total_value']))
transaction_features['z_total_items'] = np.abs(stats.zscore(transaction_features['total_items']))
transaction_features['z_unique_products'] = np.abs(stats.zscore(transaction_features['unique_products']))

# Flag as anomaly if z-score > 3 (3 standard deviations)
transaction_features['anomaly_statistical'] = (
    (transaction_features['z_total_value'] > 3) |
    (transaction_features['z_total_items'] > 3) |
    (transaction_features['z_unique_products'] > 3)
).astype(int)

n_statistical_anomalies = transaction_features['anomaly_statistical'].sum()
print(f"  Statistical anomalies detected: {n_statistical_anomalies:,} ({n_statistical_anomalies / len(transaction_features) * 100:.2f}%)")

# ============================================================================
# METHOD 2: ISOLATION FOREST
# ============================================================================
print("\n4. Method 2: Isolation Forest (ML-based)...")

# Select features for Isolation Forest
iso_features = [
    'total_value', 'total_items', 'unique_products',
    'avg_item_price', 'max_item_price', 'price_range',
    'value_to_avg_ratio', 'items_to_avg_ratio'
]

X_iso = transaction_features[iso_features].copy()

# Standardize features
scaler = StandardScaler()
X_iso_scaled = scaler.fit_transform(X_iso)

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_estimators=100
)

# Predict anomalies (-1 = anomaly, 1 = normal)
transaction_features['anomaly_iso_forest'] = iso_forest.fit_predict(X_iso_scaled)
transaction_features['anomaly_iso_forest'] = (transaction_features['anomaly_iso_forest'] == -1).astype(int)

# Anomaly score (lower = more anomalous)
transaction_features['anomaly_score'] = iso_forest.score_samples(X_iso_scaled)

n_iso_anomalies = transaction_features['anomaly_iso_forest'].sum()
print(f"  Isolation Forest anomalies detected: {n_iso_anomalies:,} ({n_iso_anomalies / len(transaction_features) * 100:.2f}%)")

# ============================================================================
# COMBINED ANOMALY FLAG
# ============================================================================
print("\n5. Creating combined anomaly flag...")

# Anomaly if EITHER method flags it
transaction_features['anomaly_combined'] = (
    (transaction_features['anomaly_statistical'] == 1) |
    (transaction_features['anomaly_iso_forest'] == 1)
).astype(int)

n_combined_anomalies = transaction_features['anomaly_combined'].sum()
print(f"  Combined anomalies: {n_combined_anomalies:,} ({n_combined_anomalies / len(transaction_features) * 100:.2f}%)")

# ============================================================================
# ANALYZE ANOMALIES
# ============================================================================
print("\n6. Analyzing detected anomalies")
print("=" * 80)

anomalies = transaction_features[transaction_features['anomaly_combined'] == 1]
normal = transaction_features[transaction_features['anomaly_combined'] == 0]

print("\nComparison: Anomalies vs Normal Transactions")
print("-" * 80)
print(f"{'Metric':<25} {'Normal (Mean)':<20} {'Anomaly (Mean)':<20} {'Ratio':<10}")
print("-" * 80)

metrics_to_compare = [
    ('total_value', 'Total Value (£)'),
    ('total_items', 'Total Items'),
    ('unique_products', 'Unique Products'),
    ('avg_item_price', 'Avg Item Price (£)')
]

for col, label in metrics_to_compare:
    normal_mean = normal[col].mean()
    anomaly_mean = anomalies[col].mean()
    ratio = anomaly_mean / normal_mean if normal_mean > 0 else 0
    print(f"{label:<25} £{normal_mean:<19.2f} £{anomaly_mean:<19.2f} {ratio:<10.1f}x")

# ============================================================================
# TOP ANOMALIES
# ============================================================================
print("\n7. Top 10 Most Suspicious Transactions")
print("=" * 80)

top_anomalies = anomalies.nsmallest(10, 'anomaly_score')[
    ['Invoice', 'total_value', 'total_items', 'unique_products', 'customer_id', 'country']
]

for idx, row in top_anomalies.iterrows():
    print(f"\nInvoice: {row['Invoice']}")
    print(f"  Total Value: £{row['total_value']:.2f}")
    print(f"  Items: {row['total_items']:.0f}")
    print(f"  Products: {row['unique_products']:.0f}")
    print(f"  Customer: {row['customer_id']}")
    print(f"  Country: {row['country']}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n8. Creating visualizations...")

# 1. Total Value Distribution (Normal vs Anomaly)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(normal['total_value'], bins=50, alpha=0.7, color='blue', label='Normal', edgecolor='black')
plt.hist(anomalies['total_value'], bins=50, alpha=0.7, color='red', label='Anomaly', edgecolor='black')
plt.xlabel('Total Transaction Value (£)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Transaction Value Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(normal['total_value']), bins=50, alpha=0.7, color='blue', label='Normal', edgecolor='black')
plt.hist(np.log1p(anomalies['total_value']), bins=50, alpha=0.7, color='red', label='Anomaly', edgecolor='black')
plt.xlabel('Log(Total Transaction Value)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Log-Transformed Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/anomaly_value_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter: Total Value vs Total Items (colored by anomaly)
plt.figure(figsize=(10, 6))
plt.scatter(normal['total_items'], normal['total_value'],
           alpha=0.5, s=20, color='blue', label='Normal')
plt.scatter(anomalies['total_items'], anomalies['total_value'],
           alpha=0.8, s=50, color='red', label='Anomaly', edgecolor='black')
plt.xlabel('Total Items', fontsize=12)
plt.ylabel('Total Value (£)', fontsize=12)
plt.title('Transaction Anomalies: Value vs Items', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/anomaly_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Box plots comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

box_data = [
    ('total_value', 'Total Value (£)', axes[0, 0]),
    ('total_items', 'Total Items', axes[0, 1]),
    ('unique_products', 'Unique Products', axes[1, 0]),
    ('avg_item_price', 'Avg Item Price (£)', axes[1, 1])
]

for col, title, ax in box_data:
    data_to_plot = [normal[col], anomalies[col]]
    bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/anomaly_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iso_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[transaction_features['anomaly_combined'] == 0, 0],
           X_pca[transaction_features['anomaly_combined'] == 0, 1],
           alpha=0.5, s=20, color='blue', label='Normal')
plt.scatter(X_pca[transaction_features['anomaly_combined'] == 1, 0],
           X_pca[transaction_features['anomaly_combined'] == 1, 1],
           alpha=0.8, s=50, color='red', label='Anomaly', edgecolor='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('Anomaly Detection - PCA Projection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/anomaly_pca.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Visualizations saved to outputs/")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save anomalies
anomalies_export = transaction_features[transaction_features['anomaly_combined'] == 1][
    ['Invoice', 'customer_id', 'country', 'date', 'total_value', 'total_items',
     'unique_products', 'anomaly_statistical', 'anomaly_iso_forest', 'anomaly_score']
].sort_values('anomaly_score')

anomalies_export.to_csv('outputs/detected_anomalies.csv', index=False)

# Save all transactions with anomaly flags
transaction_features[
    ['Invoice', 'customer_id', 'total_value', 'total_items',
     'anomaly_statistical', 'anomaly_iso_forest', 'anomaly_combined', 'anomaly_score']
].to_csv('outputs/all_transactions_with_flags.csv', index=False)

print("\nResults saved to:")
print("  - outputs/detected_anomalies.csv")
print("  - outputs/all_transactions_with_flags.csv")

print("\n" + "=" * 80)
print("ANOMALY DETECTION COMPLETE")
print("=" * 80)

# ============================================================================
# ACTIONABLE RECOMMENDATIONS
# ============================================================================
print("\n🚨 ACTIONABLE BUSINESS RECOMMENDATIONS")
print("=" * 80)

print(f"\n1. FRAUD PREVENTION:")
print(f"   - {n_combined_anomalies:,} suspicious transactions detected")
print(f"   - Review high-value anomalies manually")
print(f"   - Implement real-time fraud detection using these patterns")

print(f"\n2. CUSTOMER VERIFICATION:")
suspicious_customers = anomalies['customer_id'].nunique()
print(f"   - {suspicious_customers:,} unique customers flagged")
print(f"   - Require additional verification for anomalous orders")
print(f"   - Monitor repeat offenders")

print(f"\n3. TRANSACTION LIMITS:")
print(f"   - Set dynamic limits based on customer history")
print(f"   - Flag transactions >3x customer's average")
print(f"   - Implement velocity checks (multiple orders in short time)")

print(f"\n4. AUTOMATED ALERTS:")
print(f"   - Send alerts for Isolation Forest anomalies")
print(f"   - Escalate transactions with anomaly_score < -0.5")
print(f"   - Daily fraud summary reports")

print("\n" + "=" * 80)
