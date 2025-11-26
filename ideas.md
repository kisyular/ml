# Online Retail II - ML Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements & Context](#requirements--context)
3. [Dataset Information](#dataset-information)
4. [Business Problems](#business-problems)
5. [Technical Implementation](#technical-implementation)
6. [Visualization & Dashboard](#visualization--dashboard)
7. [Insights & Recommendations](#insights--recommendations)
8. [Future Work](#future-work)
9. [Code Templates](#code-templates)

---

## Project Overview

### Goals
Build an end-to-end machine learning project that predicts and explains **customer purchasing behavior and revenue drivers** using real transactional data.

**Dual Purpose:**
- **Academic:** Meet Dr. Chakraborty's Assignment 01 requirements
- **Professional:** Create a portfolio-worthy ML system following Chris's real-world guidance

### Key Deliverables
1. Jupyter Notebook with EDA + ML modeling
2. Streamlit dashboard for insights visualization
3. Word/PDF summary for assignment submission
4. GitHub repository with complete pipeline

---

## Requirements & Context

### Assignment 01 Requirements - DONE

**Dataset Criteria:**
- ≥ 800-1000 rows
- 8-12 useful columns
- Realistic business context

**Deliverables (90 points total):**
- **Dataset Selection (10 pts):** ≥ 800 rows & 8-12 useful columns, realistic business context
- **3 Business Problems (30 pts each):**
  - Clear, measurable question aligned with dataset (10 pts)
  - ≥ 3 Independent Variables with justification (10 pts)
  - Clear Dependent Variable capturing business goal (10 pts)
  - Preliminary exploration of data (EDA) (10 pts)

### Assignment 02 Requirements (Chris, TA)
Assignment 02: For the dataset you chose in assignment 01, identify two business problems, and apply two different methods or algorithms that are applicable.
These methods need to be consistent with the DV of choice and you can present some of the preliminary exploration of the data as well for justifying your choice of IV and DV. 
Provide code separately to the main report. Choose which method performs the best? Expand on your findings and provide your thoughts. 
Submit a word document for the report and any additional code in their native format (ipynb, .py or .r)

### Real-World Guidance (Chris, TA)

**Key Principles:**
1. **Portfolio-Worthy Work:** Build something employers want to see, not just classroom exercises
2. **End-to-End System:** Show full data flow: ingestion → transformation → modeling → presentation
3. **Tool Integration:** Use APIs, databases, cloud services, dashboards
4. **Career Alignment:** Choose datasets relevant to your target job roles
5. **Automation Focus:** Make it repeatable and scalable

**Philosophy:**
> "NEVER does data exist in a nice one-off CSV… the data is always coming FROM somewhere, getting transformed, and going TO somewhere else."

**Technologies to Explore:**
- APIs (real-time or batch)
- Databases (SQL, MongoDB, Azure Blob)
- Cloud services (Azure AI/ML tools)
- Dashboards (Streamlit, Plotly, Power BI)

---

## Dataset Information

### Source
**Online Retail II Dataset**
UCI Machine Learning Repository
🔗 [https://archive.ics.uci.edu/ml/datasets/Online+Retail+II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

### Description
- **Period:** December 2009 - December 2011
- **Records:** 1,067,371 transactions
- **Context:** UK-based online retail store (gift & home products)

### Data Fields

| Variable      | Type        | Description                        | Business Meaning                           |
|---------------|-------------|------------------------------------|--------------------------------------------|
| InvoiceNo     | Categorical | Transaction number                 | Unique sale transaction ID                 |
| StockCode     | Categorical | Product code                       | Unique identifier for each product         |
| Description   | Text        | Product name                       | Useful for product-level insights          |
| Quantity      | Numeric     | Units purchased                    | Measures demand volume                     |
| InvoiceDate   | Datetime    | Date and time of transaction       | Used for trend, seasonality, and recency   |
| UnitPrice     | Numeric     | Price per unit (in GBP)            | Indicates pricing and discounting          |
| CustomerID    | Categorical | Unique customer identifier         | Enables customer segmentation & prediction |
| Country       | Categorical | Country of customer                | Allows geographic market comparison        |

### Data Quality Issues
- ~25% missing CustomerID (guest orders)
- ~0.3% missing Description
- Negative Quantity/Price values (returns/cancellations)
- InvoiceNo starting with "C" = cancellations

---

## Business Problems

### Problem 1: Customer Churn Prediction

**Business Question:**
Which customers are at risk of not returning in the next 90 days?

**Business Context:**
Marketing wants to identify at-risk customers to prioritize retention campaigns and prevent churn.

**Hypothesis:**
Customers with declining recency, low frequency, and specific purchase patterns are more likely to churn.

| Component             | Details                                                                      |
|-----------------------|------------------------------------------------------------------------------|
| **Dependent Variable**| `will_return_90days` (1 if customer returns in next 90 days, 0 otherwise)    |
| **Independent Variables** | Recency, Frequency, Monetary (RFM), AvgBasketValue, DaysSinceFirstPurchase, Returns |
| **Model Type**        | Classification (Logistic Regression, Random Forest, **Gradient Boosting**)   |
| **Business Value**    | Identifies at-risk customers for targeted retention strategies               |
| **Performance Achieved** | **Recall: 0.688**, **F1 Score: 0.671** (Gradient Boosting with Class Weights)|

---

### Problem 2: Product Recommendation System

**Business Question:**
Which products are frequently bought together?

**Business Context:**
Increase average order value (AOV) by recommending complementary products at checkout (cross-selling).

**Hypothesis:**
Certain products (e.g., teacups and saucers) exhibit strong co-occurrence patterns in transactions.

| Component             | Details                                                                      |
|-----------------------|------------------------------------------------------------------------------|
| **Method**            | **Association Rule Mining (Apriori Algorithm)**                              |
| **Metrics**           | Support, Confidence, Lift                                                    |
| **Key Findings**      | Identified strong bundles (Lift > 20) for "Regency" tea sets and "Alarm Clocks" |
| **Business Value**    | Enables data-driven cross-selling and product bundling strategies            |
| **Implementation**    | `mlxtend` library for frequent itemsets and association rules                |

---

### Problem 3: Anomaly Detection in Transactions

**Business Question:**
Which transactions are fraudulent or suspicious?

**Business Context:**
Identify fraudulent or suspicious transactions to reduce losses and improve security.

**Hypothesis:**
Anomalous transactions deviate significantly from normal patterns in terms of value, quantity, or product combinations.

| Component             | Details                                                                      |
|-----------------------|------------------------------------------------------------------------------|
| **Method**            | **Unsupervised Anomaly Detection**                                           |
| **Techniques**        | Statistical Outliers (Z-score), Isolation Forest                             |
| **Key Metrics**       | Anomaly Score, Reconstruction Error                                          |
| **Business Value**    | Reduce fraud-related losses, improve transaction security, flag suspicious accounts |
| **Implementation**    | `scikit-learn` (Isolation Forest), `scipy.stats` (Z-score)                   |

---

## Technical Implementation

### Phase 1: Data Acquisition

**Important:** The Excel file contains two sheets that need to be combined:
- Sheet 1: "Year 2009-2010"
- Sheet 2: "Year 2010-2011"

```python
import pandas as pd

# Load dataset from both sheets
file_path = "data/raw/online_retail_II.xlsx"

# Read both sheets
df_2009_2010 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
df_2010_2011 = pd.read_excel(file_path, sheet_name='Year 2010-2011')

# Combine both sheets
df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

# Basic inspection
print("Dataset Shape:", df.shape)
print(f"Total Transactions: {df.shape[0]:,}")
df.head()
```

### Phase 2: Data Cleaning

**Objectives:**
- Remove nulls, invalid quantities/prices
- Remove cancellations (InvoiceNo starting with "C")
- Compute Revenue = Quantity × UnitPrice
- Parse InvoiceDate and extract time features

**Cleaning Steps:**
```python
# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna(subset=['CustomerID', 'Description'])

# Remove cancellations and invalid values
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Convert data types
df['CustomerID'] = df['CustomerID'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Feature engineering
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour
```

**Expected Result:** Clean dataset with ~400,000-450,000 rows

### Phase 3: Exploratory Data Analysis (EDA)

**Using seaborn for all visualizations**

#### 3.1 Revenue Distribution
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
sns.histplot(df['revenue'], bins=50, kde=True, color='skyblue')
plt.title('Revenue Distribution')
plt.xlabel('Revenue (£)')
plt.ylabel('Frequency')
plt.show()
```
**Insight:** Revenue is right-skewed—most orders are small, few large invoices dominate.

#### 3.2 Monthly Revenue Trend
```python
monthly_rev = df.groupby('month', as_index=False)['revenue'].sum()

plt.figure(figsize=(18, 6))
sns.barplot(data=monthly_rev, x='month', y='revenue', color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Total Revenue')
plt.xlabel('Month')
plt.ylabel('Total Revenue (£)')
plt.show()
```
**Insight:** Strong seasonality—revenues peak in November-December (holiday period).

#### 3.3 Top 10 Countries by Revenue
```python
country_rev = (
    df.groupby('country', as_index=False)['revenue']
      .sum()
      .sort_values('revenue', ascending=False)
      .head(10)
)

plt.figure(figsize=(18, 6))
sns.barplot(data=country_rev, x='revenue', y='country', color='seagreen', alpha=0.8)
plt.title('Top 10 Countries by Total Revenue')
plt.xlabel('Revenue (£)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
```
**Insight:** UK contributes ~85% of revenue; Germany and France are key secondary markets.

#### 3.4 Customer Purchase Frequency
```python
plt.figure(figsize=(16, 6))
sns.histplot(customer_summary['num_transactions'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Transactions per Customer')
plt.xlabel('Number of Transactions')
plt.ylabel('Customer Count')
plt.show()
```
**Insight:** ~60% of customers purchase only once—clear signal for retention strategy.

#### 3.5 Recency-Frequency Heatmap (RFM)
```python
customer_summary['recency_group'] = pd.cut(
    customer_summary['recency_days'],
    bins=[0,30,60,90,120,9999],
    labels=['<30','31-60','61-90','91-120','120+']
)
customer_summary['freq_group'] = pd.cut(
    customer_summary['num_transactions'],
    bins=[1,2,3,5,10,100],
    labels=['1','2','3-5','6-10','10+']
)

rfm = customer_summary.groupby(
    ['recency_group','freq_group'], as_index=False
).agg(mean_revenue=('total_revenue','mean'))

rfm_pivot = rfm.pivot(index='recency_group', columns='freq_group', values='mean_revenue')

plt.figure(figsize=(12, 8))
sns.heatmap(rfm_pivot, annot=True, fmt='.0f', cmap='YlGn', cbar_kws={'label': 'Avg Revenue (£)'})
plt.title('Customer Behavior Heatmap: Recency vs Frequency')
plt.xlabel('Frequency Group')
plt.ylabel('Recency Group')
plt.show()
```
**Insight:** Frequent, recent customers are most profitable—strong ML predictors.

#### 3.6 Pareto Curve (80/20 Rule)
```python
pareto = customer_summary.sort_values('total_revenue', ascending=False).reset_index(drop=True)
pareto['cum_revenue'] = pareto['total_revenue'].cumsum()
pareto['revenue_share'] = pareto['cum_revenue'] / pareto['total_revenue'].sum()
pareto['customer_share'] = (pareto.index + 1) / len(pareto)

plt.figure(figsize=(12, 8))
plt.plot(pareto['customer_share'], pareto['revenue_share'], color='steelblue', linewidth=2)
plt.axhline(y=0.8, color='red', linestyle='--', label='80% Revenue')
plt.axvline(x=0.2, color='red', linestyle='--', label='20% Customers')
plt.title('Pareto Curve: Customer Revenue Concentration')
plt.xlabel('Cumulative Share of Customers')
plt.ylabel('Cumulative Share of Revenue')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```
**Insight:** ~20% of customers generate ~80% of revenue.

#### 3.7 Hourly Revenue Pattern
```python
hourly_rev = df.groupby('hour', as_index=False)['revenue'].sum()

plt.figure(figsize=(16, 6))
sns.lineplot(data=hourly_rev, x='hour', y='revenue', color='darkred', marker='o', linewidth=2)
plt.title('Average Hourly Revenue Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Total Revenue (£)')
plt.xticks(range(0, 24, 2))
plt.grid(alpha=0.3)
plt.show()
```
**Insight:** Peak sales 10 AM–3 PM (UK time)—matches office shopping hours.

### Phase 4: Feature Engineering

#### 4.1 Customer-Level Features (RFM)
```python
from datetime import datetime

# Customer aggregation
customer_summary = (
    df.groupby('CustomerID')
      .agg(
          TotalRevenue=('Revenue','sum'),
          TotalQuantity=('Quantity','sum'),
          NumTransactions=('InvoiceNo','nunique'),
          Country=('Country','first'),
          LastPurchase=('InvoiceDate','max')
      )
      .reset_index()
)

# Recency calculation
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
customer_summary['RecencyDays'] = (analysis_date - customer_summary['LastPurchase']).dt.days
customer_summary['AvgBasketValue'] = customer_summary['TotalRevenue'] / customer_summary['NumTransactions']
customer_summary.drop(columns='LastPurchase', inplace=True)

# Target variable for Problem 2
customer_summary['IsRepeatCustomer'] = (customer_summary['NumTransactions'] > 1).astype(int)
```

#### 4.2 Product-Level Features
```python
product_summary = (
    df.groupby('StockCode', as_index=False)
      .agg(
          ProductProfitability=('Revenue','mean'),
          AvgUnitPrice=('UnitPrice','mean'),
          TotalQuantitySold=('Quantity','sum'),
          UniqueCustomers=('CustomerID','nunique'),
          AvgBasketSize=('Quantity','mean')
      )
)
```

#### 4.3 Transaction-Level Features
```python
# Customer type (new vs returning)
customer_purchase_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
df['CustomerType'] = df['CustomerID'].map(
    lambda x: 1 if customer_purchase_counts[x] > 1 else 0
)
```

### Phase 5: Machine Learning Modeling

#### 5.1 Model 1 - Customer Churn Prediction (Classification)

```python
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Gradient Boosting with SMOTE
gb_model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=class_weight_ratio,
        random_state=42
    ))
])

gb_model.fit(X_train, y_train)
# Achieved Recall: 0.688, F1: 0.671
```

#### 5.2 Model 2 - Product Recommendation (Association Rules)

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Create basket matrix (InvoiceNo x StockCode)
basket = (df.groupby(['InvoiceNo', 'StockCode'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Encode (1 if purchased, 0 otherwise)
basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

# Find frequent itemsets
frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print(rules[['antecedents', 'consequents', 'lift', 'confidence']].head())
```

#### 5.3 Model 3 - Anomaly Detection (Unsupervised)

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Select features for Isolation Forest
iso_features = ['total_value', 'total_items', 'unique_products', 'avg_item_price']
X_iso = transaction_features[iso_features].copy()

# Standardize features
scaler = StandardScaler()
X_iso_scaled = scaler.fit_transform(X_iso)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
transaction_features['anomaly_iso_forest'] = iso_forest.fit_predict(X_iso_scaled)

# Anomaly if -1
n_anomalies = (transaction_features['anomaly_iso_forest'] == -1).sum()
print(f"Anomalies detected: {n_anomalies}")
```

---

## Visualization & Dashboard

### Dashboard Components (Streamlit)

#### KPIs Section
```python
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Retail ML Dashboard", layout="wide")
st.title("📊 Online Retail Analytics Dashboard")

# Calculate KPIs
total_revenue = df['Revenue'].sum()
avg_order_value = df.groupby('InvoiceNo')['Revenue'].sum().mean()
repeat_rate = (customer_summary['IsRepeatCustomer'].mean() * 100)

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue (£)", f"{total_revenue:,.0f}")
col2.metric("Avg Order Value (£)", f"{avg_order_value:,.2f}")
col3.metric("Repeat Rate (%)", f"{repeat_rate:.1f}")
```

#### Revenue Trend Visualization
```python
st.subheader("Monthly Revenue Trend")
monthly_rev_fig = px.line(
    monthly_rev, x='Month', y='Revenue',
    title="Monthly Total Revenue",
    labels={'Revenue': 'Revenue (£)'}
)
st.plotly_chart(monthly_rev_fig, use_container_width=True)
```

#### Model Performance Visualization (Churn)
```python
st.subheader("Customer Retention Probability Distribution")
retention_df = pd.DataFrame({
    'Probability': gb_model.predict_proba(X_test)[:, 1],
    'Actual': y_test
})

fig_ret = px.histogram(
    retention_df, x='Probability', color='Actual',
    title='Distribution of Retention Probabilities',
    labels={'Probability': 'Predicted Repurchase Probability'},
    barmode='overlay', opacity=0.7
)
st.plotly_chart(fig_ret, use_container_width=True)
```

#### Product Recommendations
```python
st.subheader("Top Product Associations")
st.dataframe(rules[['antecedents', 'consequents', 'lift', 'confidence']].head(10))
```

#### Anomaly Detection
```python
st.subheader("Detected Anomalies")
anomalies_df = transaction_features[transaction_features['anomaly_iso_forest'] == -1]
st.write(f"Total Anomalies: {len(anomalies_df)}")
st.dataframe(anomalies_df[['Invoice', 'total_value', 'total_items', 'anomaly_score']].head(10))
```

#### Geographic Revenue Map
```python
st.subheader("Geographic Revenue Distribution")
country_summary = df.groupby('Country', as_index=False)['Revenue'].sum()

fig_geo = px.choropleth(
    country_summary,
    locations='Country', locationmode='country names',
    color='Revenue', title='Revenue by Country',
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig_geo, use_container_width=True)
```

### Deployment Options
1. **Streamlit Cloud:** Free hosting for public dashboards
2. **Azure App Service:** Enterprise deployment with authentication
3. **Docker Container:** Portable deployment across platforms

---

## Insights & Recommendations

### Summary of Model Performance

| Model/Method              | Metric    | Value      | Key Takeaway                      |
|---------------------------|-----------|------------|-----------------------------------|
| Churn Prediction          | Recall    | **0.688**  | Recency and frequency are key     |
| Product Recommendation    | Lift      | **> 20**   | Strong bundles exist              |
| Anomaly Detection         | Anomalies | **~1-5%**  | Fraud/Suspicious activity detected|

### Business Insights by Area

#### 1. Sales and Revenue Drivers
**Findings:**
- Quantity sold is the strongest revenue predictor
- Strong seasonality: November-December revenue spikes
- UK dominates (~85%), but EU markets show steady growth

**Recommendations:**
- Forecast staffing and inventory for Q4 spikes
- Optimize volume discounts during high-demand months
- Localize promotions for Germany and France

#### 2. Customer Behavior and Retention
**Findings:**
- Recency is the most predictive feature for repeat behavior
- ~60% of customers are one-time buyers (high churn)
- Customers active within last 30 days have >70% repurchase probability

**Recommendations:**
- Automated reactivation campaigns for RecencyDays > 60
- Frequent buyer loyalty tier for high AvgBasketValue customers
- Churn-monitoring dashboard for declining activity alerts

#### 3. Anomaly Detection and Fraud Prevention
**Findings:**
- ~1-5% of transactions flagged as anomalous
- Statistical outliers often indicate bulk purchases or potential errors
- Isolation Forest effective at finding multi-dimensional anomalies

**Recommendations:**
- Implement real-time alerts for high anomaly scores
- Manual review for transactions > 3 standard deviations from mean
- Dynamic transaction limits based on customer history

### Cross-Domain Observations

| Theme            | Finding                      | Business Impact               |
|------------------|------------------------------|-------------------------------|
| Seasonality      | Holiday demand dominates     | Planning & forecasting        |
| Customer Loyalty | Few customers drive revenue  | Loyalty programs critical     |
| Product Portfolio| Small core of profitable SKUs| Focus marketing efficiently   |
| Time Behavior    | Peak hours 10 AM–3 PM        | Schedule promotions           |
| Market Mix       | UK-heavy but EU potential    | Growth via localization       |

---

## Future Work

### Technical Enhancements
1. **Automate Data Ingestion:** Replace manual loading with API/ETL pipeline (Prefect, Airflow)
2. **Database Storage:** Store processed data in Azure SQL or MongoDB
3. **Model Retraining:** Implement monthly automated retraining
4. **Advanced Models:** Try XGBoost, LightGBM for improved accuracy
5. **Time-Series Forecasting:** Add Prophet or LSTM for revenue prediction
6. **Customer Segmentation:** K-means clustering for unsupervised grouping

### Dashboard Enhancements
1. Interactive filters (date, country, product category)
2. Anomaly detection alerts
3. Downloadable reports (PDF, Excel)
4. Real-time data refresh
5. A/B testing integration for marketing campaigns

### Deployment
1. Full cloud deployment (Azure or AWS)
2. Model API endpoints for predictions
3. Automated alert system (email/Teams)
4. User authentication and role-based access
5. Performance monitoring and logging

---

## Code Templates

### Complete Notebook Structure

```
1. Introduction
   - Project overview
   - Goals and deliverables

2. Dataset Overview
   - Load and inspect data
   - Data dictionary
   - Summary statistics

3. Data Cleaning
   - Remove nulls and invalid records
   - Feature engineering
   - Data type conversions

4. Exploratory Data Analysis
   - Revenue distribution
   - Temporal trends
   - Geographic analysis
   - Customer behavior
   - Product analysis

5. Business Problems Definition
   - Problem 1: Revenue drivers
   - Problem 2: Customer repurchase
   - Problem 3: Product profitability

6. Feature Engineering
   - Customer-level features (RFM)
   - Product-level features
   - Transaction-level features

7. Machine Learning Modeling
   - Model 1: Revenue prediction
   - Model 2: Customer retention
   - Model 3: Product profitability
   - Feature importance & SHAP

8. Insights & Recommendations
   - Model performance summary
   - Business insights
   - Actionable recommendations

9. Conclusion
   - Project summary
   - Key learnings
   - Future work
```

### Project Directory Structure

```
retail-ml-project/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── customer_summary.csv
│   │   └── product_summary.csv
│   └── models/
│       ├── revenue_model.pkl
│       ├── retention_model.pkl
│       └── profitability_model.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
│
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── config.py
│
├── images/
│   └── (EDA plots and model visualizations)
│
├── reports/
│   ├── assignment_01_submission.pdf
│   └── technical_report.pdf
│
├── README.md
└── requirements.txt
```

### Requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
seaborn>=0.12.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
streamlit>=1.20.0
plotly>=5.13.0
shap>=0.41.0
openpyxl>=3.0.0
pyjanitor>=0.24.0
```

---

## Final Checklist

### Academic Requirements
- [ ] Dataset meets criteria (≥800 rows, 8-12 columns)
- [ ] 3 business problems clearly defined
- [ ] Dependent variables identified and justified
- [ ] ≥3 independent variables per problem with rationale
- [ ] Comprehensive EDA with visualizations
- [ ] Word/PDF report prepared for submission

### Portfolio Requirements
- [ ] Complete Jupyter notebook with all sections
- [ ] Streamlit dashboard functional
- [ ] GitHub repository organized
- [ ] README with project overview and setup instructions
- [ ] Code is clean, commented, and reproducible
- [ ] Screenshots/demo video prepared

### Professional Quality
- [ ] End-to-end data flow demonstrated
- [ ] ML models evaluated with proper metrics
- [ ] Business insights clearly communicated
- [ ] Deployment plan documented
- [ ] Future work section included

---

**Last Updated:** 2025-11-06
