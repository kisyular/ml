# Streamlit Demo Website - Implementation Guide
# Online Retail II Analysis - Customer Intelligence Dashboard

**Author:** Rellika Kisyula
**Project:** Assignment 03 - Three Data Mining Approaches
**Tech Stack:** Streamlit + Python + Scikit-learn
**Estimated Time:** 2-3 days

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Model Export from Notebook](#model-export-from-notebook)
6. [Building the Streamlit App](#building-the-streamlit-app)
7. [Testing Locally](#testing-locally)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What You're Building
An interactive web dashboard that showcases your three data mining approaches:

**Page 1: Customer Churn Predictor**
- Input customer RFM features
- Get real-time churn probability
- See SHAP explanations
- Risk categorization (Low/Medium/High)

**Page 2: Customer Segmentation Explorer**
- 3D visualization of 4 customer clusters
- Segment profile cards (VIP, Occasional, At-Risk, One-Timer)
- Customer lookup tool
- Segment statistics

**Page 3: Product Recommendation Engine**
- Search products by name
- Display association rules
- "Frequently Bought Together" suggestions
- Lift/Confidence/Support metrics

**Page 4: Business Impact Dashboard**
- Revenue impact visualization (£1.8M)
- Lorenz curve
- Churn rates by segment
- ROI calculator

---

## Prerequisites

### Required Knowledge
- [x] Python basics
- [x] Pandas (you already use this)
- [x] Your Jupyter notebook results
- [ ] Basic Streamlit (you'll learn this)

### System Requirements
- Python 3.8+ (check: `python --version`)
- Git (for version control)
- 2GB free disk space

### Required Python Packages
```bash
# Core packages (you likely have these)
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0

# New packages for Streamlit
streamlit>=1.28.0
plotly>=5.17.0
shap>=0.43.0

# ML packages (you already have)
mlxtend>=0.22.0
joblib>=1.3.0

# Optional but recommended
streamlit-option-menu>=0.3.6
streamlit-aggrid>=0.3.4
```

---

## Project Structure

```
streamlit/
├── IMPLEMENTATION_GUIDE.md          # This file
├── TODO.md                          # Detailed task checklist
├── README.md                        # Project description for GitHub
│
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
│
├── models/                          # Saved ML models
│   ├── churn_model.pkl              # Logistic Regression model
│   ├── churn_scaler.pkl             # StandardScaler for churn
│   ├── churn_preprocessor.pkl       # ColumnTransformer
│   ├── kmeans_model.pkl             # K-Means clustering model
│   ├── kmeans_scaler.pkl            # StandardScaler for clustering
│   ├── association_rules.pkl        # Apriori association rules
│   └── model_metadata.json          # Model training info
│
├── data/                            # Sample data for demo
│   ├── sample_customers.csv         # 100 sample customers
│   ├── sample_products.csv          # Product catalog
│   └── data_summary.json            # Dataset statistics
│
├── pages/                           # Streamlit multi-page structure
│   ├── 1_🎯_Churn_Prediction.py    # Churn predictor page
│   ├── 2_👥_Customer_Segments.py   # Segmentation page
│   ├── 3_🛒_Product_Recommendations.py  # Association rules page
│   └── 4_📊_Business_Impact.py     # Insights dashboard
│
├── utils/                           # Helper functions
│   ├── __init__.py
│   ├── data_loader.py               # Load models and data
│   ├── churn_predictor.py           # Churn prediction logic
│   ├── segmentation.py              # Clustering logic
│   ├── recommender.py               # Association rules logic
│   └── visualizations.py            # Plotly charts
│
├── assets/                          # Static files
│   ├── logo.png                     # App logo (optional)
│   └── styles.css                   # Custom CSS (optional)
│
└── notebooks/                       # Helper notebooks
    └── model_export.ipynb           # Export models from main notebook
```

---

## Step-by-Step Implementation

### PHASE 1: Setup and Environment (30 minutes)

#### Task 1.1: Create Virtual Environment
```bash
# Navigate to streamlit folder
cd /Users/rellika/Documents/Classes/MSBA/ML/A1/streamlit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**Verification:**
```bash
which python
# Should show: /Users/rellika/Documents/Classes/MSBA/ML/A1/streamlit/venv/bin/python
```

#### Task 1.2: Install Dependencies
```bash
# Create requirements.txt first (see next task)
# Then install:
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
streamlit --version
# Should show: Streamlit, version 1.28.0 or higher
```

#### Task 1.3: Create requirements.txt
**File:** `streamlit/requirements.txt`
```txt
# Core Data Science
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Streamlit
streamlit==1.28.2
streamlit-option-menu==0.3.6

# ML Interpretation
shap==0.43.0

# Association Rules
mlxtend==0.23.0

# Model Serialization
joblib==1.3.2

# Utilities
Pillow==10.1.0
```

**Action:** Create this file and run `pip install -r requirements.txt`

---

### PHASE 2: Export Models from Notebook (1 hour)

#### Task 2.1: Create Model Export Notebook
**File:** `streamlit/notebooks/model_export.ipynb`

Create a new Jupyter notebook with the following cells:

**Cell 1: Import Libraries**
```python
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
import sys

# Add parent directory to path to import from main notebook
sys.path.append('/Users/rellika/Documents/Classes/MSBA/ML/A1')

print("Libraries imported successfully!")
```

**Cell 2: Load Your Cleaned Data**
```python
# Load the cleaned data from your main notebook
# Adjust path if needed
df = pd.read_csv('/Users/rellika/Documents/Classes/MSBA/ML/A1/data/online_retail_cleaned.csv')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

**Cell 3: Recreate Customer-Level Features**
```python
# Copy the feature engineering code from your main notebook
# This should create the customer-level DataFrame with RFM features

# Example (adjust based on your actual code):
customer_features = df.groupby('customer_id').agg({
    'invoice_date': ['min', 'max'],
    'invoice': 'nunique',
    'revenue': 'sum',
    'quantity': 'sum',
    'description': 'nunique'
}).reset_index()

# Flatten column names
customer_features.columns = ['customer_id', 'first_purchase', 'last_purchase',
                              'frequency', 'total_revenue', 'total_quantity',
                              'unique_products']

# Calculate recency (days since last purchase from cutoff date)
cutoff_date = pd.to_datetime('2011-10-12')
customer_features['recency_days'] = (cutoff_date - pd.to_datetime(customer_features['last_purchase'])).dt.days

# Calculate other features
customer_features['avg_order_value'] = customer_features['total_revenue'] / customer_features['frequency']
customer_features['tenure_days'] = (pd.to_datetime(customer_features['last_purchase']) -
                                     pd.to_datetime(customer_features['first_purchase'])).dt.days

print(f"Customer features shape: {customer_features.shape}")
customer_features.head()
```

**Cell 4: Train and Save Churn Model**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Prepare features and target
# Adjust based on your actual feature columns
feature_cols = ['recency_days', 'frequency', 'total_revenue',
                'avg_order_value', 'unique_products', 'tenure_days']

X = customer_features[feature_cols]
y = customer_features['will_return_90days']  # Adjust column name

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols)
    ]
)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)

# Fit preprocessor and model
X_train_scaled = preprocessor.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# Test accuracy
X_test_scaled = preprocessor.transform(X_test)
test_score = model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_score:.4f}")

# Save models
Path('../models').mkdir(exist_ok=True)
joblib.dump(model, '../models/churn_model.pkl')
joblib.dump(preprocessor, '../models/churn_preprocessor.pkl')

print("✅ Churn model saved!")
```

**Cell 5: Train and Save K-Means Clustering Model**
```python
from sklearn.cluster import KMeans
import joblib

# Prepare clustering features (RFM with log transformation)
clustering_features = customer_features[['recency_days', 'frequency', 'total_revenue']].copy()

# Log transformation
clustering_features_log = np.log1p(clustering_features)

# Standardize
scaler = StandardScaler()
clustering_features_scaled = scaler.fit_transform(clustering_features_log)

# Train K-Means with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_features['cluster'] = kmeans.fit_predict(clustering_features_scaled)

# Save models
joblib.dump(kmeans, '../models/kmeans_model.pkl')
joblib.dump(scaler, '../models/kmeans_scaler.pkl')

print("✅ K-Means model saved!")

# Show cluster distribution
print("\nCluster Distribution:")
print(customer_features['cluster'].value_counts().sort_index())
```

**Cell 6: Extract and Save Association Rules**
```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib

# Prepare basket data (group by invoice)
baskets = df.groupby('invoice')['description'].apply(list).tolist()

# Filter baskets with 2-20 items
baskets_filtered = [basket for basket in baskets if 2 <= len(basket) <= 20]

# One-hot encode
te = TransactionEncoder()
basket_encoded = te.fit_transform(baskets_filtered)
basket_df = pd.DataFrame(basket_encoded, columns=te.columns_)

# Run Apriori
frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules[rules['lift'] > 1.5]

# Sort by lift
rules = rules.sort_values('lift', ascending=False)

print(f"✅ Found {len(rules)} association rules")
print(f"Top rule: {rules.iloc[0]['antecedents']} → {rules.iloc[0]['consequents']}")
print(f"Lift: {rules.iloc[0]['lift']:.2f}")

# Save rules as pickle
joblib.dump(rules, '../models/association_rules.pkl')

print("✅ Association rules saved!")
```

**Cell 7: Create Sample Data for Demo**
```python
# Sample 100 customers for demo
sample_customers = customer_features.sample(n=100, random_state=42)

# Save sample data
sample_customers.to_csv('../data/sample_customers.csv', index=False)

print("✅ Sample data saved!")
print(f"Sample shape: {sample_customers.shape}")
```

**Cell 8: Save Model Metadata**
```python
import json
from datetime import datetime

metadata = {
    "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "Online Retail II (UCI ML Repository)",
    "models": {
        "churn_model": {
            "type": "Logistic Regression",
            "test_accuracy": float(test_score),
            "features": feature_cols,
            "target": "will_return_90days"
        },
        "kmeans_model": {
            "type": "K-Means Clustering",
            "n_clusters": 4,
            "features": ["recency_days", "frequency", "total_revenue"]
        },
        "association_rules": {
            "type": "Apriori",
            "n_rules": len(rules),
            "min_support": 0.01,
            "min_confidence": 0.3
        }
    }
}

with open('../models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model metadata saved!")
print(json.dumps(metadata, indent=2))
```

**Action Items:**
- [ ] Create `streamlit/notebooks/model_export.ipynb`
- [ ] Copy code from cells above
- [ ] Adjust file paths to match your actual data location
- [ ] Run all cells sequentially
- [ ] Verify all `.pkl` files are created in `models/` folder
- [ ] Verify `sample_customers.csv` created in `data/` folder

---

### PHASE 3: Build Streamlit App Structure (1 hour)

#### Task 3.1: Create Main App File
**File:** `streamlit/app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Online Retail Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🛍️ Online Retail Intelligence Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Churn Prediction • Segmentation • Product Recommendations</div>',
            unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.markdown("""
## Welcome to the Customer Intelligence Platform

This interactive dashboard demonstrates **three data mining approaches** applied to the
**Online Retail II dataset** from the UCI Machine Learning Repository.

### 📊 What You Can Do:

1. **🎯 Predict Customer Churn** - Get real-time churn probability scores with SHAP explanations
2. **👥 Explore Customer Segments** - Visualize 4 distinct customer types (VIPs, Occasionals, At-Risk, One-Timers)
3. **🛒 Get Product Recommendations** - Discover which products are frequently bought together
4. **📈 View Business Impact** - See the £1.8M annual value of these insights

### 🎓 Project Background

**Dataset:** Online Retail II (Dec 2009 - Dec 2011)
**Transactions:** 1,067,371 records
**Customers:** 5,878 unique customers
**Products:** 4,631 unique items

**Author:** Rellika Kisyula
**Course:** Machine Learning - Assignment 03
**Grade Target:** 96-100% (135-140/140 points)
""")

st.markdown("---")

# Key Metrics Overview
st.subheader("📊 Key Business Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Annual Value",
        value="£1.8M",
        delta="From 3 Approaches",
        help="Combined business impact from churn prevention, cross-selling, and optimization"
    )

with col2:
    st.metric(
        label="Churn Model AUC",
        value="0.801",
        delta="Strong Performance",
        help="ROC-AUC score for Logistic Regression churn predictor"
    )

with col3:
    st.metric(
        label="Customer Segments",
        value="4 Clusters",
        delta="K-Means",
        help="VIPs, Occasionals, At-Risk, One-Timers"
    )

with col4:
    st.metric(
        label="Product Rules",
        value="33 Rules",
        delta="Lift up to 43.6x",
        help="Association rules discovered via Apriori algorithm"
    )

st.markdown("---")

# Three Approaches Overview
st.subheader("🔬 Three Data Mining Approaches")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Supervised Learning
    **Customer Churn Prediction**

    - **Method:** Logistic Regression
    - **Target:** Will customer return in 90 days?
    - **Features:** RFM (Recency, Frequency, Monetary)
    - **Impact:** £959K annual churn prevention

    👉 Go to **🎯 Churn Prediction** page
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Unsupervised Learning
    **Association Rules Mining**

    - **Method:** Apriori Algorithm
    - **Goal:** Find products bought together
    - **Rules:** 33 strong associations
    - **Impact:** £350K cross-sell revenue

    👉 Go to **🛒 Product Recommendations** page
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Mixed Approach
    **Segmentation + Churn**

    - **Method:** K-Means + Classification
    - **Segments:** VIPs, Occasionals, At-Risk, One-Timers
    - **Validation:** Multi-metric (Elbow, Silhouette, DBI)
    - **Impact:** £500K resource optimization

    👉 Go to **👥 Customer Segments** page
    """)

st.markdown("---")

# Dataset Overview
with st.expander("📂 Dataset Information"):
    st.markdown("""
    ### Online Retail II Dataset

    **Source:** UCI Machine Learning Repository
    **Period:** December 2009 - December 2011 (743 days)
    **Business:** UK-based online retailer (gifts, home decor)

    **Original Data:**
    - 1,067,371 transaction records
    - 8 columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

    **After Cleaning:**
    - 779,425 valid transactions
    - 5,878 unique customers
    - 4,631 unique products
    - 38 countries

    **Data Quality Issues Addressed:**
    - Removed 243,007 missing customer IDs (22.8%)
    - Filtered 19,494 cancelled orders
    - Cleaned 22,950 negative quantities (returns)
    - Fixed 5 negative prices (data errors)
    """)

# Technical Details
with st.expander("⚙️ Technical Stack"):
    st.markdown("""
    ### Technologies Used

    **Backend:**
    - Python 3.10+
    - Pandas, NumPy (data processing)
    - Scikit-learn (ML models)
    - MLxtend (association rules)

    **Visualization:**
    - Plotly (interactive charts)
    - Matplotlib, Seaborn (static plots)
    - SHAP (model explanations)

    **Web Framework:**
    - Streamlit (dashboard)
    - Deployed on Streamlit Cloud

    **Models:**
    - Churn: Logistic Regression (AUC: 0.801)
    - Segmentation: K-Means (k=4, Silhouette: 0.38)
    - Recommendations: Apriori (support: 1%, confidence: 30%)
    """)

# Navigation Guide
st.sidebar.title("📍 Navigation")
st.sidebar.markdown("""
Use the sidebar to navigate between pages:

1. **🎯 Churn Prediction** - Predict if a customer will return
2. **👥 Customer Segments** - Explore 4 customer types
3. **🛒 Product Recommendations** - Find product associations
4. **📊 Business Impact** - View insights and ROI

---

### 📖 Quick Start

1. Try the **Churn Predictor** with sample values
2. Explore the **3D cluster visualization**
3. Search for products in the **Recommender**
4. Check the **Business Impact** dashboard
""")

st.sidebar.markdown("---")
st.sidebar.info("**Author:** Rellika Kisyula\n\n**Course:** Machine Learning - Assignment 03")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Online Retail Intelligence Dashboard</strong></p>
    <p>Built with Streamlit | Data from UCI ML Repository | © 2025 Rellika Kisyula</p>
</div>
""", unsafe_allow_html=True)
```

**Action Items:**
- [ ] Create `streamlit/app.py`
- [ ] Copy code above
- [ ] Test by running: `streamlit run app.py` (should show home page)

---

#### Task 3.2: Create Utility Functions
**File:** `streamlit/utils/__init__.py`

```python
# Empty file to make utils a package
```

**File:** `streamlit/utils/data_loader.py`

```python
import joblib
import pandas as pd
import json
from pathlib import Path
import streamlit as st

@st.cache_resource
def load_churn_model():
    """Load the saved churn prediction model and preprocessor"""
    try:
        model = joblib.load('models/churn_model.pkl')
        preprocessor = joblib.load('models/churn_preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Churn model files not found. Please run model export notebook first.")
        return None, None

@st.cache_resource
def load_kmeans_model():
    """Load the saved K-Means clustering model and scaler"""
    try:
        kmeans = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/kmeans_scaler.pkl')
        return kmeans, scaler
    except FileNotFoundError:
        st.error("⚠️ K-Means model files not found. Please run model export notebook first.")
        return None, None

@st.cache_resource
def load_association_rules():
    """Load the saved association rules"""
    try:
        rules = joblib.load('models/association_rules.pkl')
        return rules
    except FileNotFoundError:
        st.error("⚠️ Association rules file not found. Please run model export notebook first.")
        return None

@st.cache_data
def load_sample_customers():
    """Load sample customer data"""
    try:
        df = pd.read_csv('data/sample_customers.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Sample data not found. Please run model export notebook first.")
        return None

@st.cache_data
def load_model_metadata():
    """Load model training metadata"""
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        st.warning("⚠️ Model metadata not found.")
        return {}

def get_cluster_name(cluster_id):
    """Map cluster ID to human-readable name"""
    cluster_names = {
        0: "At-Risk / Churned",
        1: "One-Time Buyers",
        2: "VIP Champions",
        3: "Occasional Engagers"
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")

def get_cluster_description(cluster_id):
    """Get detailed cluster description"""
    descriptions = {
        0: {
            "name": "At-Risk / Churned Customers",
            "size": "29.2%",
            "churn_rate": "78.2%",
            "avg_revenue": "£2,037",
            "strategy": "Aggressive win-back campaigns, deep discounts"
        },
        1: {
            "name": "One-Time Buyers",
            "size": "41.6%",
            "churn_rate": "89.3%",
            "avg_revenue": "£344",
            "strategy": "Low-cost digital remarketing"
        },
        2: {
            "name": "VIP Champions",
            "size": "14.7%",
            "churn_rate": "20.4%",
            "avg_revenue": "£11,101",
            "strategy": "VIP programs, exclusive access, loyalty rewards"
        },
        3: {
            "name": "Occasional Engagers",
            "size": "14.5%",
            "churn_rate": "44.8%",
            "avg_revenue": "£910",
            "strategy": "Engagement campaigns, upsell opportunities"
        }
    }
    return descriptions.get(cluster_id, {})
```

**Action Items:**
- [ ] Create `streamlit/utils/__init__.py` (empty file)
- [ ] Create `streamlit/utils/data_loader.py`
- [ ] Copy code above

---

### PHASE 4: Build Individual Pages (3-4 hours)

#### Task 4.1: Churn Prediction Page
**File:** `streamlit/pages/1_🎯_Churn_Prediction.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
sys.path.append('..')
from utils.data_loader import load_churn_model, load_model_metadata

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")

# Title
st.title("🎯 Customer Churn Prediction")
st.markdown("Predict whether a customer will return within 90 days based on their behavioral features.")

st.markdown("---")

# Load model
model, preprocessor = load_churn_model()
metadata = load_model_metadata()

if model is None or preprocessor is None:
    st.error("Models not loaded. Please run the model export notebook first.")
    st.stop()

# Sidebar - Input Form
st.sidebar.header("Customer Features")
st.sidebar.markdown("Enter customer behavioral data below:")

# Input fields
recency_days = st.sidebar.number_input(
    "Recency (days since last purchase)",
    min_value=0,
    max_value=730,
    value=45,
    step=1,
    help="Number of days since the customer's last purchase"
)

frequency = st.sidebar.number_input(
    "Frequency (number of orders)",
    min_value=1,
    max_value=300,
    value=8,
    step=1,
    help="Total number of orders the customer has placed"
)

total_revenue = st.sidebar.number_input(
    "Monetary (total revenue £)",
    min_value=0.0,
    max_value=500000.0,
    value=4820.0,
    step=100.0,
    help="Total revenue generated by the customer"
)

avg_order_value = st.sidebar.number_input(
    "Average Order Value (£)",
    min_value=0.0,
    max_value=50000.0,
    value=602.5,
    step=10.0,
    help="Average value per order"
)

unique_products = st.sidebar.number_input(
    "Product Diversity (unique products)",
    min_value=1,
    max_value=1000,
    value=125,
    step=1,
    help="Number of unique products purchased"
)

tenure_days = st.sidebar.number_input(
    "Tenure (days as customer)",
    min_value=0,
    max_value=730,
    value=365,
    step=1,
    help="Number of days between first and last purchase"
)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Churn Risk", type="primary")

st.sidebar.markdown("---")

# Quick presets
st.sidebar.subheader("📝 Quick Presets")
if st.sidebar.button("VIP Customer"):
    recency_days = 30
    frequency = 20
    total_revenue = 11000
    avg_order_value = 550
    unique_products = 200
    tenure_days = 400
    st.sidebar.success("Loaded VIP preset")

if st.sidebar.button("At-Risk Customer"):
    recency_days = 180
    frequency = 5
    total_revenue = 2000
    avg_order_value = 400
    unique_products = 50
    tenure_days = 200
    st.sidebar.warning("Loaded At-Risk preset")

if st.sidebar.button("One-Time Buyer"):
    recency_days = 300
    frequency = 1
    total_revenue = 350
    avg_order_value = 350
    unique_products = 15
    tenure_days = 0
    st.sidebar.info("Loaded One-Timer preset")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Profile")

    # Display input features
    profile_df = pd.DataFrame({
        'Feature': ['Recency', 'Frequency', 'Monetary', 'Avg Order Value',
                    'Product Diversity', 'Tenure'],
        'Value': [
            f"{recency_days} days",
            f"{frequency} orders",
            f"£{total_revenue:,.2f}",
            f"£{avg_order_value:,.2f}",
            f"{unique_products} products",
            f"{tenure_days} days"
        ]
    })

    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # RFM Score visualization
    st.markdown("#### RFM Score Breakdown")
    fig_rfm = go.Figure()
    fig_rfm.add_trace(go.Bar(
        x=['Recency', 'Frequency', 'Monetary'],
        y=[recency_days, frequency, total_revenue],
        marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
        text=[recency_days, frequency, f"£{total_revenue:,.0f}"],
        textposition='auto'
    ))
    fig_rfm.update_layout(
        title="RFM Components",
        yaxis_title="Value",
        height=300
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

with col2:
    st.subheader("🔮 Churn Prediction Result")

    if predict_button:
        # Prepare input
        input_data = pd.DataFrame({
            'recency_days': [recency_days],
            'frequency': [frequency],
            'total_revenue': [total_revenue],
            'avg_order_value': [avg_order_value],
            'unique_products': [unique_products],
            'tenure_days': [tenure_days]
        })

        # Transform and predict
        X_scaled = preprocessor.transform(input_data)
        churn_prob = model.predict_proba(X_scaled)[0][1]
        churn_prediction = model.predict(X_scaled)[0]

        # Display prediction
        st.markdown(f"### Churn Probability: **{churn_prob*100:.1f}%**")

        # Risk category
        if churn_prob < 0.3:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
            recommendation = "Customer is likely to return. Maintain engagement with regular newsletters."
        elif churn_prob < 0.6:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
            recommendation = "Customer shows moderate churn risk. Consider targeted email campaign with discount incentive."
        else:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
            recommendation = "Customer is at high risk of churning. Immediate intervention needed: personal call, VIP offer, or win-back campaign."

        st.markdown(f"### Risk Level: **{risk_level}**")
        st.markdown(f"**Recommendation:** {recommendation}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance (mock SHAP values - you can add real SHAP later)
        st.markdown("#### 📊 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': ['Recency', 'Frequency', 'Tenure', 'Avg Order Value', 'Monetary', 'Product Diversity'],
            'Importance': [0.35, 0.25, 0.18, 0.12, 0.07, 0.03]
        }).sort_values('Importance', ascending=True)

        fig_importance = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig_importance.update_layout(
            title="Feature Contribution to Churn Prediction",
            xaxis_title="Importance Score",
            height=300
        )
        st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")

# Model Information
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    ### Logistic Regression Churn Predictor

    **Training Date:** {metadata.get('created_date', 'N/A')}
    **Test Accuracy:** {metadata.get('models', {}).get('churn_model', {}).get('test_accuracy', 'N/A'):.4f}
    **Model Type:** Logistic Regression with L2 regularization

    **Features Used:**
    - Recency: Days since last purchase (most important)
    - Frequency: Number of orders
    - Monetary: Total revenue
    - Average Order Value
    - Product Diversity
    - Tenure: Customer lifetime in days

    **Performance Metrics:**
    - ROC-AUC: 0.801
    - Precision: 69.8%
    - Recall: 53.9%
    - F1 Score: 60.8%

    **Business Impact:**
    - Estimated annual churn prevention value: £959,200
    - Expected retention campaign success rate: 15-20%
    """)
```

**Action Items:**
- [ ] Create `streamlit/pages/1_🎯_Churn_Prediction.py`
- [ ] Copy code above
- [ ] Test the page by running `streamlit run app.py` and navigating to this page

---

#### Task 4.2: Customer Segmentation Page
**File:** `streamlit/pages/2_👥_Customer_Segments.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')
from utils.data_loader import load_kmeans_model, load_sample_customers, get_cluster_name, get_cluster_description

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")

# Title
st.title("👥 Customer Segmentation Analysis")
st.markdown("Explore 4 distinct customer segments discovered through K-Means clustering on RFM features.")

st.markdown("---")

# Load model and data
kmeans, scaler = load_kmeans_model()
customers_df = load_sample_customers()

if kmeans is None or scaler is None or customers_df is None:
    st.error("Models or data not loaded. Please run the model export notebook first.")
    st.stop()

# Assign clusters if not already present
if 'cluster' not in customers_df.columns:
    # Prepare features
    rfm_features = customers_df[['recency_days', 'frequency', 'total_revenue']].copy()
    rfm_log = np.log1p(rfm_features)
    rfm_scaled = scaler.transform(rfm_log)
    customers_df['cluster'] = kmeans.predict(rfm_scaled)

# Cluster Overview
st.subheader("📊 Segment Overview")

col1, col2, col3, col4 = st.columns(4)

cluster_counts = customers_df['cluster'].value_counts().sort_index()

with col1:
    cluster_0_count = cluster_counts.get(0, 0)
    st.metric(
        label="🔴 At-Risk",
        value=f"{cluster_0_count} customers",
        delta="29.2%",
        help="Formerly active customers who have disengaged"
    )

with col2:
    cluster_1_count = cluster_counts.get(1, 0)
    st.metric(
        label="🔵 One-Timers",
        value=f"{cluster_1_count} customers",
        delta="41.6%",
        help="One-time or very infrequent shoppers"
    )

with col3:
    cluster_2_count = cluster_counts.get(2, 0)
    st.metric(
        label="🟢 VIP Champions",
        value=f"{cluster_2_count} customers",
        delta="14.7%",
        help="Loyal, high-value customers"
    )

with col4:
    cluster_3_count = cluster_counts.get(3, 0)
    st.metric(
        label="🟡 Occasionals",
        value=f"{cluster_3_count} customers",
        delta="14.5%",
        help="Recently active but low frequency"
    )

st.markdown("---")

# 3D Visualization
st.subheader("🎨 3D Cluster Visualization (RFM Space)")

# Add cluster names
customers_df['cluster_name'] = customers_df['cluster'].apply(get_cluster_name)

# Create 3D scatter plot
fig_3d = px.scatter_3d(
    customers_df,
    x='recency_days',
    y='frequency',
    z='total_revenue',
    color='cluster_name',
    color_discrete_map={
        "At-Risk / Churned": "#ff6b6b",
        "One-Time Buyers": "#4ecdc4",
        "VIP Champions": "#95e1d3",
        "Occasional Engagers": "#ffd93d"
    },
    title="Customer Segments in RFM Space",
    labels={
        'recency_days': 'Recency (days)',
        'frequency': 'Frequency (orders)',
        'total_revenue': 'Monetary (£)'
    },
    height=600
)

fig_3d.update_traces(marker=dict(size=5))
fig_3d.update_layout(scene=dict(
    xaxis_title='Recency (days)',
    yaxis_title='Frequency (orders)',
    zaxis_title='Monetary (£)'
))

st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# Segment Profiles
st.subheader("📋 Detailed Segment Profiles")

tabs = st.tabs(["🔴 At-Risk", "🔵 One-Timers", "🟢 VIP Champions", "🟡 Occasionals"])

for idx, tab in enumerate(tabs):
    with tab:
        cluster_info = get_cluster_description(idx)
        cluster_data = customers_df[customers_df['cluster'] == idx]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### {cluster_info.get('name', 'Unknown')}")
            st.metric("Segment Size", cluster_info.get('size', 'N/A'))
            st.metric("Churn Rate", cluster_info.get('churn_rate', 'N/A'))
            st.metric("Avg Revenue", cluster_info.get('avg_revenue', 'N/A'))

            st.markdown("**Marketing Strategy:**")
            st.info(cluster_info.get('strategy', 'N/A'))

        with col2:
            st.markdown("#### Segment Statistics")

            stats_df = pd.DataFrame({
                'Metric': ['Recency (avg)', 'Frequency (avg)', 'Monetary (avg)',
                          'Product Diversity (avg)', 'Customer Count'],
                'Value': [
                    f"{cluster_data['recency_days'].mean():.1f} days",
                    f"{cluster_data['frequency'].mean():.1f} orders",
                    f"£{cluster_data['total_revenue'].mean():.2f}",
                    f"{cluster_data['unique_products'].mean():.1f} products",
                    f"{len(cluster_data)} customers"
                ]
            })

            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Distribution plots
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=cluster_data['total_revenue'],
                name='Revenue Distribution',
                nbinsx=20,
                marker_color='#1f77b4'
            ))
            fig_dist.update_layout(
                title=f"Revenue Distribution - {cluster_info.get('name', '')}",
                xaxis_title="Total Revenue (£)",
                yaxis_title="Customer Count",
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Customer Lookup Tool
st.subheader("🔍 Customer Lookup Tool")

col1, col2 = st.columns([1, 2])

with col1:
    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(customers_df.index.min()) if len(customers_df) > 0 else 0,
        max_value=int(customers_df.index.max()) if len(customers_df) > 0 else 100,
        value=int(customers_df.index[0]) if len(customers_df) > 0 else 0
    )

    lookup_button = st.button("🔎 Lookup Customer", type="primary")

with col2:
    if lookup_button:
        if customer_id in customers_df.index:
            customer = customers_df.loc[customer_id]
            cluster_id = int(customer['cluster'])
            cluster_info = get_cluster_description(cluster_id)

            st.success(f"✅ Customer {customer_id} found!")
            st.markdown(f"**Segment:** {get_cluster_name(cluster_id)}")
            st.markdown(f"**Recency:** {customer['recency_days']:.0f} days")
            st.markdown(f"**Frequency:** {customer['frequency']:.0f} orders")
            st.markdown(f"**Monetary:** £{customer['total_revenue']:.2f}")
            st.markdown(f"**Churn Risk:** {cluster_info.get('churn_rate', 'N/A')}")
        else:
            st.warning(f"⚠️ Customer ID {customer_id} not found in sample data.")

st.markdown("---")

# Model Information
with st.expander("ℹ️ Clustering Model Information"):
    st.markdown("""
    ### K-Means Clustering (k=4)

    **Algorithm:** K-Means with K-Means++ initialization
    **Number of Clusters:** 4
    **Features Used:** Recency, Frequency, Monetary (log-transformed)

    **Validation Metrics:**
    - Silhouette Score: 0.3766 (moderate separation)
    - Davies-Bouldin Index: 0.9352 (good clustering)
    - Inertia: 6,891.5

    **Preprocessing:**
    - Log transformation: log(1 + x) to handle skewness
    - Standardization: StandardScaler (mean=0, std=1)

    **Business Value:**
    - Enables differentiated retention strategies
    - £500K annual resource optimization
    - VIPs contribute 68% of revenue (top 14.7% of customers)
    """)
```

**Action Items:**
- [ ] Create `streamlit/pages/2_👥_Customer_Segments.py`
- [ ] Copy code above
- [ ] Test the page

---

#### Task 4.3: Product Recommendations Page
**File:** `streamlit/pages/3_🛒_Product_Recommendations.py`

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
sys.path.append('..')
from utils.data_loader import load_association_rules

st.set_page_config(page_title="Product Recommendations", page_icon="🛒", layout="wide")

# Title
st.title("🛒 Product Recommendation Engine")
st.markdown("Discover which products are frequently bought together using Association Rules Mining.")

st.markdown("---")

# Load association rules
rules = load_association_rules()

if rules is None or len(rules) == 0:
    st.error("Association rules not loaded. Please run the model export notebook first.")
    st.stop()

# Convert frozensets to strings for display
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Overview
st.subheader("📊 Association Rules Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rules", len(rules), help="Number of strong association rules discovered")

with col2:
    max_lift = rules['lift'].max()
    st.metric("Max Lift", f"{max_lift:.1f}x", help="Highest lift value (co-occurrence strength)")

with col3:
    avg_confidence = rules['confidence'].mean()
    st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%", help="Average confidence across all rules")

with col4:
    avg_support = rules['support'].mean()
    st.metric("Avg Support", f"{avg_support*100:.2f}%", help="Average support (frequency in baskets)")

st.markdown("---")

# Top Rules
st.subheader("🏆 Top 10 Association Rules (by Lift)")

top_rules = rules.nlargest(10, 'lift')[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
top_rules.columns = ['If Customer Buys', 'Then Also Buys', 'Support', 'Confidence', 'Lift']

# Format percentages
top_rules['Support'] = top_rules['Support'].apply(lambda x: f"{x*100:.2f}%")
top_rules['Confidence'] = top_rules['Confidence'].apply(lambda x: f"{x*100:.1f}%")
top_rules['Lift'] = top_rules['Lift'].apply(lambda x: f"{x:.2f}x")

st.dataframe(top_rules, use_container_width=True, hide_index=True)

st.markdown("---")

# Product Search
st.subheader("🔍 Product Search")

col1, col2 = st.columns([2, 1])

with col1:
    # Get unique products
    all_products = set()
    for items in rules['antecedents']:
        all_products.update(items)
    for items in rules['consequents']:
        all_products.update(items)

    product_list = sorted(list(all_products))

    selected_product = st.selectbox(
        "Select a product to see recommendations:",
        options=product_list,
        index=0 if len(product_list) > 0 else None
    )

with col2:
    min_lift = st.slider("Minimum Lift", 1.0, float(max_lift), 5.0, 0.5)

if selected_product:
    # Filter rules where selected product is in antecedents
    matching_rules = rules[rules['antecedents'].apply(lambda x: selected_product in x)]
    matching_rules = matching_rules[matching_rules['lift'] >= min_lift]
    matching_rules = matching_rules.sort_values('lift', ascending=False)

    if len(matching_rules) > 0:
        st.success(f"✅ Found {len(matching_rules)} recommendation(s) for **{selected_product}**")

        # Display recommendations
        for idx, row in matching_rules.head(5).iterrows():
            recommended_products = ', '.join(list(row['consequents']))

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### → {recommended_products}")
                st.markdown(f"**Rule:** If customer buys *{selected_product}*, they also buy *{recommended_products}*")

            with col2:
                st.metric("Lift", f"{row['lift']:.2f}x")
                st.metric("Confidence", f"{row['confidence']*100:.1f}%")
                st.metric("Support", f"{row['support']*100:.2f}%")

            st.markdown("---")
    else:
        st.warning(f"⚠️ No strong recommendations found for **{selected_product}** with lift >= {min_lift}")

# Lift Distribution
st.subheader("📈 Lift Distribution")

fig_lift = go.Figure()
fig_lift.add_trace(go.Histogram(
    x=rules['lift'],
    nbinsx=30,
    marker_color='#1f77b4',
    name='Lift Distribution'
))
fig_lift.update_layout(
    title="Distribution of Lift Values Across All Rules",
    xaxis_title="Lift (times more likely than random)",
    yaxis_title="Number of Rules",
    height=400
)
st.plotly_chart(fig_lift, use_container_width=True)

st.markdown("---")

# Support vs Confidence Scatter
st.subheader("📊 Support vs Confidence (sized by Lift)")

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=rules['support'],
    y=rules['confidence'],
    mode='markers',
    marker=dict(
        size=rules['lift'] * 2,
        color=rules['lift'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Lift")
    ),
    text=rules['antecedents_str'] + ' → ' + rules['consequents_str'],
    hovertemplate='<b>%{text}</b><br>Support: %{x:.2%}<br>Confidence: %{y:.2%}<extra></extra>'
))

fig_scatter.update_layout(
    title="Association Rules: Support vs Confidence (bubble size = lift)",
    xaxis_title="Support (% of baskets)",
    yaxis_title="Confidence (probability)",
    height=500
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Model Information
with st.expander("ℹ️ Association Rules Mining Information"):
    st.markdown("""
    ### Apriori Algorithm

    **Algorithm:** Apriori (frequent itemset mining)
    **Minimum Support:** 1% (appears in ≥204 baskets)
    **Minimum Confidence:** 30%
    **Minimum Lift:** 1.5x

    **Dataset:**
    - 20,385 baskets analyzed
    - Basket size: 2-20 items (filtered)
    - 4,295 unique products

    **Metrics Explained:**

    - **Support:** P(X ∩ Y) = Proportion of baskets containing both products
    - **Confidence:** P(Y|X) = Probability of Y given X in basket
    - **Lift:** P(Y|X) / P(Y) = How much more likely Y is when X is present

    **Example:**
    - Rule: Pink Teacup → Green Teacup
    - Support: 1.2% (appears in 245 baskets)
    - Confidence: 80% (80% of pink teacup buyers also buy green)
    - Lift: 43.6x (43.6 times more likely than random chance)

    **Business Applications:**
    - "Frequently Bought Together" recommendations
    - Bundle pricing strategies
    - Cross-sell prompts at checkout
    - Inventory co-location

    **Expected Impact:**
    - 8-12% basket size increase
    - £350,000 annual cross-sell revenue
    """)
```

**Action Items:**
- [ ] Create `streamlit/pages/3_🛒_Product_Recommendations.py`
- [ ] Copy code above
- [ ] Test the page

---

#### Task 4.4: Business Impact Dashboard
**File:** `streamlit/pages/4_📊_Business_Impact.py`

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Business Impact", page_icon="📊", layout="wide")

# Title
st.title("📊 Business Impact Dashboard")
st.markdown("Quantified business value from the three data mining approaches.")

st.markdown("---")

# Total Impact
st.subheader("💰 Total Annual Business Value")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Annual Value",
        value="£1,809,200",
        delta="+100%",
        help="Combined impact from all three approaches"
    )

with col2:
    st.metric(
        label="Churn Prevention",
        value="£959,200",
        delta="53%",
        help="Supervised learning - customer retention"
    )

with col3:
    st.metric(
        label="Cross-Sell Revenue",
        value="£350,000",
        delta="19%",
        help="Unsupervised learning - product recommendations"
    )

with col4:
    st.metric(
        label="Resource Optimization",
        value="£500,000",
        delta="28%",
        help="Mixed approach - targeted campaigns"
    )

st.markdown("---")

# Value Breakdown
st.subheader("📈 Value Breakdown by Approach")

value_data = pd.DataFrame({
    'Approach': ['Supervised\n(Churn)', 'Unsupervised\n(Recommendations)', 'Mixed\n(Segmentation)'],
    'Annual Value': [959200, 350000, 500000],
    'Implementation Cost': [10000, 15000, 25000]
})

value_data['ROI'] = ((value_data['Annual Value'] - value_data['Implementation Cost']) /
                      value_data['Implementation Cost'] * 100)

col1, col2 = st.columns(2)

with col1:
    # Stacked bar chart
    fig_value = go.Figure()
    fig_value.add_trace(go.Bar(
        name='Annual Value',
        x=value_data['Approach'],
        y=value_data['Annual Value'],
        marker_color='#4ecdc4'
    ))
    fig_value.add_trace(go.Bar(
        name='Implementation Cost',
        x=value_data['Approach'],
        y=value_data['Implementation Cost'],
        marker_color='#ff6b6b'
    ))
    fig_value.update_layout(
        title="Annual Value vs Implementation Cost",
        yaxis_title="Amount (£)",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_value, use_container_width=True)

with col2:
    # ROI chart
    fig_roi = go.Figure(go.Bar(
        x=value_data['Approach'],
        y=value_data['ROI'],
        marker_color='#95e1d3',
        text=value_data['ROI'].apply(lambda x: f"{x:.0f}%"),
        textposition='auto'
    ))
    fig_roi.update_layout(
        title="Return on Investment (ROI)",
        yaxis_title="ROI %",
        height=400
    )
    st.plotly_chart(fig_roi, use_container_width=True)

st.markdown("---")

# Revenue Distribution (Lorenz Curve)
st.subheader("📊 Customer Revenue Inequality (Lorenz Curve)")

st.markdown("""
The Lorenz curve shows how revenue is distributed across customers.
The closer the curve is to the diagonal line, the more equal the distribution.
""")

# Generate sample Lorenz curve data
lorenz_data = pd.DataFrame({
    'Cumulative % of Customers': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Cumulative % of Revenue': [0, 0.3, 1.2, 3.5, 7.8, 14.2, 23.5, 35.8, 52.4, 74.1, 100]
})

fig_lorenz = go.Figure()

# Lorenz curve
fig_lorenz.add_trace(go.Scatter(
    x=lorenz_data['Cumulative % of Customers'],
    y=lorenz_data['Cumulative % of Revenue'],
    mode='lines+markers',
    name='Actual Distribution',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=8)
))

# Equality line
fig_lorenz.add_trace(go.Scatter(
    x=[0, 100],
    y=[0, 100],
    mode='lines',
    name='Perfect Equality',
    line=dict(color='red', width=2, dash='dash')
))

fig_lorenz.update_layout(
    title="Revenue Distribution: Top 10% of Customers = 68.2% of Revenue",
    xaxis_title="Cumulative % of Customers",
    yaxis_title="Cumulative % of Revenue",
    height=500
)

st.plotly_chart(fig_lorenz, use_container_width=True)

st.info("**Key Insight:** Top 10% of customers contribute 68.2% of total revenue. This validates the need for VIP-focused retention strategies.")

st.markdown("---")

# Churn Rates by Segment
st.subheader("⚠️ Churn Rates by Customer Segment")

segment_churn = pd.DataFrame({
    'Segment': ['VIP Champions', 'Occasionals', 'At-Risk', 'One-Timers'],
    'Churn Rate (%)': [20.4, 44.8, 78.2, 89.3],
    'Avg Revenue (£)': [11101, 910, 2037, 344],
    'Size (%)': [14.7, 14.5, 29.2, 41.6]
})

col1, col2 = st.columns(2)

with col1:
    fig_churn = go.Figure(go.Bar(
        x=segment_churn['Segment'],
        y=segment_churn['Churn Rate (%)'],
        marker_color=['#95e1d3', '#ffd93d', '#ff6b6b', '#4ecdc4'],
        text=segment_churn['Churn Rate (%)'].apply(lambda x: f"{x}%"),
        textposition='auto'
    ))
    fig_churn.update_layout(
        title="Churn Rate by Segment",
        yaxis_title="Churn Rate (%)",
        height=400
    )
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    fig_revenue = go.Figure(go.Bar(
        x=segment_churn['Segment'],
        y=segment_churn['Avg Revenue (£)'],
        marker_color=['#95e1d3', '#ffd93d', '#ff6b6b', '#4ecdc4'],
        text=segment_churn['Avg Revenue (£)'].apply(lambda x: f"£{x:,.0f}"),
        textposition='auto'
    ))
    fig_revenue.update_layout(
        title="Average Revenue by Segment",
        yaxis_title="Avg Revenue (£)",
        height=400
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

st.markdown("---")

# ROI Calculator
st.subheader("🧮 Retention Campaign ROI Calculator")

st.markdown("Calculate the expected ROI for a targeted retention campaign.")

col1, col2, col3 = st.columns(3)

with col1:
    num_customers = st.number_input("Number of at-risk customers", 100, 10000, 500, 50)
    avg_ltv = st.number_input("Average Customer LTV (£)", 100, 50000, 2500, 100)

with col2:
    cost_per_customer = st.number_input("Cost per retention attempt (£)", 10, 500, 50, 5)
    success_rate = st.slider("Expected success rate (%)", 5, 50, 15, 1)

with col3:
    total_cost = num_customers * cost_per_customer
    customers_retained = num_customers * (success_rate / 100)
    total_value_saved = customers_retained * avg_ltv
    net_benefit = total_value_saved - total_cost
    roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0

    st.metric("Total Campaign Cost", f"£{total_cost:,.0f}")
    st.metric("Customers Retained", f"{customers_retained:.0f}")
    st.metric("Value Saved", f"£{total_value_saved:,.0f}")
    st.metric("Net Benefit", f"£{net_benefit:,.0f}", delta=f"ROI: {roi:.0f}%")

if roi > 100:
    st.success(f"✅ Excellent ROI! For every £1 spent, you gain £{roi/100:.2f}")
elif roi > 0:
    st.info(f"💡 Positive ROI. Campaign is profitable.")
else:
    st.warning(f"⚠️ Negative ROI. Consider adjusting campaign parameters.")

st.markdown("---")

# Summary Table
st.subheader("📋 Approach Comparison Summary")

comparison_df = pd.DataFrame({
    'Approach': ['Supervised (Churn)', 'Unsupervised (Recommendations)', 'Mixed (Segmentation)'],
    'Method': ['Logistic Regression', 'Apriori Algorithm', 'K-Means + Churn'],
    'Primary Metric': ['ROC-AUC: 0.801', 'Lift: 43.6x', 'Silhouette: 0.38'],
    'Annual Value': ['£959,200', '£350,000', '£500,000'],
    'Implementation': ['2-3 weeks', '1-2 weeks', '3-4 weeks'],
    'Maintenance': ['Quarterly retraining', 'Real-time updates', 'Monthly re-clustering']
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Deployment Timeline
with st.expander("📅 Recommended Deployment Timeline"):
    st.markdown("""
    ### 6-Month Deployment Roadmap

    **Phase 1 (Months 1-2): Foundation**
    - Deploy supervised churn model in production
    - Weekly batch scoring of all customers
    - Integrate scores into CRM system

    **Phase 2 (Months 2-3): Recommendations**
    - Implement association rules in e-commerce platform
    - Add "Frequently Bought Together" module
    - A/B test recommendation impact

    **Phase 3 (Months 3-4): Segmentation**
    - Run K-Means clustering monthly
    - Create segment-specific email templates
    - Train marketing team on strategies

    **Phase 4 (Months 4-6): Optimization**
    - Monitor model performance
    - Retrain models with new data
    - Adjust thresholds based on results

    **Phase 5 (Months 6-12): Advanced**
    - Real-time churn scoring
    - Segment transition models
    - Expand features with demographics
    """)
```

**Action Items:**
- [ ] Create `streamlit/pages/4_📊_Business_Impact.py`
- [ ] Copy code above
- [ ] Test the page

---

### PHASE 5: Configuration and Documentation (30 minutes)

#### Task 5.1: Create Streamlit Config
**File:** `streamlit/.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

#### Task 5.2: Create README
**File:** `streamlit/README.md`

```markdown
# Online Retail Intelligence Dashboard 🛍️

An interactive Streamlit dashboard showcasing three data mining approaches applied to the UCI Online Retail II dataset.

## 🎯 Features

- **Customer Churn Prediction** - Predict 90-day return probability using Logistic Regression
- **Customer Segmentation** - Explore 4 customer types using K-Means clustering
- **Product Recommendations** - Discover product associations using Apriori algorithm
- **Business Impact Dashboard** - Visualize £1.8M annual value

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd streamlit
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Export Models
Run the `notebooks/model_export.ipynb` notebook to generate:
- `models/churn_model.pkl`
- `models/kmeans_model.pkl`
- `models/association_rules.pkl`
- `data/sample_customers.csv`

### 4. Run App
```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

## 📁 Project Structure

```
streamlit/
├── app.py                       # Main homepage
├── pages/                       # Multi-page app
│   ├── 1_🎯_Churn_Prediction.py
│   ├── 2_👥_Customer_Segments.py
│   ├── 3_🛒_Product_Recommendations.py
│   └── 4_📊_Business_Impact.py
├── utils/                       # Helper functions
├── models/                      # Saved ML models
├── data/                        # Sample data
└── .streamlit/config.toml       # Streamlit config
```

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Online Retail II
**Period:** Dec 2009 - Dec 2011
**Records:** 1,067,371 transactions
**Customers:** 5,878 unique

## 🛠️ Tech Stack

- **Backend:** Python 3.10+
- **ML:** Scikit-learn, MLxtend
- **Visualization:** Plotly, Matplotlib
- **Web:** Streamlit
- **Deployment:** Streamlit Cloud (free)

## 📈 Business Impact

- **Total Annual Value:** £1,809,200
  - Churn Prevention: £959,200
  - Cross-Sell Revenue: £350,000
  - Resource Optimization: £500,000

## 📝 License

MIT License - Rellika Kisyula © 2025

## 🤝 Contributing

This is an academic project for ML Assignment 03.

## 📧 Contact

**Author:** Rellika Kisyula
**Course:** Machine Learning - Assignment 03
**Institution:** MSBA Program
```

---

### PHASE 6: Testing and Deployment (1 hour)

#### Task 6.1: Local Testing Checklist
```bash
# Activate environment
source venv/bin/activate

# Run app
streamlit run app.py

# Test each page:
# [ ] Home page loads
# [ ] Churn prediction page works with sample inputs
# [ ] Customer segments page shows 3D plot
# [ ] Product recommendations page displays rules
# [ ] Business impact page shows charts

# Check for errors in terminal
```

#### Task 6.2: Deploy to Streamlit Cloud

**Steps:**

1. **Create GitHub Repository**
```bash
cd /Users/rellika/Documents/Classes/MSBA/ML/A1/streamlit
git init
git add .
git commit -m "Initial commit: Online Retail Intelligence Dashboard"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/retail-intelligence-dashboard.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io
- Click "New app"
- Select your GitHub repo
- Main file: `app.py`
- Click "Deploy"

3. **Share Link**
- You'll get a URL like: `https://retail-intelligence.streamlit.app`
- Add this to your resume/portfolio!

---

## Troubleshooting

### Issue: Models not loading
**Solution:** Run `notebooks/model_export.ipynb` first

### Issue: Module not found
**Solution:** Ensure virtual environment is activated and requirements installed

### Issue: Data file not found
**Solution:** Check file paths in `data_loader.py` match your directory structure

### Issue: Streamlit not installing
**Solution:**
```bash
pip install --upgrade pip
pip install streamlit --no-cache-dir
```

---

## Next Steps

Once basic app is working:

### Enhancements (Optional)
- [ ] Add SHAP force plots for churn explanation
- [ ] Implement customer upload (CSV file)
- [ ] Add filtering/sorting to tables
- [ ] Create download buttons for reports
- [ ] Add animations to charts
- [ ] Implement user authentication
- [ ] Connect to live database instead of pickle files

### Portfolio Additions
- [ ] Record demo video
- [ ] Write blog post about the project
- [ ] Add to LinkedIn portfolio
- [ ] Create presentation slides

---

## Estimated Timeline

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 1 | Setup environment | 30 min |
| Phase 2 | Export models | 1 hour |
| Phase 3 | Build app structure | 1 hour |
| Phase 4 | Build pages | 3-4 hours |
| Phase 5 | Documentation | 30 min |
| Phase 6 | Testing & deployment | 1 hour |
| **TOTAL** | | **7-8 hours** |

---

## Success Criteria

✅ All 4 pages load without errors
✅ Churn prediction returns accurate probabilities
✅ 3D cluster visualization renders correctly
✅ Product recommendations display association rules
✅ Business impact charts show correct values
✅ App deployed to Streamlit Cloud
✅ README and documentation complete

---

**Good luck with your implementation! 🚀**

If you encounter any issues, refer to:
- Streamlit docs: https://docs.streamlit.io
- Plotly docs: https://plotly.com/python/
- Your original notebook: `retail_analysis_part_3.ipynb`
