import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ---------------------------
# Data Loading & Generation
# ---------------------------
@st.cache_data
def generate_synthetic_data(n_orders=1000, random_seed=42):
    """
    Generate synthetic supply chain order data.
    """
    np.random.seed(random_seed)
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    products = ['P1', 'P2', 'P3', 'P4', 'P5']
    warehouses = ['W1', 'W2', 'W3']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for i in range(1, n_orders + 1):
        order_date = np.random.choice(date_range)
        product_id = np.random.choice(products)
        warehouse = np.random.choice(warehouses)
        region = np.random.choice(regions)
        order_quantity = np.random.randint(10, 200)
        lead_time = np.random.randint(2, 10)
        shipping_time = lead_time + np.random.randint(0, 5)
        unit_cost = np.random.uniform(5, 20)
        cost = round(order_quantity * unit_cost, 2)
        data.append({
            'order_id': i,
            'order_date': order_date,
            'product_id': product_id,
            'warehouse': warehouse,
            'region': region,
            'order_quantity': order_quantity,
            'lead_time': lead_time,
            'shipping_time': shipping_time,
            'unit_cost': round(unit_cost, 2),
            'cost': cost
        })
    df = pd.DataFrame(data)
    df.sort_values('order_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_data():
    """
    Loads supply chain data from an uploaded CSV file.
    If no file is uploaded, synthetic data will be generated.
    """
    st.sidebar.header("Data Input Options")
    data_file = st.sidebar.file_uploader("Upload your Supply Chain CSV", type=["csv"])
    if data_file is not None:
        try:
            df = pd.read_csv(data_file)
            # Expecting a column 'order_date' in the CSV, convert it to datetime
            df['order_date'] = pd.to_datetime(df['order_date'])
            st.success("Supply chain dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.info("Falling back to synthetic data.")
            df = generate_synthetic_data(1000)
    else:
        st.info("No dataset uploaded. Using synthetic data.")
        df = generate_synthetic_data(1000)
    return df

# ---------------------------
# EDA Plot Functions
# ---------------------------
def plot_order_quantity_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['order_quantity'], kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of Order Quantities")
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Frequency")
    return fig

def plot_lead_vs_shipping_time(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='lead_time', y='shipping_time', data=df, hue='product_id', ax=ax)
    ax.set_title("Lead Time vs Shipping Time")
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel("Shipping Time (days)")
    ax.legend(title='Product ID')
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[['order_quantity', 'lead_time', 'shipping_time', 'unit_cost', 'cost']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    return fig

# ---------------------------
# Forecasting Function
# ---------------------------
def forecast_demand_plot(df, product_id='P1'):
    """
    Forecast demand for a selected product using Exponential Smoothing.
    Returns a matplotlib figure and the head of the forecast series.
    """
    product_df = df[df['product_id'] == product_id].copy()
    product_df['order_date'] = pd.to_datetime(product_df['order_date'])
    # Aggregate daily demand (fill missing dates with 0)
    daily_demand = product_df.groupby('order_date')['order_quantity'].sum().asfreq('D').fillna(0)
    
    if len(daily_demand) < 60:
        st.warning("Not enough data for a reliable forecast. Consider using a dataset with a longer time span.")
        return None, None
    
    train = daily_demand.iloc[:-30]
    test = daily_demand.iloc[-30:]
    
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
        model_fit = model.fit()
        forecast = model_fit.forecast(30)
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Test', color='green')
    ax.plot(forecast.index, forecast, label='Forecast', color='red')
    ax.set_title(f"Demand Forecast for Product {product_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Order Quantity")
    ax.legend()
    
    return fig, forecast.head()

# ---------------------------
# Inventory Optimization Function
# ---------------------------
def optimize_inventory_df(df):
    """
    Calculate Economic Order Quantity (EOQ) for each product.
    Returns a DataFrame with the EOQ results.
    """
    ordering_cost = 50  # dollars per order
    holding_cost = 2    # dollars per unit per year
    
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['order_date'].dt.year
    annual_demand = df.groupby(['product_id', 'year'])['order_quantity'].sum().reset_index()
    
    eoq_results = []
    for product in df['product_id'].unique():
        product_demand = annual_demand[annual_demand['product_id'] == product]
        avg_demand = product_demand['order_quantity'].mean()
        eoq = np.sqrt((2 * avg_demand * ordering_cost) / holding_cost)
        eoq_results.append({
            'product_id': product,
            'avg_annual_demand': round(avg_demand, 2),
            'EOQ': round(eoq, 2)
        })
    return pd.DataFrame(eoq_results)

# ---------------------------
# Predictive Modeling Function
# ---------------------------
def predictive_model_plot(df):
    """
    Build a Linear Regression model to predict shipping time.
    Returns a scatter plot figure, Mean Squared Error, and R-squared.
    """
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=['product_id', 'region', 'warehouse'], drop_first=True)
    
    features = ['order_quantity', 'lead_time', 'unit_cost']
    features += [col for col in df_model.columns 
                 if col.startswith('product_id_') or col.startswith('region_') or col.startswith('warehouse_')]
    target = 'shipping_time'
    
    X = df_model[features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_xlabel("Actual Shipping Time")
    ax.set_ylabel("Predicted Shipping Time")
    ax.set_title("Actual vs Predicted Shipping Time")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    
    return fig, mse, r2

# ---------------------------
# Main Streamlit Application
# ---------------------------
def main():
    st.title("Supply Chain Analytics Dashboard")
    st.markdown("""
    This interactive dashboard demonstrates key supply chain analytics:
    - **Exploratory Data Analysis (EDA)**
    - **Demand Forecasting**
    - **Inventory Optimization (EOQ)**
    - **Predictive Modeling for Shipping Time**
    
    Upload your own supply chain dataset (CSV) in the sidebar or use the synthetic dataset.
    """)
    
    # Load data: try reading an uploaded CSV; otherwise, use synthetic data.
    df = load_data()
    
    # Sidebar Navigation
    analysis_option = st.sidebar.selectbox(
        "Choose an Analysis Section",
        ["Overview", "Exploratory Data Analysis", "Demand Forecast", "Inventory Optimization", "Predictive Modeling", "Raw Data"]
    )
    
    if analysis_option == "Overview":
        st.header("Overview")
        st.write("Below is a sample of the supply chain data:")
        st.dataframe(df.head())
    
    elif analysis_option == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Distribution of Order Quantities")
        fig1 = plot_order_quantity_distribution(df)
        st.pyplot(fig1)
        
        st.subheader("Lead Time vs Shipping Time")
        fig2 = plot_lead_vs_shipping_time(df)
        st.pyplot(fig2)
        
        st.subheader("Correlation Matrix")
        fig3 = plot_correlation_heatmap(df)
        st.pyplot(fig3)
    
    elif analysis_option == "Demand Forecast":
        st.header("Demand Forecast")
        # Allow user to choose a product from the available list
        product_list = df['product_id'].unique()
        product_id = st.selectbox("Select Product ID", product_list, index=0)
        fig, forecast_head = forecast_demand_plot(df, product_id)
        if fig is not None:
            st.pyplot(fig)
            st.write("Forecast (first 5 days):")
            st.write(forecast_head)
    
    elif analysis_option == "Inventory Optimization":
        st.header("Inventory Optimization (EOQ)")
        eoq_df = optimize_inventory_df(df)
        st.dataframe(eoq_df)
    
    elif analysis_option == "Predictive Modeling":
        st.header("Predictive Modeling: Shipping Time Prediction")
        fig, mse, r2 = predictive_model_plot(df)
        st.pyplot(fig)
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R-squared:** {r2:.2f}")
    
    elif analysis_option == "Raw Data":
        st.header("Raw Data")
        st.dataframe(df)
    
if __name__ == '__main__':
    main()
