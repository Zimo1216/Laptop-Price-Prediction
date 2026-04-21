import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. Load Assets
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('laptop_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# ==========================================
# 2. Page Configuration
# ==========================================
st.set_page_config(page_title="Laptop Pricing Tool", layout="wide")
st.title("💻 AI-Driven Laptop Pricing Support System")
st.markdown("Professional decision support based on Hardware Architecture & Macroeconomic Inflation Index")

# Current Exchange Rate: 1 USD approx. 83.5 INR
EXCHANGE_RATE_INR_TO_USD = 83.5

# Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Core Specifications")
    
    # --- CPU Section ---
    cpu_brand = st.radio("Processor (CPU) Brand", ['Intel', 'AMD'])
    core_count = st.number_input("Processor Cores", min_value=2, max_value=24, value=8)
    ram_gb = st.selectbox("RAM Size (GB)", [4, 8, 16, 32, 64], index=2)
    storage_gb = st.selectbox("Storage Capacity (GB)", [256, 512, 1024, 2048], index=1)
    
    st.divider()
    
    # --- GPU Section (The Logic Fix) ---
    st.subheader("Graphics (GPU) Configuration")
    gpu_memory = st.selectbox("Dedicated Video Memory (GB)", [0, 2, 4, 6, 8, 12, 16], 
                             help="Select 0 for Integrated Graphics")
    
    if gpu_memory == 0:
        # Auto-match Integrated GPU brand to CPU brand
        gpu_brand = cpu_brand
        st.info(f"Integrated Graphics detected. Automatically using {gpu_brand} Graphics.")
    else:
        # Allow choosing Dedicated GPU brand
        gpu_brand = st.selectbox("Discrete GPU Brand", ['NVIDIA', 'AMD'])

    st.divider()
    
    # --- Others ---
    display_size = st.number_input("Display Size (Inches)", min_value=10.0, max_value=20.0, value=15.6)
    touch_screen = st.checkbox("Touch Screen Support")
    inflation_rate = st.slider("Global Inflation Index (%)", min_value=0.0, max_value=15.0, value=5.8)

with col2:
    st.header("2. Pricing Prediction")
    
    if st.button("Generate Suggested Price", type="primary"):
        # Build raw input dict
        input_dict = {
            'Core_per_processor': core_count,
            'RAM_GB': ram_gb,
            'Storage_capacity_GB': storage_gb,
            'Graphics_GB': gpu_memory,
            'Display_size_inches': display_size,
            'Touch_screen': bool(touch_screen),
            'World_Inflation_Index': float(inflation_rate) 
        }
        
        # One-hot encoding for Categorical Features
        input_dict[f'Processor_brand_{cpu_brand}'] = 1
        input_dict[f'Graphics_brand_{gpu_brand}'] = 1
        
        input_df = pd.DataFrame([input_dict])
        
        # 🌟 Feature Alignment with Scaler
        expected_cols = list(scaler.feature_names_in_)
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_cols]
        
        # Scaling and Prediction (Model output is INR)
        input_scaled = scaler.transform(input_df)
        predicted_price_inr = model.predict(input_scaled)[0]
        
        # Currency Conversion to USD
        predicted_price_usd = predicted_price_inr / EXCHANGE_RATE_INR_TO_USD
        
        # Display Result
        st.success(f"### Suggested Retail Price: ${predicted_price_usd:,.2f} USD")
        st.metric("Estimated Price (USD)", f"${predicted_price_usd:,.2f}")
        st.info(f"Base logic: Model processed 24 features including {gpu_brand} GPU ({gpu_memory}GB).")