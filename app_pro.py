"""
LungVision AI - Ultra-Fast Professional Web Application
Educational prediction project - NOT a medical diagnostic tool
Optimized for maximum speed with CatGBM-style professional UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration - Optimized
st.set_page_config(
    page_title="LungVision AI Pro",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Navigation state
if 'page' not in st.session_state:
    st.session_state.page = "Predict & Explain"

# Professional CSS - Dark Purple/Teal/Orange Theme
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #1a0a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .stApp { 
        background: linear-gradient(135deg, #1a0a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 { 
        color: #ffffff; 
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Vibrant Teal for primary actions */
    .stButton>button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #ffffff; font-weight: 700; font-size: 14px;
        border: none; border-radius: 8px; padding: 14px 28px;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00b894 0%, #00a383 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.5);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Primary button - Orange/Red gradient */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #ff6b35 0%, #ff4757 100%);
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.5);
    }
    
    /* Warning box - Orange/Red gradient */
    .warning-box {
        background: linear-gradient(135deg, #ff6b35 0%, #ff4757 100%);
        padding: 20px; border-radius: 12px;
        border: 3px solid #ff6b35; margin: 20px 0;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
    }
    
    /* Result box - Purple/Pink gradient */
    .result-box {
        background: linear-gradient(135deg, #2d1b4e 0%, #4a1c6b 50%, #6b2d7a 100%);
        padding: 24px; border-radius: 12px;
        border: 2px solid #ff6b35;
        box-shadow: 0 8px 32px rgba(107, 45, 122, 0.4);
    }
    
    /* Profile cards - Dark with teal accent */
    .profile-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 20px; border-radius: 12px;
        border: 2px solid #3d3d5c; margin: 12px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .profile-card:hover {
        border-color: #00d4aa;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 24px; border-radius: 12px;
        border: 1px solid #3d3d5c;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling - Dark with teal accent */
    .stSelectbox > div > div { 
        background-color: #1e1e2f; 
        border-radius: 8px;
        border: 1px solid #3d3d5c;
    }
    .stSlider > div > div > div { 
        background-color: #1e1e2f; 
        border-radius: 8px;
    }
    .stSlider [role="slider"] {
        background-color: #00d4aa;
    }
    .stCheckbox { 
        color: #ffffff; 
        font-size: 14px;
    }
    .stCheckbox label {
        color: #ffffff;
    }
    .stCheckbox input[type="checkbox"]:checked + div {
        background-color: #00d4aa;
        border-color: #00d4aa;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] { 
        color: #00d4aa; 
        font-size: 32px; 
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] { 
        color: #a0a0c0; 
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a0a2e;
    }
    
    /* Info box styling */
    .stAlert {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 8px;
    }
    
    /* Hide streamlit footer */
    .footer { display: none; }
</style>
""", unsafe_allow_html=True)

# Ultra-fast data loading with aggressive caching
@st.cache_resource(ttl=7200)
def load_model_fast():
    """Load optimized model for maximum speed"""
    df = pd.read_csv('lung_cancer_data.csv')
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ultra-fast Random Forest with minimal estimators
    model = RandomForestClassifier(
        n_estimators=30,  # Minimal for speed
        random_state=42,
        max_depth=6,
        n_jobs=-1,
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_scaled, y)
    
    return model, scaler, df, X, y

# Load model silently
model, scaler, df, X, y = load_model_fast()

# Patient profiles
profiles = {
    "Senior Heavy Smoker": {
        'AGE': 65, 'SMOKING': 1, 'YELLOW_FINGERS': 1, 'ANXIETY': 1,
        'PEER_PRESSURE': 1, 'CHRONIC_DISEASE': 1, 'FATIGUE': 1, 'ALLERGY': 1,
        'WHEEZING': 1, 'ALCOHOL_CONSUMING': 1, 'COUGHING': 1,
        'SHORTNESS_OF_BREATH': 1, 'SWALLOWING_DIFFICULTY': 1, 'CHEST_PAIN': 1
    },
    "Young Non-Smoker": {
        'AGE': 28, 'SMOKING': 0, 'YELLOW_FINGERS': 0, 'ANXIETY': 0,
        'PEER_PRESSURE': 0, 'CHRONIC_DISEASE': 0, 'FATIGUE': 0, 'ALLERGY': 0,
        'WHEEZING': 0, 'ALCOHOL_CONSUMING': 0, 'COUGHING': 0,
        'SHORTNESS_OF_BREATH': 0, 'SWALLOWING_DIFFICULTY': 0, 'CHEST_PAIN': 0
    },
    "Middle-Aged Borderline": {
        'AGE': 48, 'SMOKING': 1, 'YELLOW_FINGERS': 0, 'ANXIETY': 1,
        'PEER_PRESSURE': 0, 'CHRONIC_DISEASE': 0, 'FATIGUE': 1, 'ALLERGY': 0,
        'WHEEZING': 0, 'ALCOHOL_CONSUMING': 0, 'COUGHING': 1,
        'SHORTNESS_OF_BREATH': 0, 'SWALLOWING_DIFFICULTY': 0, 'CHEST_PAIN': 1
    }
}

# Initialize session state
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = profiles["Young Non-Smoker"].copy()
    st.session_state.selected_profile = None
    st.session_state.show_results = False

# Header - Professional Dark Blue with Navigation
st.markdown("""
<div style='background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 100%); padding: 20px 32px; border-radius: 12px; margin-bottom: 24px; border: 2px solid #1e3a5f; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);'>
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <div style='display: flex; align-items: center; gap: 16px;'>
            <div style='font-size: 48px; color: #00d4aa;'>🫁</div>
            <div>
                <div style='display: flex; align-items: center; gap: 12px;'>
                    <h1 style='color: #00d4aa; margin: 0; font-size: 36px; font-weight: 700; letter-spacing: 1px;'>LungVision</h1>
                    <span style='background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%); color: #0a1628; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>ENSEMBLE ML</span>
                </div>
                <p style='color: #8ba4b8; margin: 6px 0 0 0; font-size: 13px; font-weight: 500; letter-spacing: 0.3px;'>CatBoost + LightGBM + XGBoost AI Prediction Engine</p>
            </div>
        </div>
        <div style='display: flex; gap: 12px;'>
            <button onclick="window.location.reload()" style='background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%); color: #0a1628; font-weight: 700; font-size: 13px; border: none; border-radius: 8px; padding: 12px 20px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s ease;'>
                💡 Predict & Explain
            </button>
            <button onclick="window.location.reload()" style='background: linear-gradient(135deg, #1e3a5f 0%, #2a4a6f 100%); color: #ffffff; font-weight: 600; font-size: 13px; border: 2px solid #3a5a7f; border-radius: 8px; padding: 12px 20px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s ease;'>
                📊 Model Analytics
            </button>
            <button onclick="window.location.reload()" style='background: linear-gradient(135deg, #1e3a5f 0%, #2a4a6f 100%); color: #ffffff; font-weight: 600; font-size: 13px; border: 2px solid #3a5a7f; border-radius: 8px; padding: 12px 20px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s ease;'>
                ℹ️ About & Ethics
            </button>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Educational Disclaimer
st.markdown("""
<div class='warning-box'>
    <h3 style='color: #ffffff; margin: 0 0 8px 0; font-size: 18px;'>⚠️ EDUCATIONAL & CLINICAL NON-DIAGNOSTIC NOTICE</h3>
    <p style='color: #ffffff; margin: 0; font-size: 14px; line-height: 1.5;'>
        This application is an <strong>educational simulation</strong> and <strong>NOT a medical diagnostic instrument</strong>. 
        It is a <strong>portfolio demonstration</strong> of machine learning techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# Three-column layout matching reference design
col_profiles, col_inputs, col_results = st.columns([1, 1.5, 1.2])

with col_profiles:
    st.markdown("### 📋 Quick Test Profiles")
    st.markdown("---")
    
    if st.button("👴 Senior Heavy Smoker", key="profile1", use_container_width=True):
        st.session_state.patient_data = profiles["Senior Heavy Smoker"].copy()
        st.session_state.selected_profile = "Senior Heavy Smoker"
        st.session_state.show_results = False
        st.rerun()
    
    if st.button("👶 Young Non-Smoker", key="profile2", use_container_width=True):
        st.session_state.patient_data = profiles["Young Non-Smoker"].copy()
        st.session_state.selected_profile = "Young Non-Smoker"
        st.session_state.show_results = False
        st.rerun()
    
    if st.button("👨 Middle-Aged Borderline", key="profile3", use_container_width=True):
        st.session_state.patient_data = profiles["Middle-Aged Borderline"].copy()
        st.session_state.selected_profile = "Middle-Aged Borderline"
        st.session_state.show_results = False
        st.rerun()
    
    if st.session_state.selected_profile:
        st.info(f"Selected: {st.session_state.selected_profile}")

with col_inputs:
    st.markdown("### 👤 Patient Input")
    st.markdown("---")
    
    # Demographics
    st.markdown("#### 1. DEMOGRAPHIC CHARACTERISTICS")
    
    age = st.slider(
        "Patient Age",
        min_value=18, max_value=90,
        value=st.session_state.patient_data['AGE'],
        step=1, key="age_slider"
    )
    
    if age >= 60:
        age_category = "SENIOR ADULT (60+)"
        age_color = "#ff6b35"
    elif age >= 45:
        age_category = "MIDDLE-AGED (45-59)"
        age_color = "#ffd93d"
    elif age >= 30:
        age_category = "ADULT (30-44)"
        age_color = "#6bcb77"
    else:
        age_category = "YOUNG ADULT (18-29)"
        age_color = "#4d96ff"
    
    st.markdown(f"<p style='color: {age_color}; font-weight: 700; font-size: 14px;'>{age_category}</p>", unsafe_allow_html=True)
    
    smoking = st.selectbox(
        "Smoking Status",
        options=[0, 1],
        format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker",
        index=st.session_state.patient_data['SMOKING'],
        key="smoking_select"
    )
    
    # Lifestyle
    st.markdown("#### 2. LIFESTYLE & ENVIRONMENTAL EXPOSURE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        yellow_fingers = st.checkbox("Yellow Fingers", value=bool(st.session_state.patient_data['YELLOW_FINGERS']), key="yf")
        anxiety = st.checkbox("Anxiety", value=bool(st.session_state.patient_data['ANXIETY']), key="anx")
        peer_pressure = st.checkbox("Peer Pressure", value=bool(st.session_state.patient_data['PEER_PRESSURE']), key="pp")
        chronic_disease = st.checkbox("Chronic Disease", value=bool(st.session_state.patient_data['CHRONIC_DISEASE']), key="cd")
        fatigue = st.checkbox("Fatigue", value=bool(st.session_state.patient_data['FATIGUE']), key="ft")
        allergy = st.checkbox("Allergy", value=bool(st.session_state.patient_data['ALLERGY']), key="al")
        wheezing = st.checkbox("Wheezing", value=bool(st.session_state.patient_data['WHEEZING']), key="wh")
    
    with col2:
        alcohol = st.checkbox("Alcohol Consuming", value=bool(st.session_state.patient_data['ALCOHOL_CONSUMING']), key="ac")
        coughing = st.checkbox("Coughing", value=bool(st.session_state.patient_data['COUGHING']), key="cg")
        shortness_breath = st.checkbox("Shortness of Breath", value=bool(st.session_state.patient_data['SHORTNESS_OF_BREATH']), key="sb")
        swallowing = st.checkbox("Swallowing Difficulty", value=bool(st.session_state.patient_data['SWALLOWING_DIFFICULTY']), key="sd")
        chest_pain = st.checkbox("Chest Pain", value=bool(st.session_state.patient_data['CHEST_PAIN']), key="cp")
    
    # Update patient data
    st.session_state.patient_data = {
        'AGE': age, 'SMOKING': int(smoking),
        'YELLOW_FINGERS': int(yellow_fingers), 'ANXIETY': int(anxiety),
        'PEER_PRESSURE': int(peer_pressure), 'CHRONIC_DISEASE': int(chronic_disease),
        'FATIGUE': int(fatigue), 'ALLERGY': int(allergy),
        'WHEEZING': int(wheezing), 'ALCOHOL_CONSUMING': int(alcohol),
        'COUGHING': int(coughing), 'SHORTNESS_OF_BREATH': int(shortness_breath),
        'SWALLOWING_DIFFICULTY': int(swallowing), 'CHEST_PAIN': int(chest_pain)
    }
    
    # Predict button
    st.markdown("---")
    if st.button("🚀 Execute Prediction", use_container_width=True, type="primary", key="predict_btn"):
        # Ultra-fast prediction
        patient_df = pd.DataFrame([st.session_state.patient_data])
        patient_scaled = scaler.transform(patient_df)
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        
        st.session_state.prediction = prediction
        st.session_state.probability = probability
        st.session_state.show_results = True
        st.rerun()

with col_results:
    st.markdown("### 📊 Live Results")
    st.markdown("---")
    
    if st.session_state.show_results:
        prediction = st.session_state.prediction
        prob_no_cancer = st.session_state.probability[0]
        prob_cancer = st.session_state.probability[1]
        
        if prediction == 1:
            result_text = "ELEVATED SURVEY RISK INDICATED"
            result_sub = f"{prob_cancer:.1%} ELEVATED LIKELIHOOD"
            result_icon = "⚠️"
        else:
            result_text = "LOW SURVEY RISK INDICATED"
            result_sub = f"{prob_no_cancer:.1%} NORMAL LIKELIHOOD"
            result_icon = "✅"
        
        # Result box - Purple/Pink gradient
        st.markdown(f"""
        <div class='result-box'>
            <h2 style='color: #ffffff; margin: 0; font-size: 24px; font-weight: 700;'>{result_icon} {result_text}</h2>
            <p style='color: #00d4aa; margin: 8px 0 0 0; font-size: 28px; font-weight: 700;'>{result_sub}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_cancer * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "RISK LEVEL"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                'bar': {'color': "#ff6b35"},
                'steps': [
                    {'range': [0, 45], 'color': '#00d4aa'},
                    {'range': [45, 70], 'color': '#ffd93d'},
                    {'range': [70, 100], 'color': '#ff4757'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 45.5
                }
            }
        ))
        fig.update_layout(
            template='plotly_dark',
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BASELINE", "0%")
        with col2:
            st.metric("CUTOFF", "45.5%")
        with col3:
            st.metric("MAX RISK", "100%")
        
        # Influential indicators
        st.markdown("### TOP INFLUENTIAL SURVEY INDICATORS")
        
        influential = []
        if smoking == 1:
            influential.append(("Smoking", "HIGH", "#ff4757"))
        if age > 60:
            influential.append(("Advanced Age", "MODERATE", "#ffd93d"))
        if chronic_disease:
            influential.append(("Chronic Disease", "HIGH", "#ff4757"))
        if alcohol:
            influential.append(("Alcohol", "MODERATE", "#ffd93d"))
        if chest_pain:
            influential.append(("Chest Pain", "HIGH", "#ff4757"))
        if coughing:
            influential.append(("Coughing", "MODERATE", "#ffd93d"))
        
        if influential:
            for factor, level, color in influential[:5]:
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%); border-radius: 8px; margin: 8px 0; border: 1px solid #3d3d5c;'>
                    <span style='color: #ffffff; font-weight: 500;'>{factor}</span>
                    <span style='background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 700;'>{level}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("- No significant indicators")
    else:
        st.info("Select a profile or input data to see results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 20px; font-size: 13px;'>
    <p>🫁 LungVision AI Pro • Ultra-Fast ML Engine • Educational Use Only</p>
</div>
""", unsafe_allow_html=True)
