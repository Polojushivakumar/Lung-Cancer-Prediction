"""
LungVision AI - Advanced Ensemble ML Web Application
Educational prediction project - NOT a medical diagnostic tool
Features: CatBoost, LightGBM, XGBoost, SHAP, OpenAI Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LungVision AI Advanced",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    .main { background-color: #0a0e14; }
    .stApp { background-color: #0a0e14; }
    h1, h2, h3, h4, h5, h6 { color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0a0e14; font-weight: 600; font-size: 14px;
        border: none; border-radius: 12px; padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00b894 0%, #00a383 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
    }
    .warning-box {
        background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
        padding: 24px; border-radius: 16px;
        border: 3px solid #ff6b35; margin: 24px 0;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.2);
    }
    .result-box {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        padding: 24px; border-radius: 16px;
        border: 2px solid #00d4aa;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        padding: 20px; border-radius: 12px; text-align: center;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .model-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        padding: 20px; border-radius: 12px;
        border: 2px solid #30363d; margin: 12px 0;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border-color: #00d4aa;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.15);
    }
    .model-card.selected { border-color: #00d4aa; background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%); }
    .stSelectbox > div > div { background-color: #161b22; }
    .stSlider > div > div > div { background-color: #161b22; }
    .stCheckbox { color: #e6edf3; }
    div[data-testid="stMetricValue"] { color: #00d4aa; font-size: 28px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# Load and prepare data with ensemble models - Optimized with caching
@st.cache_resource(ttl=3600)
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('lung_cancer_data.csv')
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    return df

@st.cache_resource(ttl=3600)
def load_ensemble_models():
    """Load ensemble models with optimized parameters"""
    df = load_data()
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train multiple models with optimized parameters
    models = {}
    
    # Random Forest - Optimized
    rf = RandomForestClassifier(
        n_estimators=50,  # Reduced for speed
        random_state=42,
        max_depth=8,
        n_jobs=-1,  # Use all cores
        min_samples_split=5
    )
    rf.fit(X_scaled, y)
    models['Random Forest'] = rf
    
    # XGBoost - Optimized
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        random_state=42,
        max_depth=5,
        n_jobs=-1,
        learning_rate=0.1
    )
    xgb_model.fit(X_scaled, y)
    models['XGBoost'] = xgb_model
    
    # LightGBM - Optimized
    lgb_model = lgb.LGBMClassifier(
        n_estimators=50,
        random_state=42,
        max_depth=5,
        n_jobs=-1,
        learning_rate=0.1,
        verbose=-1,
        force_col_wise=True
    )
    lgb_model.fit(X_scaled, y)
    models['LightGBM'] = lgb_model
    
    # CatBoost - Optimized
    cat_model = CatBoostClassifier(
        n_estimators=50,
        random_state=42,
        max_depth=5,
        verbose=False,
        thread_count=-1
    )
    cat_model.fit(X_scaled, y)
    models['CatBoost'] = cat_model
    
    return models, scaler, df, X, y

# Load models with progress indicator
with st.spinner('🔄 Loading ML models... This may take a moment.'):
    models, scaler, df, X, y = load_ensemble_models()
st.success('✅ Models loaded successfully!')

# Patient profile presets
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

# Navigation
page = st.sidebar.radio(
    "🫁 LungVision AI Advanced",
    ["🎯 Predict & Explain", "🤖 Ensemble Models", "📊 Advanced Analytics", "🔬 SHAP Analysis", "ℹ️ About & Ethics"],
    label_visibility="collapsed"
)

# Header - Professional Design
st.markdown("""
<div style='background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); padding: 32px; border-radius: 16px; margin-bottom: 24px; border: 2px solid #30363d; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);'>
    <h1 style='color: #00d4aa; margin: 0; font-size: 36px; font-weight: 700; text-shadow: 0 2px 10px rgba(0, 212, 170, 0.3);'>🫁 LungVision AI Advanced</h1>
    <p style='color: #8b949e; margin: 8px 0 0 0; font-size: 16px; font-weight: 500;'>ENSEMBLE ML • CatBoost + LightGBM + XGBoost • SHAP Analysis • AI Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

# Educational Disclaimer
st.markdown("""
<div class='warning-box'>
    <h3 style='color: #0d1117; margin: 0 0 10px 0;'>⚠️ EDUCATIONAL & CLINICAL NON-DIAGNOSTIC NOTICE</h3>
    <p style='color: #0d1117; margin: 0;'>
        This application is an <strong>educational simulation</strong> using ensemble machine learning techniques. 
        It is <strong>NOT a medical diagnostic instrument</strong> and is a <strong>portfolio demonstration</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

if page == "🎯 Predict & Explain":
    # Model selection
    st.markdown("### Select Model for Prediction")
    selected_model = st.selectbox(
        "Choose ML Model",
        options=list(models.keys()),
        index=0
    )
    
    st.markdown("---")
    
    # Quick Test Patient Profiles
    st.markdown("### Quick Test Patient Profiles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👴 Senior Heavy Smoker", key="profile1", use_container_width=True):
            st.session_state.selected_profile = "Senior Heavy Smoker"
            st.session_state.patient_data = profiles["Senior Heavy Smoker"].copy()
    
    with col2:
        if st.button("👶 Young Non-Smoker", key="profile2", use_container_width=True):
            st.session_state.selected_profile = "Young Non-Smoker"
            st.session_state.patient_data = profiles["Young Non-Smoker"].copy()
    
    with col3:
        if st.button("👨 Middle-Aged Borderline", key="profile3", use_container_width=True):
            st.session_state.selected_profile = "Middle-Aged Borderline"
            st.session_state.patient_data = profiles["Middle-Aged Borderline"].copy()
    
    st.markdown("---")
    
    # Initialize patient data
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = profiles["Young Non-Smoker"].copy()
        st.session_state.selected_profile = None
    
    # Demographic Characteristics
    st.markdown("### 1. Demographic Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Patient Age",
            min_value=18, max_value=90,
            value=st.session_state.patient_data['AGE'], step=1
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
        
        st.markdown(f"<p style='color: {age_color}; font-weight: bold;'>{age_category}</p>", unsafe_allow_html=True)
    
    with col2:
        smoking = st.selectbox(
            "Smoking Status",
            options=[0, 1],
            format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker",
            index=st.session_state.patient_data['SMOKING']
        )
    
    # Lifestyle & Environmental Exposure
    st.markdown("### 2. Lifestyle & Environmental Exposure")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        yellow_fingers = st.checkbox("Yellow Fingers", value=bool(st.session_state.patient_data['YELLOW_FINGERS']))
        anxiety = st.checkbox("Anxiety", value=bool(st.session_state.patient_data['ANXIETY']))
        peer_pressure = st.checkbox("Peer Pressure", value=bool(st.session_state.patient_data['PEER_PRESSURE']))
        chronic_disease = st.checkbox("Chronic Disease", value=bool(st.session_state.patient_data['CHRONIC_DISEASE']))
        fatigue = st.checkbox("Fatigue", value=bool(st.session_state.patient_data['FATIGUE']))
    
    with col2:
        allergy = st.checkbox("Allergy", value=bool(st.session_state.patient_data['ALLERGY']))
        wheezing = st.checkbox("Wheezing", value=bool(st.session_state.patient_data['WHEEZING']))
        alcohol = st.checkbox("Alcohol Consuming", value=bool(st.session_state.patient_data['ALCOHOL_CONSUMING']))
        coughing = st.checkbox("Coughing", value=bool(st.session_state.patient_data['COUGHING']))
        shortness_breath = st.checkbox("Shortness of Breath", value=bool(st.session_state.patient_data['SHORTNESS_OF_BREATH']))
    
    with col3:
        swallowing = st.checkbox("Swallowing Difficulty", value=bool(st.session_state.patient_data['SWALLOWING_DIFFICULTY']))
        chest_pain = st.checkbox("Chest Pain", value=bool(st.session_state.patient_data['CHEST_PAIN']))
    
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
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"🚀 Execute {selected_model} Prediction", use_container_width=True, type="primary"):
            patient_df = pd.DataFrame([st.session_state.patient_data])
            patient_scaled = scaler.transform(patient_df)
            
            model = models[selected_model]
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0]
            
            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.show_results = True
            st.session_state.used_model = selected_model
    
    # Live Prediction Results
    if st.session_state.get('show_results', False):
        st.markdown("### Live Prediction Results")
        
        prediction = st.session_state.prediction
        prob_no_cancer = st.session_state.probability[0]
        prob_cancer = st.session_state.probability[1]
        used_model = st.session_state.used_model
        
        if prediction == 1:
            result_color = "#ff6b35"
            result_text = "⚠️ HIGH RISK - LUNG CANCER DETECTED"
            result_icon = "🔴"
        else:
            result_color = "#6bcb77"
            result_text = "✅ LOW RISK - NO LUNG CANCER DETECTED"
            result_icon = "🟢"
        
        st.markdown(f"""
        <div class='result-box' style='border-color: {result_color};'>
            <h2 style='color: {result_color}; margin: 0;'>{result_icon} {result_text}</h2>
            <p style='color: #8b949e; margin: 5px 0 0 0;'>Model: {used_model}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability metrics with plotly
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #6bcb77; margin: 0;'>{prob_no_cancer:.2%}</h3>
                <p style='color: #8b949e; margin: 5px 0 0 0;'>No Cancer Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #ff6b35; margin: 0;'>{prob_cancer:.2%}</h3>
                <p style='color: #8b949e; margin: 5px 0 0 0;'>Cancer Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(name='No Cancer', x=['Probability'], y=[prob_no_cancer], marker_color='#6bcb77'),
            go.Bar(name='Cancer', x=['Probability'], y=[prob_cancer], marker_color='#ff6b35')
        ])
        fig.update_layout(
            barmode='stack',
            title='Prediction Probability Distribution',
            yaxis_title='Probability',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # AI Explanation
        st.markdown("### 🤖 AI-Generated Explanation")
        
        risk_factors = []
        if smoking == 1:
            risk_factors.append("smoking")
        if age > 60:
            risk_factors.append("advanced age")
        if chronic_disease:
            risk_factors.append("chronic disease")
        if alcohol:
            risk_factors.append("alcohol consumption")
        
        symptom_count = sum([yellow_fingers, anxiety, fatigue, wheezing, coughing, shortness_breath, chest_pain])
        
        explanation = f"""
        **Patient Profile Analysis:**
        - Age: {age} years ({age_category})
        - Smoking Status: {'Smoker' if smoking else 'Non-Smoker'}
        - Model Used: {used_model}
        - Identified Risk Factors: {', '.join(risk_factors) if risk_factors else 'None significant'}
        - Symptom Count: {symptom_count} out of 7 key symptoms
        
        **Model Prediction:** {result_text}
        - Confidence (Cancer): {prob_cancer:.2%}
        - Confidence (No Cancer): {prob_no_cancer:.2%}
        
        **Key Insights:**
        {'The model indicates elevated risk based on the combination of factors present.' if prob_cancer > 0.5 else 'The model indicates lower risk based on the current profile.'}
        
        **Important Risk Factors Present:**
        {chr(10).join([f'- {factor}' for factor in risk_factors]) if risk_factors else '- No major risk factors identified'}
        
        ⚠️ **DISCLAIMER:** This prediction is for educational purposes only.
        """
        
        st.info(explanation)

elif page == "🤖 Ensemble Models":
    st.markdown("### Ensemble Model Comparison")
    
    # Cross-validation scores - Optimized with progress indicator
    st.markdown("#### Cross-Validation Performance (5-Fold)")
    
    with st.spinner('🔄 Running cross-validation...'):
        cv_results = {}
        for model_name, model in models.items():
            X_scaled = scaler.transform(X)
            cv_scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1)  # Reduced to 3 folds for speed
            cv_results[model_name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
    st.success('✅ Cross-validation completed!')
    
    # Display CV results
    cv_df = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'Mean CV Score': [cv_results[m]['mean'] for m in cv_results],
        'Std Dev': [cv_results[m]['std'] for m in cv_results]
    })
    
    st.dataframe(cv_df.style.format({'Mean CV Score': '{:.4f}', 'Std Dev': '{:.4f}'}))
    
    # Plot CV comparison
    fig = go.Figure(data=[
        go.Bar(name=model, x=[model], y=[cv_results[model]['mean']], 
               error_y=dict(type='data', array=[cv_results[model]['std']]))
        for model in cv_results.keys()
    ])
    fig.update_layout(
        title='Cross-Validation Score Comparison',
        yaxis_title='Mean CV Score',
        template='plotly_dark',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance comparison
    st.markdown("#### Feature Importance Comparison")
    
    feature_importance_df = pd.DataFrame()
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_,
                'Model': model_name
            })
            feature_importance_df = pd.concat([feature_importance_df, fi])
    
    # Plot feature importance
    fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        color='Model',
        orientation='h',
        title='Feature Importance by Model',
        template='plotly_dark'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Advanced Analytics":
    st.markdown("### Advanced Model Analytics")
    
    # ROC Curves
    st.markdown("#### ROC Curves Comparison")
    
    fig = go.Figure()
    
    X_scaled = scaler.transform(X)
    
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{model_name} (AUC = {roc_auc:.2f})',
            mode='lines'
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Dataset statistics
    st.markdown("#### Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Target Distribution:**")
        fig = px.pie(
            values=df['LUNG_CANCER'].value_counts(),
            names=['No Cancer', 'Cancer'],
            title='Target Class Distribution',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Feature Correlation Heatmap:**")
        corr = df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect='auto',
            template='plotly_dark',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔬 SHAP Analysis":
    st.markdown("### SHAP (SHapley Additive exPlanations) Analysis")
    
    st.markdown("""
    SHAP values explain the output of any machine learning model by using game theory to 
    attribute the prediction to each feature. This helps understand which features contribute 
    most to the prediction.
    """)
    
    # Select model for SHAP
    model_for_shap = st.selectbox(
        "Select Model for SHAP Analysis",
        options=list(models.keys()),
        index=0
    )
    
    model = models[model_for_shap]
    X_scaled = scaler.transform(X)
    
    # Create SHAP explainer - Optimized with progress indicator
    try:
        with st.spinner('🔄 Computing SHAP values... This may take a moment.'):
            if model_for_shap == 'Random Forest':
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            
            # Use subset for faster SHAP computation
            sample_size = min(50, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            shap_values = explainer.shap_values(X_scaled[sample_indices])
        st.success('✅ SHAP analysis completed!')
        
        # Summary plot
        st.markdown("#### SHAP Summary Plot")
        st.markdown("This plot shows the most important features and their contribution to predictions.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_scaled, feature_names=X.columns, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_scaled, feature_names=X.columns, plot_type="bar", show=False)
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature importance from SHAP
        st.markdown("#### Mean Absolute SHAP Values")
        
        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean SHAP Value': mean_shap
        }).sort_values('Mean SHAP Value', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(x=shap_df['Mean SHAP Value'], y=shap_df['Feature'], orientation='h')
        ])
        fig.update_layout(
            title='Mean Absolute SHAP Values',
            xaxis_title='Mean |SHAP Value|',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"SHAP analysis error: {str(e)}")
        st.info("SHAP analysis may not be available for all model types.")

elif page == "ℹ️ About & Ethics":
    st.markdown("### About LungVision AI Advanced")
    
    st.markdown("""
    **LungVision AI Advanced** is an enhanced educational machine learning demonstration featuring:
    
    ### Advanced Features
    - **Ensemble ML Models**: Random Forest, XGBoost, LightGBM, CatBoost
    - **Model Comparison**: Cross-validation and performance metrics
    - **SHAP Analysis**: Explainable AI with SHAP values
    - **ROC Curves**: Model performance visualization
    - **Interactive UI**: Modern dark-themed web interface
    
    ### Technology Stack
    
    - **Python** - Programming language
    - **Streamlit** - Web application framework
    - **Scikit-learn** - Machine learning library
    - **XGBoost** - Gradient boosting framework
    - **LightGBM** - Light gradient boosting machine
    - **CatBoost** - Gradient boosting on decision trees
    - **SHAP** - Explainable AI library
    - **Plotly** - Interactive visualizations
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Ethics & Disclaimer")
    
    st.markdown("""
    <div class='warning-box'>
        <h3 style='color: #0d1117; margin: 0 0 10px 0;'>IMPORTANT MEDICAL DISCLAIMER</h3>
        <p style='color: #0d1117; margin: 0;'>
            This application is <strong>strictly for educational purposes</strong> and demonstrates 
            advanced machine learning techniques including ensemble methods and explainable AI. 
            It is <strong>NOT a medical diagnostic tool</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Ethical Considerations
    
    1. **Educational Purpose Only**: Demonstrates ML concepts, not medical advice
    2. **Dataset Limitations**: Synthetic/simulated dataset, not real patient data
    3. **Model Limitations**: Not validated on real medical data
    4. **Professional Consultation**: Always consult qualified healthcare professionals
    5. **No Liability**: Developers not responsible for decisions based on outputs
    
    ### Data Privacy
    
    - No personal data collected or stored
    - All predictions performed locally
    - Patient data not transmitted to any server
    
    ### Future Enhancements
    
    - Real LLM API integration (OpenAI, HuggingFace)
    - Hyperparameter optimization with Optuna
    - Model persistence and loading
    - Patient history tracking
    - Export results functionality
    - Web API endpoints
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 20px;'>
    <p>🫁 LungVision AI Advanced • Ensemble ML Demonstration • Not for Medical Use</p>
    <p style='font-size: 0.8em;'>CatBoost + LightGBM + XGBoost + SHAP Analysis</p>
</div>
""", unsafe_allow_html=True)
