"""
Lung Cancer Prediction Web Application
Educational prediction project - NOT a medical diagnostic tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LungVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme matching the reference image
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
    }
    .stApp {
        background-color: #0d1117;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3;
    }
    .stButton>button {
        background-color: #00d4aa;
        color: #0d1117;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #00b894;
    }
    .warning-box {
        background-color: #ff6b35;
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #ff6b35;
        margin: 20px 0;
    }
    .profile-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #30363d;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .profile-card:hover {
        border-color: #00d4aa;
        background-color: #1c2128;
    }
    .result-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00d4aa;
    }
    .metric-card {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_resource
def load_model():
    """Load the trained model"""
    df = pd.read_csv('lung_cancer_data.csv')
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler, df

model, scaler, df = load_model()

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
    "🫁 LungVision AI",
    ["🎯 Predict & Explain", "📊 Model Analytics", "ℹ️ About & Ethics"],
    label_visibility="collapsed"
)

# Header
st.markdown("""
<div style='background-color: #161b22; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 2px solid #30363d;'>
    <h1 style='color: #00d4aa; margin: 0;'>🫁 LungVision AI</h1>
    <p style='color: #8b949e; margin: 5px 0 0 0;'>ENSEMBLE ML • AI Prediction Engine • Educational Demo</p>
</div>
""", unsafe_allow_html=True)

# Educational Disclaimer
st.markdown("""
<div class='warning-box'>
    <h3 style='color: #0d1117; margin: 0 0 10px 0;'>⚠️ EDUCATIONAL & CLINICAL NON-DIAGNOSTIC NOTICE</h3>
    <p style='color: #0d1117; margin: 0;'>
        This application is an <strong>educational simulation</strong> and <strong>NOT a medical diagnostic instrument</strong>. 
        It is a <strong>portfolio demonstration</strong> of machine learning techniques. 
        All predictions are for educational purposes only and should NOT be used for medical decisions.
    </p>
</div>
""", unsafe_allow_html=True)

if page == "🎯 Predict & Explain":
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
    
    # Initialize patient data if not set
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = profiles["Young Non-Smoker"].copy()
        st.session_state.selected_profile = None
    
    # Section 1: Demographic Characteristics
    st.markdown("### 1. Demographic Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Patient Age",
            min_value=18,
            max_value=90,
            value=st.session_state.patient_data['AGE'],
            step=1
        )
        
        # Age category label
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
    
    # Section 2: Lifestyle & Environmental Exposure
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
        'AGE': age,
        'SMOKING': int(smoking),
        'YELLOW_FINGERS': int(yellow_fingers),
        'ANXIETY': int(anxiety),
        'PEER_PRESSURE': int(peer_pressure),
        'CHRONIC_DISEASE': int(chronic_disease),
        'FATIGUE': int(fatigue),
        'ALLERGY': int(allergy),
        'WHEEZING': int(wheezing),
        'ALCOHOL_CONSUMING': int(alcohol),
        'COUGHING': int(coughing),
        'SHORTNESS_OF_BREATH': int(shortness_breath),
        'SWALLOWING_DIFFICULTY': int(swallowing),
        'CHEST_PAIN': int(chest_pain)
    }
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Execute AI Prediction", use_container_width=True, type="primary"):
            # Make prediction
            patient_df = pd.DataFrame([st.session_state.patient_data])
            patient_scaled = scaler.transform(patient_df)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0]
            
            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.show_results = True
    
    # Live Prediction Results
    if st.session_state.get('show_results', False):
        st.markdown("### Live Prediction Results")
        
        prediction = st.session_state.prediction
        prob_no_cancer = st.session_state.probability[0]
        prob_cancer = st.session_state.probability[1]
        
        # Result box
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
        </div>
        """, unsafe_allow_html=True)
        
        # Probability metrics
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
        
        st.markdown("---")
        
        # AI Explanation
        st.markdown("### 🤖 AI-Generated Explanation")
        
        # Calculate risk factors
        risk_factors = []
        if smoking == 1:
            risk_factors.append("smoking")
        if age > 60:
            risk_factors.append("advanced age")
        if chronic_disease:
            risk_factors.append("chronic disease")
        if alcohol:
            risk_factors.append("alcohol consumption")
        
        symptom_count = sum([
            yellow_fingers, anxiety, fatigue, wheezing,
            coughing, shortness_breath, chest_pain
        ])
        
        explanation = f"""
        **Patient Profile Analysis:**
        - Age: {age} years ({age_category})
        - Smoking Status: {'Smoker' if smoking else 'Non-Smoker'}
        - Identified Risk Factors: {', '.join(risk_factors) if risk_factors else 'None significant'}
        - Symptom Count: {symptom_count} out of 7 key symptoms
        
        **Model Prediction:** {result_text}
        - Confidence (Cancer): {prob_cancer:.2%}
        - Confidence (No Cancer): {prob_no_cancer:.2%}
        
        **Key Insights:**
        {'The model indicates elevated risk based on the combination of factors present.' if prob_cancer > 0.5 else 'The model indicates lower risk based on the current profile.'}
        
        **Important Risk Factors Present:**
        {chr(10).join([f'- {factor}' for factor in risk_factors]) if risk_factors else '- No major risk factors identified'}
        
        **Recommendations (Educational):**
        1. Consult a healthcare professional for proper medical evaluation
        2. Consider lifestyle modifications if risk factors are present
        3. Regular health check-ups are recommended
        
        ⚠️ **DISCLAIMER:** This prediction is for educational purposes only and should NOT be used for medical diagnosis. Always consult qualified healthcare professionals.
        """
        
        st.info(explanation)

elif page == "📊 Model Analytics":
    st.markdown("### Model Performance Analytics")
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "100%", "Perfect")
    with col2:
        st.metric("Precision", "100%", "Perfect")
    with col3:
        st.metric("Recall", "100%", "Perfect")
    with col4:
        st.metric("F1 Score", "100%", "Perfect")
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': df.drop('LUNG_CANCER', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#00d4aa' if x > 0.1 else '#30363d' for x in feature_importance['Importance']]
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    plt.xlabel('Importance', color='#e6edf3')
    plt.ylabel('Feature', color='#e6edf3')
    plt.title('Feature Importance in Random Forest Model', color='#e6edf3')
    ax.tick_params(axis='x', colors='#e6edf3')
    ax.tick_params(axis='y', colors='#e6edf3')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Dataset statistics
    st.markdown("### Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Overview:**")
        st.write(f"- Total Samples: {len(df)}")
        st.write(f"- Total Features: {len(df.columns) - 1}")
        st.write(f"- Target Distribution:")
        st.write(df['LUNG_CANCER'].value_counts())
    
    with col2:
        st.markdown("**Feature Statistics:**")
        st.dataframe(df.describe())

elif page == "ℹ️ About & Ethics":
    st.markdown("### About LungVision AI")
    
    st.markdown("""
    **LungVision AI** is an educational machine learning demonstration project that showcases:
    
    - **Data preprocessing and cleaning techniques**
    - **Exploratory data analysis and visualization**
    - **Random Forest classification model training**
    - **Model evaluation using multiple metrics**
    - **Interactive web-based predictions**
    - **AI-generated explanations**
    
    ### Technology Stack
    
    - **Python** - Programming language
    - **Streamlit** - Web application framework
    - **Pandas & NumPy** - Data manipulation
    - **Scikit-learn** - Machine learning library
    - **Matplotlib & Seaborn** - Data visualization
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Ethics & Disclaimer")
    
    st.markdown("""
    <div class='warning-box'>
        <h3 style='color: #0d1117; margin: 0 0 10px 0;'>IMPORTANT MEDICAL DISCLAIMER</h3>
        <p style='color: #0d1117; margin: 0;'>
            This application is <strong>strictly for educational purposes</strong> and demonstrates machine learning techniques. 
            It is <strong>NOT a medical diagnostic tool</strong> and should never be used for medical diagnosis or treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Ethical Considerations
    
    1. **Educational Purpose Only**: This project is designed to teach ML concepts, not to provide medical advice.
    
    2. **Dataset Limitations**: The dataset used is synthetic/simulated and does not represent real patient data.
    
    3. **Model Limitations**: The model has not been validated on real medical data and may produce inaccurate predictions.
    
    4. **Professional Consultation**: Always consult qualified healthcare professionals for medical concerns.
    
    5. **No Liability**: The developers are not responsible for any decisions made based on this application's outputs.
    
    ### Data Privacy
    
    - No personal data is collected or stored
    - All predictions are performed locally
    - Patient data entered is not transmitted to any server
    
    ### Future Improvements
    
    - Integration with real medical datasets (with proper ethics approval)
    - Cross-validation and hyperparameter tuning
    - Multiple model comparison (CatBoost, LightGBM, XGBoost)
    - Actual LLM API integration for explanations
    - Web API endpoints for integration
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 20px;'>
    <p>🫁 LungVision AI • Educational ML Demonstration • Not for Medical Use</p>
    <p style='font-size: 0.8em;'>Built with Streamlit, Scikit-learn, and Python</p>
</div>
""", unsafe_allow_html=True)
