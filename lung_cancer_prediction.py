"""
Lung Cancer Prediction Project
Educational prediction project - NOT a medical diagnostic tool
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LungCancerPredictor:
    def __init__(self, data_path='lung_cancer_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        print("=" * 60)
        print("STEP 1: LOADING AND CLEANING DATA")
        print("=" * 60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Check data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Convert target variable to binary
        self.df['LUNG_CANCER'] = self.df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        print("\nData cleaning completed!")
        print(f"Target variable distribution:")
        print(self.df['LUNG_CANCER'].value_counts())
        
        return self.df
    
    def perform_data_analysis(self):
        """Perform data analysis using Pandas & NumPy"""
        print("\n" + "=" * 60)
        print("STEP 2: DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Correlation matrix
        print("\nCorrelation Matrix:")
        correlation = self.df.corr()
        print(correlation)
        
        # Feature statistics by target
        print("\nMean values by Lung Cancer Status:")
        print(self.df.groupby('LUNG_CANCER').mean())
        
        # Age distribution analysis
        print("\nAge Statistics:")
        print(f"Mean Age: {self.df['AGE'].mean():.2f}")
        print(f"Median Age: {self.df['AGE'].median():.2f}")
        print(f"Std Age: {self.df['AGE'].std():.2f}")
        
        # Smoking analysis
        print("\nSmoking vs Lung Cancer:")
        smoking_crosstab = pd.crosstab(self.df['SMOKING'], self.df['LUNG_CANCER'])
        print(smoking_crosstab)
        
        return correlation
    
    def create_visualizations(self):
        """Create data visualizations"""
        print("\n" + "=" * 60)
        print("STEP 3: CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Target distribution
        ax1 = axes[0, 0]
        self.df['LUNG_CANCER'].value_counts().plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'])
        ax1.set_title('Lung Cancer Distribution')
        ax1.set_xlabel('Lung Cancer (0=No, 1=Yes)')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['No', 'Yes'], rotation=0)
        
        # 2. Age distribution
        ax2 = axes[0, 1]
        self.df[self.df['LUNG_CANCER'] == 0]['AGE'].plot(kind='hist', alpha=0.5, label='No Cancer', ax=ax2, color='#3498db')
        self.df[self.df['LUNG_CANCER'] == 1]['AGE'].plot(kind='hist', alpha=0.5, label='Cancer', ax=ax2, color='#e74c3c')
        ax2.set_title('Age Distribution by Cancer Status')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Smoking vs Cancer
        ax3 = axes[0, 2]
        smoking_crosstab = pd.crosstab(self.df['SMOKING'], self.df['LUNG_CANCER'])
        smoking_crosstab.plot(kind='bar', ax=ax3, color=['#3498db', '#e74c3c'])
        ax3.set_title('Smoking vs Lung Cancer')
        ax3.set_xlabel('Smoking (0=No, 1=Yes)')
        ax3.set_ylabel('Count')
        ax3.legend(['No Cancer', 'Cancer'])
        ax3.set_xticklabels(['Non-Smoker', 'Smoker'], rotation=0)
        
        # 4. Correlation heatmap
        ax4 = axes[1, 0]
        correlation = self.df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax4, fmt='.2f')
        ax4.set_title('Correlation Heatmap')
        
        # 5. Feature importance preview
        ax5 = axes[1, 1]
        feature_means = self.df.drop('LUNG_CANCER', axis=1).mean()
        feature_means.sort_values(ascending=True).plot(kind='barh', ax=ax5, color='#2ecc71')
        ax5.set_title('Feature Means')
        ax5.set_xlabel('Mean Value')
        
        # 6. Symptoms count
        ax6 = axes[1, 2]
        symptom_cols = ['YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 
                       'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 
                       'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
        self.df['symptom_count'] = self.df[symptom_cols].sum(axis=1)
        self.df.groupby('LUNG_CANCER')['symptom_count'].mean().plot(kind='bar', ax=ax6, color=['#3498db', '#e74c3c'])
        ax6.set_title('Average Symptom Count by Cancer Status')
        ax6.set_xlabel('Lung Cancer (0=No, 1=Yes)')
        ax6.set_ylabel('Average Symptom Count')
        ax6.set_xticklabels(['No', 'Yes'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('lung_cancer_visualizations.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'lung_cancer_visualizations.png'")
        plt.close()
        
        return fig
    
    def split_data(self):
        """Split data into training and testing sets"""
        print("\n" + "=" * 60)
        print("STEP 4: SPLITTING DATA")
        print("=" * 60)
        
        # Prepare features and target
        X = self.df.drop(['LUNG_CANCER', 'symptom_count'], axis=1)
        y = self.df['LUNG_CANCER']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"Training target distribution:\n{self.y_train.value_counts()}")
        print(f"Testing target distribution:\n{self.y_test.value_counts()}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data scaling completed!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the classification model"""
        print("\n" + "=" * 60)
        print("STEP 5: TRAINING MODEL")
        print("=" * 60)
        
        # Initialize Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("Random Forest Classifier trained successfully!")
        print(f"Model parameters: {self.model.get_params()}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.model, feature_importance
    
    def evaluate_model(self):
        """Evaluate the model using various metrics"""
        print("\n" + "=" * 60)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Cancer', 'Cancer']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict_new_data(self, new_data):
        """Predict lung cancer for new input data"""
        print("\n" + "=" * 60)
        print("STEP 7: PREDICTION FOR NEW DATA")
        print("=" * 60)
        
        # Convert to DataFrame if needed
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
        
        # Scale the data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make prediction
        prediction = self.model.predict(new_data_scaled)
        probability = self.model.predict_proba(new_data_scaled)
        
        results = []
        for i in range(len(new_data)):
            result = {
                'prediction': 'Lung Cancer Detected' if prediction[i] == 1 else 'No Lung Cancer',
                'probability_no_cancer': probability[i][0],
                'probability_cancer': probability[i][1]
            }
            results.append(result)
            print(f"\nSample {i+1}:")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability (No Cancer): {result['probability_no_cancer']:.4f}")
            print(f"Probability (Cancer): {result['probability_cancer']:.4f}")
        
        return results
    
    def generate_llm_explanation(self, patient_data, prediction_result):
        """Generate LLM-based explanation for prediction"""
        print("\n" + "=" * 60)
        print("STEP 8: LLM EXPLANATION GENERATION")
        print("=" * 60)
        
        # This is a simulated LLM explanation
        # In a real implementation, you would use an actual LLM API like OpenAI, HuggingFace, etc.
        
        risk_factors = []
        if patient_data.get('SMOKING', 0) == 1:
            risk_factors.append("smoking")
        if patient_data.get('AGE', 0) > 60:
            risk_factors.append("advanced age")
        if patient_data.get('CHRONIC_DISEASE', 0) == 1:
            risk_factors.append("chronic disease")
        if patient_data.get('ALCOHOL_CONSUMING', 0) == 1:
            risk_factors.append("alcohol consumption")
        
        symptom_count = sum([
            patient_data.get('YELLOW_FINGERS', 0),
            patient_data.get('ANXIETY', 0),
            patient_data.get('FATIGUE', 0),
            patient_data.get('WHEEZING', 0),
            patient_data.get('COUGHING', 0),
            patient_data.get('SHORTNESS_OF_BREATH', 0),
            patient_data.get('CHEST_PAIN', 0)
        ])
        
        explanation = f"""
=== AI-Generated Explanation (Educational Purpose Only) ===

Patient Profile Analysis:
- Age: {patient_data.get('AGE', 'N/A')} years
- Smoking Status: {'Yes' if patient_data.get('SMOKING', 0) == 1 else 'No'}
- Identified Risk Factors: {', '.join(risk_factors) if risk_factors else 'None significant'}
- Symptom Count: {symptom_count} out of 7 key symptoms

Model Prediction: {prediction_result['prediction']}
- Confidence (Cancer): {prediction_result['probability_cancer']:.2%}
- Confidence (No Cancer): {prediction_result['probability_no_cancer']:.2%}

Key Insights:
{'The model indicates elevated risk based on the combination of factors present.' if prediction_result['probability_cancer'] > 0.5 else 'The model indicates lower risk based on the current profile.'}

Important Risk Factors Present:
{chr(10).join([f'- {factor}' for factor in risk_factors]) if risk_factors else '- No major risk factors identified'}

Recommendations (Educational):
1. Consult a healthcare professional for proper medical evaluation
2. Consider lifestyle modifications if risk factors are present
3. Regular health check-ups are recommended
4. This is an educational prediction and NOT a medical diagnosis

⚠️ DISCLAIMER: This prediction is for educational purposes only and should NOT be used for medical diagnosis. Always consult qualified healthcare professionals.
"""
        
        print(explanation)
        return explanation
    
    def run_complete_pipeline(self):
        """Run the complete prediction pipeline"""
        print("\n" + "=" * 60)
        print("LUNG CANCER PREDICTION - COMPLETE PIPELINE")
        print("=" * 60)
        print("⚠️  EDUCATIONAL PURPOSE ONLY - NOT A MEDICAL DIAGNOSTIC TOOL")
        print("=" * 60)
        
        # Step 1: Load and clean data
        self.load_and_clean_data()
        
        # Step 2: Data analysis
        self.perform_data_analysis()
        
        # Step 3: Visualizations
        self.create_visualizations()
        
        # Step 4: Split data
        self.split_data()
        
        # Step 5: Train model
        self.train_model()
        
        # Step 6: Evaluate model
        metrics = self.evaluate_model()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return metrics


def main():
    """Main function to run the lung cancer prediction project"""
    
    # Initialize predictor
    predictor = LungCancerPredictor()
    
    # Run complete pipeline
    metrics = predictor.run_complete_pipeline()
    
    # Example prediction with new data
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION WITH NEW DATA")
    print("=" * 60)
    
    # Test case 1: High risk patient
    patient_1 = {
        'AGE': 65,
        'SMOKING': 1,
        'YELLOW_FINGERS': 1,
        'ANXIETY': 1,
        'PEER_PRESSURE': 1,
        'CHRONIC_DISEASE': 1,
        'FATIGUE': 1,
        'ALLERGY': 1,
        'WHEEZING': 1,
        'ALCOHOL_CONSUMING': 1,
        'COUGHING': 1,
        'SHORTNESS_OF_BREATH': 1,
        'SWALLOWING_DIFFICULTY': 1,
        'CHEST_PAIN': 1
    }
    
    # Test case 2: Low risk patient
    patient_2 = {
        'AGE': 35,
        'SMOKING': 0,
        'YELLOW_FINGERS': 0,
        'ANXIETY': 0,
        'PEER_PRESSURE': 0,
        'CHRONIC_DISEASE': 0,
        'FATIGUE': 0,
        'ALLERGY': 0,
        'WHEEZING': 0,
        'ALCOHOL_CONSUMING': 0,
        'COUGHING': 0,
        'SHORTNESS_OF_BREATH': 0,
        'SWALLOWING_DIFFICULTY': 0,
        'CHEST_PAIN': 0
    }
    
    # Predict for both patients
    results = predictor.predict_new_data([patient_1, patient_2])
    
    # Generate LLM explanations
    print("\n" + "-" * 60)
    print("Patient 1 - High Risk Profile:")
    print("-" * 60)
    predictor.generate_llm_explanation(patient_1, results[0])
    
    print("\n" + "-" * 60)
    print("Patient 2 - Low Risk Profile:")
    print("-" * 60)
    predictor.generate_llm_explanation(patient_2, results[1])
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated Files:")
    print("- lung_cancer_visualizations.png")
    print("- confusion_matrix.png")
    print("\nModel Performance Summary:")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")
    print(f"- F1 Score: {metrics['f1']:.4f}")
    print("\n⚠️  REMINDER: This is an educational project, not a medical diagnostic tool!")


if __name__ == "__main__":
    main()
