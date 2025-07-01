import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition Model",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class=\"main-header\">🔍 Facial Recognition Model</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Train Model", "Make Predictions", "Model Performance"])

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None

def load_data():
    """Load the facial recognition dataset from facerec.csv"""
    try:
        df = pd.read_csv("facerec.csv")
        return df
    except FileNotFoundError:
        st.error("Error: facerec.csv not found. Please make sure it's in the same directory as streamlit_app.py")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(df):
    """Train the Random Forest model"""
    try:
        # Split features and labels
        features = df.drop(columns=['Label']).values
        labels = df['Label'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Balance training data using SMOTE
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=42
        )
        model.fit(X_train_scaled, y_train_bal)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        return model, scaler, accuracy, y_test, predictions
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

def make_prediction(model, scaler, features):
    """Make prediction on new data"""
    try:
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Home Page
if page == "Home":
    st.markdown("<h2 class=\"sub-header\">Welcome to the Facial Recognition Model</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 About This Application
        
        This Streamlit application demonstrates a facial recognition model using machine learning.
        The model uses a Random Forest classifier to distinguish between recognized and unrecognized faces.
        
        **Features:**
        - 🤖 Train Random Forest model
        - 📊 View model performance metrics
        - 🔮 Make predictions on new data
        - 📈 Visualize results
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 How to Use
        
        1. **Train Model**: Upload your dataset and train the model
        2. **Make Predictions**: Input features to get predictions
        3. **View Performance**: Check accuracy and other metrics
        
        **Dataset Options:**
        - The `facerec.csv` file is included and will be used automatically.
        """)
    
    # Model status
    if st.session_state.model_trained:
        st.success("✅ Model is trained and ready for predictions!")
        st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
    else:
        st.warning("⚠️ Model not trained yet. Please go to 'Train Model' page.")

# Train Model Page
elif page == "Train Model":
    st.markdown("<h2 class=\"sub-header\">🤖 Train the Facial Recognition Model</h2>", unsafe_allow_html=True)
    
    # Data loading section
    st.subheader("📥 Load Dataset")
    
    df = load_data()
    
    if df is not None:
        # Display dataset info
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Classes", df['Label'].nunique())
        
        # Show data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        
        # Class distribution
        st.subheader("📈 Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        df['Label'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Labels')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Train model button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few moments."):
                model, scaler, accuracy, y_test, predictions = train_model(df)
                
                if model is not None:
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.accuracy = accuracy
                    st.session_state.model_trained = True
                    st.session_state.y_test = y_test
                    st.session_state.predictions = predictions
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
                    
                    # Show classification report
                    st.subheader("📋 Classification Report")
                    from sklearn.metrics import classification_report
                    report = classification_report(y_test, predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)

# Make Predictions Page
elif page == "Make Predictions":
    st.markdown("<h2 class=\"sub-header\">🔮 Make Predictions</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
    else:
        st.success("✅ Model is ready for predictions!")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
        
        if input_method == "Manual Input":
            st.subheader("📝 Enter Feature Values")
            
            # Get number of features from the trained model
            n_features = st.session_state.model.n_features_in_
            
            # Create input fields for features
            features = []
            cols = st.columns(5)  # 5 columns for better layout
            
            for i in range(n_features):
                with cols[i % 5]:
                    feature_value = st.number_input(
                        f"Feature {i+1}",
                        value=0.0,
                        step=0.01,
                        key=f"feature_{i}"
                    )
                    features.append(feature_value)
            
            if st.button("🔍 Make Prediction", type="primary"):
                features_array = np.array(features)
                prediction, probability = make_prediction(
                    st.session_state.model,
                    st.session_state.scaler,
                    features_array
                )
                
                if prediction is not None:
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.success("✅ **RECOGNIZED**")
                        else:
                            st.error("❌ **NOT RECOGNIZED**")
                    
                    with col2:
                        st.metric("Confidence", f"{max(probability):.2%}")
                    
                    # Show probability distribution
                    st.subheader("📊 Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': ['Not Recognized', 'Recognized'],
                        'Probability': probability
                    })
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(prob_df['Class'], prob_df['Probability'])
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    
                    # Color bars based on prediction
                    bars[prediction].set_color('green')
                    bars[1-prediction].set_color('red')
                    
                    st.pyplot(fig)
        
        elif input_method == "Upload CSV":
            st.subheader("📁 Upload CSV for Batch Predictions")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    test_df = pd.read_csv(uploaded_file)
                    st.dataframe(test_df.head())
                    
                    if st.button("🔍 Make Batch Predictions"):
                        # Make predictions
                        features_scaled = st.session_state.scaler.transform(test_df.values)
                        predictions = st.session_state.model.predict(features_scaled)
                        probabilities = st.session_state.model.predict_proba(features_scaled)
                        
                        # Create results dataframe
                        results_df = test_df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Confidence'] = np.max(probabilities, axis=1)
                        
                        st.subheader("📊 Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

# Model Performance Page
elif page == "Model Performance":
    st.markdown("<h2 class=\"sub-header\">📈 Model Performance</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
    else:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
        
        with col2:
            st.metric("Model Type", "Random Forest")
        
        with col3:
            st.metric("Features", st.session_state.model.n_features_in_)
        
        # Confusion Matrix
        if 'y_test' in st.session_state and 'predictions' in st.session_state:
            st.subheader("🔄 Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            
            feature_importance = st.session_state.model.feature_importances_
            top_features = 20  # Show top 20 features
            
            indices = np.argsort(feature_importance)[::-1][:top_features]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(top_features), feature_importance[indices])
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
            ax.set_title(f'Top {top_features} Feature Importances')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Facial Recognition Model** - Built with Streamlit 🚀")

