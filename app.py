import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title
st.title("Breast Cancer Classification App")

# Data loading
st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Drop ID column if present
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)
    
    # Drop unnamed columns (if any)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    st.write("First 5 rows of data:")
    st.dataframe(data.head())
    
    # EDA - Descriptive Statistics
    st.header("2. Exploratory Data Analysis")
    
    # Show descriptive statistics
    st.subheader("2.1 Descriptive Statistics")
    st.write(data.describe())
    
    # Scatter Plot
    st.subheader("2.2 Scatter Plot: Radius vs Texture")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['radius_mean'], y=data['texture_mean'], 
                   hue=data['diagnosis'], ax=ax, palette='viridis')
    plt.title("Scatter Plot: Radius vs Texture")
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    st.pyplot(fig)
    
    # Modeling Section
    st.header("3. Model Training and Evaluation")
    
    # Data preprocessing
    le = LabelEncoder()
    data['diagnosis'] = le.fit_transform(data['diagnosis'])  # M=1, B=0
    
    # Features & Target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Scaling numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Selection
    st.subheader("3.1 Select Model")
    model_choice = st.radio(
        "Choose a model:",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )
    
    # Model Training
    if st.button("Train Model"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        # Model selection and training
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)
        # Train model
        model.fit(X_train, y_train)
        # Store model and scaler in session state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['X_columns'] = X.columns.tolist()
        # Make predictions
        y_pred = model.predict(X_test)
        # Display results
        st.subheader("3.2 Model Performance")
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2%}")
        # Classification Report
        st.write("Detailed Performance Metrics:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        # Confusion Matrix
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=['Actual Benign', 'Actual Malignant'],
            columns=['Predicted Benign', 'Predicted Malignant']
        )
        st.dataframe(cm_df)
        # Feature Importance (for Random Forest)
        if model_choice == "Random Forest":
            st.subheader("3.3 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
            plt.title("Top 10 Most Important Features")
            st.pyplot(fig)
            
    # Add prediction for new data
    st.header("4. Make New Predictions")
    st.write("Enter values for a new patient:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=15.0)
        texture = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=20.0)
        perimeter = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=100.0)
        area = st.number_input("Area Mean", min_value=0.0, max_value=3000.0, value=800.0)
        
    with col2:
        smoothness = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1)
        compactness = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.1)
        concavity = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.1)
        symmetry = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2)
        
    if st.button("Predict"):
        # Check if model and scaler are available
        if 'model' in st.session_state and 'scaler' in st.session_state and 'X_columns' in st.session_state:
            # Create a dataframe with the input values
            input_dict = {
                'radius_mean': radius,
                'texture_mean': texture,
                'perimeter_mean': perimeter,
                'area_mean': area,
                'smoothness_mean': smoothness,
                'compactness_mean': compactness,
                'concavity_mean': concavity,
                'symmetry_mean': symmetry
            }
            # Fill missing columns with zeros if needed
            for col in st.session_state['X_columns']:
                if col not in input_dict:
                    input_dict[col] = 0.0
            new_data = pd.DataFrame({col: [input_dict[col]] for col in st.session_state['X_columns']})
            # Scale the new data
            new_data_scaled = st.session_state['scaler'].transform(new_data)
            # Make prediction
            prediction = st.session_state['model'].predict(new_data_scaled)
            probability = st.session_state['model'].predict_proba(new_data_scaled)
            # Display result
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error("Prediction: Malignant")
            else:
                st.success("Prediction: Benign")
            st.write(f"Probability of being benign: {probability[0][0]:.2%}")
            st.write(f"Probability of being malignant: {probability[0][1]:.2%}")
        else:
            st.error("Please train a model first!")
