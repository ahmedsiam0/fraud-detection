import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import streamlit as st

def preprocess_data(df):
    # Fill null values with mean in 'is_fraud' and 'Amount' columns
    df['is_fraud'].fillna(df['is_fraud'].mean(), inplace=True)
    df['Amount'].fillna(df['Amount'].mean(), inplace=True)
    
    # Drop rows with null values in 'firstName', 'lastName', 'trans_num' columns
    df.dropna(subset=['firstName', 'lastName', 'trans_num'], inplace=True)
    
    # Convert 'Time' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Label encode 'category' and 'merchant' columns
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    df['merchant_encoded'] = label_encoder.fit_transform(df['merchant'])
    
    # Scale the 'Amount' column
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Create new feature 'Hour'
    df['Hour'] = df['Time'].dt.hour
    
    # Create 'Cardholder Name' feature and calculate average amount by cardholder
    df['Cardholder Name'] = df['firstName'] + ' ' + df['lastName']
    df['Average Amount By Cardholder'] = df.groupby('Cardholder Name')['Amount'].transform('mean')
    df['Amount Diff From Average'] = df['Amount'] - df['Average Amount By Cardholder']
    
    # Sort DataFrame by 'Time'
    df.sort_values('Time', inplace=True)
    
    # Drop unneeded columns
    df.drop(['ID', 'Card Number', 'trans_num', 'firstName', 'lastName', 'merchant', 'Cardholder Name', 'trans_num', 'category'], axis=1, inplace=True)
    
    return df

# Function to preprocess a single transaction
def preprocess_transaction(transaction):
    # Create DataFrame from the user input (single transaction)
    df = pd.DataFrame([transaction])
    
    # Fill null values with mean in 'Amount' columns
    df['Amount'].fillna(df['Amount'].mean(), inplace=True)
    
    # Convert 'Time' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Label encode 'category' and 'merchant' columns
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    df['merchant_encoded'] = label_encoder.fit_transform(df['merchant'])
    
    # Scale the 'Amount' column
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Create new feature 'Hour'
    df['Hour'] = df['Time'].dt.hour
    
    # Create 'Cardholder Name' feature and calculate average amount by cardholder
    df['Cardholder Name'] = df['firstName'] + ' ' + df['lastName']
    df['Average Amount By Cardholder'] = df.groupby('Cardholder Name')['Amount'].transform('mean')
    df['Amount Diff From Average'] = df['Amount'] - df['Average Amount By Cardholder']
    
    # Drop unneeded columns
    df.drop(['ID', 'Time', 'Card Number', 'trans_num', 'firstName', 'lastName', 'merchant', 'Cardholder Name', 'trans_num', 'category'], axis=1, inplace=True)
    
    return df

# Main function to build and run the Streamlit app
def main():
    st.title("Fraud Detection with XGBoost")

    # Read csv file
    df = pd.read_csv("fraudTrain.csv")

    # Preprocess the data
    processed_df = preprocess_data(df.copy())

    # Display preprocessed DataFrame
    st.subheader("Preprocessed Data")
    st.write(processed_df.head())

    # Split data into X (features) and y (target)
    X = processed_df.drop(['is_fraud', 'Time'], axis=1)
    y = processed_df['is_fraud']

    # Train XGBoost classifier
    XGBClassifier_model = XGBClassifier()
    XGBClassifier_model.fit(X, y)

    # Show model accuracy
    accuracy = XGBClassifier_model.score(X, y)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.4f}")

    # Show feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': XGBClassifier_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    st.write(feature_importance)

    # Prediction section
    st.subheader("Predict Fraud")

    # Create form to input new transaction
    with st.form("input_form"):
        st.subheader("Enter Transaction Details")
        transaction = {}
        for col_name in ['ID', 'Time', 'Card Number', 'merchant', 'category', 'Amount', 'firstName', 'lastName', 'trans_num']:
            if col_name != 'is_fraud':  # Exclude ID and is_fraud columns
                if col_name == 'Amount':
                    transaction[col_name] = st.number_input(col_name, value=0.0)
                else:
                    transaction[col_name] = st.text_input(col_name)
        
        submitted = st.form_submit_button("Predict")

        print(transaction)
    

        # Preprocess the new transaction
        processed_transaction = preprocess_transaction(transaction)
        
        # Perform prediction
        prediction = XGBClassifier_model.predict(processed_transaction)
        
        # Display prediction result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Fraudulent transaction detected!")
        else:
            st.success("Non-fraudulent transaction.")
        
if __name__ == '__main__':
    main()
