import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit UI
st.title('Bias-Variance Trade-off Demonstration')

# Load dataset
file_path = st.sidebar.file_uploader('Upload your dataset (Excel)', type='xlsx')
target_col = "Salary"  # Assuming "Salary" is the label column

# Model parameters
test_size = st.sidebar.slider('Test Size:', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
k_value = st.sidebar.slider('K for K-fold Cross Validation:', min_value=2, max_value=10, value=5)

if file_path is not None:
    df = pd.read_excel(file_path)

    # Splitting dataset into features (X) and target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate bias, variance, and error
    bias_train = np.mean((y_pred_train - y_train)**2)
    bias_test = np.mean((y_pred_test - y_test)**2)
    variance_train = np.mean((y_pred_train - np.mean(y_pred_train))**2)
    variance_test = np.mean((y_pred_test - np.mean(y_pred_test))**2)
    error_train = mean_squared_error(y_train, y_pred_train)
    error_test = mean_squared_error(y_test, y_pred_test)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_pred_train, label='Training Data', color='blue')
    ax.scatter(y_test, y_pred_test, label='Testing Data', color='red')
    ax.plot([min(y), max(y)], [min(y), max(y)], color='green', linestyle='--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('True vs. Predicted')
    ax.legend()
    st.pyplot(fig)

    # Displaying results in a table
    results = {
        'Metric': ['Bias', 'Variance', 'Error'],
        'Training': [bias_train, variance_train, error_train],
        'Testing': [bias_test, variance_test, error_test]
    }
    results_df = pd.DataFrame(results)
    st.write("## Training and Testing Metrics")
    st.dataframe(results_df)

    # K-fold cross-validation
    kf = KFold(n_splits=k_value)
    fold_errors = []
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        fold_error = mean_squared_error(y_test_fold, y_pred_fold)
        fold_errors.append(fold_error)

    # Display average K-fold error
    avg_fold_error = np.mean(fold_errors)
    st.write("## K-fold Cross Validation")
    st.write(f'Average K-fold Cross Validation Error: {avg_fold_error:.2f}')
