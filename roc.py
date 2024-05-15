import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score

# # Function to generate sample data
# def generate_data(num_rows):
#     X, y = make_classification(n_samples=num_rows, n_features=2, n_classes=2, random_state=42)
#     return X, y

# Function to generate sample data
def generate_data(num_rows):
    X, y = make_classification(n_samples=num_rows, n_features=2, n_classes=2, random_state=42,
                               n_clusters_per_class=1, flip_y=0.1, class_sep=1.0,
                               n_informative=2, n_redundant=0, n_repeated=0)
    return X, y


# Function to train logistic regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, accuracy




# Function to calculate confusion matrix, ROC curve, and metrics
def evaluate_model(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    return cm, fpr, tpr, roc_auc, f1

# Streamlit UI
st.title('Logistic Regression Evaluation')

# User input for number of rows and threshold
num_rows = st.sidebar.slider('Number of Rows', min_value=100, max_value=1000, value=500, step=100)
threshold = st.sidebar.slider('Threshold', min_value=0.1, max_value=0.9, value=0.5, step=0.01)

# Generate sample data
X, y = generate_data(num_rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Print number of rows in train and test sets
st.markdown(f'<p style="font-size:16px; color:blue;">Number of rows in training set: <strong>{X_train.shape[0]}</strong></p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:16px; color:green;">Number of rows in testing set: <strong>{X_test.shape[0]}</strong></p>', unsafe_allow_html=True)


# Train logistic regression model
model = train_logistic_regression(X_train, y_train)

# Evaluate model
cm, fpr, tpr, roc_auc, f1 = evaluate_model(model, X_test, y_test, threshold)

# Calculate precision, recall, and accuracy
precision, recall, accuracy = calculate_metrics(cm)



# Display confusion matrix
st.subheader('Confusion Matrix:')
st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))



# Display AUC score
# st.write(f'AUC Score: {roc_auc}')


selected_index = np.argmax(tpr >= threshold)
# Display TPR, FPR, and F1 Score at selected threshold in a table
# Display metrics in a table
st.subheader('Matrices Values:')
data = {
    'Metric': ['AUC Score', 'TPR', 'FPR', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
    'Value': [roc_auc, tpr[selected_index], fpr[selected_index], f1, precision, recall, accuracy]
}
df = pd.DataFrame(data).T
st.write(df)




# Display ROC curve
st.subheader('ROC Curve:')
st.line_chart(pd.DataFrame({'FPR': fpr, 'TPR': tpr}).set_index('FPR'))



