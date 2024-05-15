import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Import VIF function

# Function to calculate VIF
def calculate_vif(data, threshold=5):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data

# Main function
st.title("Multicollinearity Analysis")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Automatic feature selection
    feature_columns = data.columns.tolist()

    # Selector for label column
    label_column = st.sidebar.selectbox("Select Label Column", options=feature_columns)

    # Remove label column from feature selection
    feature_columns.remove(label_column)

    # Filter data to selected columns
    selected_data = data[feature_columns]

    # Calculate VIF
    vif_results = calculate_vif(selected_data.dropna())

    # Display VIF results
    st.subheader("Variance Inflation Factor (VIF)")
    st.write(vif_results.T)

    # Interpretation for infinite VIF values
    # infinite_vif_features = vif_results[vif_results['VIF'] == float('inf')]['Feature'].tolist()
    # if infinite_vif_features:
    #     st.warning("""
    #     **Interpretation:**
        
    #     Features with infinite VIF values indicate perfect multicollinearity. This means that the following features can be exactly predicted from other independent variables with a linear combination: {}
        
    #     In such cases, these features are redundant for regression analysis and may need to be removed from the model to avoid issues like overfitting.
    #     """.format(', '.join(infinite_vif_features)))


     # Interpretation for low VIF values
    low_vif_threshold = 10  # Define your threshold here
    low_vif_features = vif_results[vif_results['VIF'] < low_vif_threshold]['Feature'].tolist()
    if low_vif_features:
        interpretation_html = """
        <div style="background-color:#BBDEFB;padding:15px;border-radius:10px;margin-bottom:10px;">
        <h2 style="font-size:18px;color:#2196F3;"><b>Interpretation for low VIF values:</b></h2>
        <p style="font-size:16px;color:#1565C0;">Features with VIF values below {} indicate low multicollinearity. This means that these features have relatively low correlations with other independent variables:</p>
        <p style="font-size:16px;color:#1565C0;"><b>{}</b>.</p>
        <p style="font-size:16px;color:#1565C0;">These features are likely suitable for inclusion in the model.</p>
        </div>
        """.format(low_vif_threshold, ', '.join(low_vif_features))
        st.markdown(interpretation_html, unsafe_allow_html=True)



    # Interpretation for infinite VIF values
    infinite_vif_features = vif_results[vif_results['VIF'] == float('inf')]['Feature'].tolist()
    if infinite_vif_features:
        interpretation_html = """
        <div style="background-color:#E0F2F1;padding:15px;border-radius:10px;margin-bottom:10px;">
        <h2 style="font-size:18px;color:#009688;"><b>Interpretation for infinite VIF values:</b></h2>
        <p style="font-size:16px;color:#37474F;">Features with infinite VIF values indicate perfect multicollinearity. This means that the following features can be exactly predicted from other independent variables with a linear combination: <b>%s</b>.</p>
        <p style="font-size:16px;color:#37474F;">In such cases, these features are redundant for regression analysis and may need to be removed from the model to avoid issues like overfitting.</p>
        </div>
        """ % (', '.join(infinite_vif_features))
        st.markdown(interpretation_html, unsafe_allow_html=True)

    # Interpretation for high VIF values
    high_vif_threshold = 10  # Define your threshold here
    high_vif_features = vif_results[vif_results['VIF'] > high_vif_threshold]['Feature'].tolist()
    if high_vif_features:
        interpretation_html = """
        <div style="background-color:#F1F8E9;padding:15px;border-radius:10px;margin-bottom:10px;">
        <h2 style="font-size:18px;color:#689F38;"><b>Interpretation for high VIF values:</b></h2>
        <p style="font-size:16px;color:#33691E;">Features with VIF values above {} indicate high multicollinearity. This means that the following features have a strong correlation with other independent variables, potentially leading to issues such as unstable coefficient estimates and decreased interpretability:</p>
        <p style="font-size:16px;color:#33691E;"><b>{}</b>.</p>
        <p style="font-size:16px;color:#33691E;">In such cases, consider removing or combining these features to improve the model's performance.</p>
        </div>
        """.format(high_vif_threshold, ', '.join(high_vif_features))
        st.markdown(interpretation_html, unsafe_allow_html=True)

       




