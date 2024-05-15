import streamlit as st
import numpy as np
from scipy import stats

# Streamlit UI
st.title('Type I and Type II Errors')

# User input for parameters
true_mean = st.sidebar.number_input('True Mean of Population', min_value=0.0, max_value=100.0, value=50.0)
sample_size = st.sidebar.slider('Sample Size', min_value=10, max_value=100, value=50)
alpha = st.sidebar.slider('Significance Level (α)', min_value=0.01, max_value=0.10, value=0.05, step=0.01)
effect_size = st.sidebar.slider('Effect Size', min_value=0.1, max_value=10.0, value=1.0)
alternative = st.sidebar.selectbox('Alternative Hypothesis', ['Two-sided', 'One-sided (greater)', 'One-sided (less)'])

# Generate sample data
np.random.seed(0)
sample_data = np.random.normal(loc=true_mean, scale=1.0, size=sample_size)

# Perform hypothesis test
if alternative == 'Two-sided':
    test_result = stats.ttest_1samp(sample_data, true_mean)
elif alternative == 'One-sided (greater)':
    test_result = stats.ttest_1samp(sample_data, true_mean, alternative='greater')
else:  # One-sided (less)
    test_result = stats.ttest_1samp(sample_data, true_mean, alternative='less')

# Interpret test result
p_value = test_result.pvalue
if p_value < alpha:
    if alternative == 'Two-sided':
        st.write('Null hypothesis rejected: There is evidence of a significant difference from the true mean.')
    else:
        st.write('Null hypothesis rejected: There is evidence of a significant difference from the true mean in the specified direction.')
else:
    st.write('Null hypothesis not rejected: No significant difference detected from the true mean.')

# Display p-value
st.write(f'p-value: {p_value}')

# Calculate Type II Error Rate
if alternative == 'Two-sided':
    type_i_error_rate = alpha / 2
    type_ii_error_rate = None
else:
    if alternative == 'One-sided (less)':
        type_ii_error_rate = stats.norm(loc=true_mean + effect_size, scale=1.0).cdf(stats.norm.ppf(1 - alpha))
    else:
        type_ii_error_rate = stats.norm(loc=true_mean - effect_size, scale=1.0).cdf(stats.norm.ppf(alpha))
    type_i_error_rate = alpha if alternative == 'One-sided (less)' else 0.5 - alpha / 2

st.write(f'Type I Error Rate (α): {type_i_error_rate}')
if type_ii_error_rate is not None:
    st.write(f'Type II Error Rate (β): {type_ii_error_rate}')
