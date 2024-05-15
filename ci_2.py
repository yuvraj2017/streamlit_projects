import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Function to calculate confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_error = stats.sem(data)
    margin_error = std_error * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin_error, mean, mean + margin_error

# Function to calculate prediction interval
def prediction_interval(x, y, confidence=0.95):
    n = len(x)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    s_err = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    interval = t_val * s_err * np.sqrt(1 + 1 / n + (x - np.mean(x)) ** 2 / ((n - 1) * np.var(x)))
    return y_pred - interval, y_pred, y_pred + interval

# Streamlit UI
st.title('Confidence and Prediction Intervals for Linear Regression')
st.write('This app demonstrates confidence interval and prediction interval for linear regression.')

# User input
population_mean = st.sidebar.number_input('Enter population mean:', value=0.0, format="%.2f")
population_std = st.sidebar.number_input('Enter population standard deviation:', value=1.0, format="%.2f")
sample_size = st.sidebar.number_input('Enter sample size:', value=100, format="%d", step=20)
confidence_level = st.sidebar.slider('Select confidence level:', min_value=80, max_value=99, value=95, step=1)

# Convert confidence level to decimal
confidence_level = confidence_level / 100

# Generate random sample from population
np.random.seed(42)
sample = np.random.normal(population_mean, population_std, sample_size)

# Calculate confidence interval
lower_ci, mean, upper_ci = confidence_interval(sample, confidence_level)

# Calculate prediction interval
x = np.arange(1, sample_size + 1)
lower_pi, y_pred, upper_pi = prediction_interval(x, sample, confidence_level)

# Confidence Interval Interpretation
ci_interpretation = f"The confidence interval (CI) represents the range of values within which we are {int(confidence_level*100)}% confident that the true mean of the population lies. In other words, if we were to take multiple samples from the same population and compute confidence intervals for each sample, approximately {int(confidence_level*100)}% of those intervals would contain the true population mean."

# Prediction Interval Interpretation
pi_interpretation = f"The prediction interval (PI) represents the range of values within which a new observation is expected to fall with {int(confidence_level*100)}% confidence. Unlike the confidence interval, the prediction interval takes into account both the uncertainty in estimating the mean value and the variability of individual data points around the regression line."

# Print interpretations
st.subheader('Interpretations:')
st.write("Confidence Interval Interpretation:")
st.info(ci_interpretation)
st.write("Prediction Interval Interpretation:")
st.info(pi_interpretation)

# Print results
st.subheader('Results:')
st.write(f"Population Mean: {population_mean}")
st.write(f"Population Standard Deviation: {population_std}")
st.write(f"Sample Size: {sample_size}")
st.write(f"Confidence Level: {confidence_level}")
st.write(f"Confidence Interval: [{lower_ci}, {upper_ci}]")
st.write("Prediction Interval:")
# for i in range(len(x)):
#     st.write(f"For x={x[i]}, the prediction interval is [{lower_pi[i]}, {upper_pi[i]}]")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=x, y=sample, label='Sample Data', ax=ax)
sns.lineplot(x=x, y=y_pred, color='red', label='Regression Line', ax=ax)
ax.fill_between(x, lower_ci, upper_ci, alpha=0.2, color='green', label='Confidence Interval')
ax.fill_between(x, lower_pi, upper_pi, alpha=0.2, color='blue', label='Prediction Interval')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression with Confidence and Prediction Intervals')
ax.legend()
st.pyplot(fig)
