import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Central Limit Theorem Demonstration")

def plot_clt_distribution(distribution, sample_size, num_samples, population_size):
    if distribution == "Uniform":
        population_data = np.random.uniform(0, 1, population_size)
    elif distribution == "Normal":
        population_data = np.random.normal(0, 1, population_size)
    elif distribution == "Exponential":
        population_data = np.random.exponential(1, population_size)
    
    if distribution == "Uniform":
        data = np.random.uniform(0, 1, (num_samples, sample_size))
    elif distribution == "Normal":
        data = np.random.normal(0, 1, (num_samples, sample_size))
    elif distribution == "Exponential":
        data = np.random.exponential(1, (num_samples, sample_size))
    
    sample_means = np.mean(data, axis=1)
    population_mean = np.mean(population_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(sample_means, kde=True, stat="density", color='skyblue', ax=ax1)
    ax1.set_title(f"Distribution of Sample Means (n={sample_size}, N={num_samples})")
    ax1.set_xlabel("Sample Mean")
    ax1.set_ylabel("Density")

    sns.histplot(population_data, kde=True, stat="density", color='lightcoral', ax=ax2)
    ax2.set_title("Distribution of Population Data")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")

    ax2.axvline(population_mean, color='red', linestyle='--', label=f'Population Mean ({population_mean:.2f})')
    ax2.legend()

    st.table({"Sample Mean": [np.mean(sample_means)], "Population Mean": [population_mean]})

    st.pyplot(fig)

st.sidebar.header("Adjust Parameters")
distribution = st.sidebar.selectbox("Select Distribution", ["Uniform", "Normal", "Exponential"])
population_size = st.sidebar.slider("Population Size", min_value=100, max_value=10000, value=1000, step=100)
sample_size = st.sidebar.slider("Sample Size (n)", min_value=5, max_value=100, value=30, step=5)
num_samples = st.sidebar.slider("Number of Samples (N)", min_value=10, max_value=1000, value=100, step=10)


st.info("Central Limit Theorem (CLT):\n"
        "- The CLT states that the distribution of sample means approaches a normal distribution as the sample size increases.\n"
        "- Regardless of the shape of the original population distribution, the distribution of sample means tends towards normality.\n"
        "- This theorem is crucial for making inferences about a population based on samples, even if the population's distribution is unknown.\n"
        "- The CLT holds under certain conditions, such as samples being independent and having finite variance.\n"
        "- It has widespread applications in fields like finance, quality control, and hypothesis testing.")

plot_clt_distribution(distribution, sample_size, num_samples, population_size)
