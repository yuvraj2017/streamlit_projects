
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
plot_spot = st.empty()


st.sidebar.header('SELECT BELOW')
length=st.sidebar.selectbox('Select the length of DataFrame',[10,20,30,40])
LR=st.sidebar.selectbox('Learning Rate', [0.01, 0.1, 0.001, 0.005, 0.0005])
b0=7
b1=15
# X= np.arange(0,length,1)
# y=np.random.randint(0,2,length)
# X=X.reshape(-1,1)
# y=y.reshape(-1,1)
def log_lr(b0,b1):
    y_hat=(np.exp(b0+b1*X))/(1+np.exp(b0+b1*X))
    error=y_hat-y
    grad_b0=np.sum(error)
    grad_b1=np.sum(X*error)
    return grad_b0,grad_b1



# total=1e-2
# for i in range(1000000):
#     grad_b0,grad_b1 = log_lr(b0,b1)
#     if (np.abs(grad_b0)<=total) and (np.abs(grad_b1)<=total):
#         #print(f'the value of b0 and b1 are found at {i}th iteration, b0={b0},b1={b1}')
#         break
#     b0=b0-LR*grad_b0
#     b1=b1-LR*grad_b1
#     y_hat=(np.exp(b1*X+b0)/(1+(np.exp(b1*X+b0))))
#     #plt.scatter(X, y, color='blue', label='Data Points')
#     plt.plot(X, y_hat, color='red', label='Regression Line')

# def plot_best_line(X, y, b0, b1):
#     y_hat=(np.exp(b1*X+b0)/(1+(np.exp(b1*X+b0))))
#     plt.scatter(X, y, color='blue', label='Data Points')
#     plt.plot(X, y_hat, color='red', label='Regression Line')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#     st.pyplot()    

# plot_best_line(X,y,b0,b1)



    


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Define the logistic regression function
def logistic_regression(X, b0, b1):
    return np.exp(b1 * X + b0) / (1 + np.exp(b1 * X + b0))

# Initialize Streamlit app
st.title('Updating Logistic Regression Plot in Streamlit Loop')

# Generate random data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.random.randint(2, size=100)

# Total threshold for gradient descent convergence
total = 1e-5

# Create a placeholder for the plot
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data Points')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Logistic Regression Plot')

# Update the plot within a loop
for i in range(1000000):
    grad_b0, grad_b1 = log_lr(b0, b1)  
    
    if np.abs(grad_b0) <= total and np.abs(grad_b1) <= total:
        st.write(f'The value of b0 and b1 are found at {i}th iteration, b0={b0}, b1={b1}')
        break
    
    b0 -= LR * grad_b0
    b1 -= LR * grad_b1
    
    y_hat = logistic_regression(X, b0, b1)
    
    # Clear previous plot
    clear_output(wait=True)
    ax.clear()
    with plot_spot:
        # Plot data points
        ax.scatter(X, y, color='blue', label='Data Points')
    
        # Plot regression line
        ax.plot(X, y_hat, color='red', label='Regression Line')
    
        # Refresh the plot in Streamlit
        st.pyplot(fig)

   



