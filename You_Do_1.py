# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:02:29 2022

@author: mehme
"""

import numpy as np
import streamlit as st
from sklearn.datasets import fetch_california_housing
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


np.random.seed(0)

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

df = pd.DataFrame(
dict(MedInc=X['MedInc'], Price=cal_housing.target))
x = np.array(df["MedInc"])

st.latex(r'''\huge{\text{You Do 1}}''')
st.latex(r'''\huge{\text{Mehmet Selim Çetin}}''')


st.latex(r'''\text{I prepared two models, since I understood the problem in two different ways. First one is,}''')
         
st.latex(r''' \text{the y values that we are predicting are in an interval of (0,5.1), although we may encounter}''')
st.latex(r''' \text{values outside this interval, we do not want to consider them in the loss function.}''')
st.latex(r''' \text{The second one is, we need a function that is constant after a treshold $\theta$. Therefore I took a}''')
st.latex(r''' \text{constant value for penalty after a treshold of error.}''')



st.latex("Model 1")
st.latex(r'''\text{Loss}_i=
\begin{cases}
    (y_{true_i} - (\beta_0+\beta_1*x_i))^2,& \text{if } y_i \leq \theta \\ 
    \theta^2, & \text{otherwise} 
\end{cases}''')

st.latex("Model 2")
st.latex(r'''\text{Loss}_i=
\begin{cases}
    (y_{true_i} - (\beta_0+\beta_1*x_i))^2,& \text{if } (y_i-(\beta_0+\beta_1*x_i))^2 \leq \theta^2 \\ 
    \theta^2, & \text{otherwise} 
\end{cases}''')


st.latex(r"""\text{Gradients for model 1:}""")
st.latex(r''' d(L_1(\beta))/(d\beta_0) = 
\begin{cases}
    -2 * (y_i-y_{pred_i}),& \text{if } y_i \leq \theta \\ 
    0, & \text{otherwise} 
\end{cases}''')

st.latex(r''' d(L_1(\beta))/(d\beta_1) = 
\begin{cases}
    -2 * x_i*(y_i-y_{pred_i}),& \text{if } y_i \leq \theta \\ 
    0, & \text{otherwise} 
\end{cases}''')




st.latex(r"""\text{Gradients for model 2:}""")
st.latex(r''' d(L_1(\beta))/(d\beta_0) = 
\begin{cases}
    -2 * (y_i-y_{pred_i}),& \text{if } (y_i-(\beta_0+\beta_1*x_i))^2 \leq \theta^2 \\
    0, & \text{otherwise} 
\end{cases}''')

st.latex(r''' d(L_1(\beta))/(d\beta_1) = 
\begin{cases}
    -2 * x_i*(y_i-y_{pred_i}),& \text{if } (y_i-(\beta_0+\beta_1*x_i))^2 \leq \theta^2 \\
    0, & \text{otherwise} 
\end{cases}''')



st.latex(r"""\text{Checking "Almost" Convexity for model 1:}""")
beta = np.random.random(2)
y_pred = beta[0] + beta[1] * x


error_list  = []
errors = []
for i in range(len(y_pred)):
    error_list.append((y[i] - y_pred[i]) ** 2)
    errors.append(y[i] - y_pred[i])
error = np.array(error_list)

theta = 2
p_list = []
for i in range(len(error)):
    if  (y[i]) <= theta**2:
        pen = (error[i])
    else:
        pen = theta**2
    p_list.append(pen)
    
penalty = np.array(p_list)


l = pd.DataFrame({"Error" : errors, "Penalty" : penalty})

fig = px.scatter(l, x="Error", y="Penalty")
st.plotly_chart(fig, use_container_width=True)

st.latex(r"""\text{The first model is not convex for most of the values, we cannot use the gradient decent method but I tried :)}""")

########################################################

st.latex(r"""\text{Checking "Almost" Convexity for model 2:}""")
error_list  = []
errors = []
for i in range(len(y_pred)):
    error_list.append((y[i] - y_pred[i]) ** 2)
    errors.append(y[i] - y_pred[i])
error = np.array(error_list)

theta = 3
p_list = []
for i in range(len(error)):
    if  (error[i]) <= theta**2:
        pen = (error[i])
    else:
        pen = theta**2
    p_list.append(pen)
    
penalty = np.array(p_list)


l = pd.DataFrame({"Error" : errors, "Penalty" : penalty})

fig = px.scatter(l, x="Error", y="Penalty")
st.plotly_chart(fig, use_container_width=True)

st.latex(r"""\text{The second model is convex for most of the values, we can use the gradient decent method.}""")

st.latex(r''' \text{(I added L2 regularization terms in the models.)}''')


def isNaN(num):
    return num != num



def model1(x, y, lam,theta, alpha=0.00001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)
    filter_array = y > theta
    for i in range(2000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y[filter_array] - y_pred[filter_array]).mean() + 2 * lam * beta[0]
        g_b1 = -2 * (x[filter_array] * (y[filter_array] - y_pred[filter_array])).mean() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break
        elif isNaN(beta[0]):
            print("nan geldi")
            break
    return beta


beta3 = model1(x,y,0.001,3)



def model2(x, y, lam,theta, alpha=0.00001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)
    y_pred: np.ndarray = beta[0] + beta[1] * x
    filter_array = np.square(y - y_pred) > theta**2
    for i in range(2000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y[filter_array] - y_pred[filter_array]).mean() + 2 * lam * beta[0]
        g_b1 = -2 * (x[filter_array] * (y[filter_array] - y_pred[filter_array])).mean() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break
        elif isNaN(beta[0]):
            print("nan geldi")
            break
    return beta


beta4 = model2(x,y,0.001,3)

# Mean Percentage error:
    
    
def PE(y,y_pred):
    pen = 0
    for i in range(len(y)):
        pen += abs((y[i] - y_pred[i])/y[i])/len(y)
        
    return pen



######################### WE DO Solution ##############################



def modelwedo(x, y, lam, alpha=0.000001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        if beta[0] >= 0:
            g_b0 = -2 * (y - y_pred).sum() + lam
        else:
            g_b0 = -2 * (y - y_pred).sum() - lam

        if beta[1] >= 0:
            g_b1 = -2 * (x * (y - y_pred)).sum() + lam
        else:
            g_b1 = -2 * (x * (y - y_pred)).sum() - lam

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta

betaw = modelwedo(x,y,0.1)


###### Prediction ##################
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data points'))
fig.add_trace(go.Scatter(x=x, y=beta3[0] + beta3[1] * x, mode='lines', name='Model 1'))
fig.add_trace(go.Scatter(x=x, y=beta4[0] + beta4[1] * x, mode='lines', name='Model 2'))
fig.add_trace(go.Scatter(x=x, y=betaw[0] + betaw[1] * x, mode='lines', name='Model We do'))


st.plotly_chart(fig, use_container_width=True)


fig1 = px.scatter(x=y, y=beta3[0] + beta3[1] * x)
fig2 = px.scatter(x=y, y=beta4[0] + beta4[1] * x)
fig3 = px.scatter(x=y, y=betaw[0] + betaw[1] * x)

st.write("{Model 1 y vs y_pred:}")

st.plotly_chart(fig1, use_container_width=True)

st.write("{Model 2 y vs y_pred:}")

st.plotly_chart(fig2, use_container_width=True)

st.write("{Model wedo y vs y_pred:}")

st.plotly_chart(fig3, use_container_width=True)

st.write("\text{Comparison of percentage errors:}""")

st.write("Model 1",PE(y,beta3[0] + beta3[1] * x))
st.write("Model 2",PE(y,beta4[0] + beta4[1] * x))
st.write("Model we do",PE(y,betaw[0] + betaw[1] * x))

st.write("Model we do and 2 are close, and are better then model 1 when compared with MPE")
st.write("Since they have similar error percentages, the second model is validated.")
st.write("When we try different values of theta, Model 2 gets slightly better.")
st.write("However, since we use random functions, sometimes the second model gives high MPE. But most of the time it is lower than 0.4.")
st.write("Since model 1 has less error for higher y values, its slope is higher, closer to observations in front.")
st.write("")




