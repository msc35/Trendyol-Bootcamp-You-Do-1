# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 08:06:34 2022

@author: mehme
"""


import pandas as pd
import numpy as np


df_a = pd.read_csv('articles.csv.gz')
df_c = pd.read_csv('customers.csv.gz')
df_t = pd.read_csv('transactions.csv.gz')


df_a_s = df_a.sample(n = 10000)
df_c_s = df_c.sample(n = 50000)
df_t_s = df_t.sample(n = 150000)

mergedRes = pd.merge(df_t_s, df_a, on ='article_id',how = "inner")

dfMerged = pd.merge(mergedRes, df_c_s, on ='customer_id',how = "inner")

mergedDF = dfMerged.drop_duplicates()


################ Data Analysis + Feature Selection ############################



mergedDF["t_dat"] = pd.to_datetime(mergedDF["t_dat"])

mergedDF["t_dat"].max()

mergedDF["time_passed"] = (pd.Timestamp('2020-09-30 00:00:00') - mergedDF["t_dat"]).dt.days #Added 8 days to maximum to not to get infinite values on rate.

mergedDF["fashion_news_freq"] = np.where(mergedDF['fashion_news_frequency'] == "Regularly", 1, 0)



"""
After the analysis, I decided to continue with matrix factorization. However, I will create a new column as RATE and
use it as a rating between 0 and 1. (0 - No interest, 1 - High interest). 
"""
mergedDF["Rate"] =  np.where(mergedDF['time_passed'] >= 100, 0, 1) # Transactions on last 100 days
# mergedDF["Rate"] = np.ones(mergedDF.shape[0])
w = np.random.rand(4)



################# Creating Weighted Rates #####################################
mergedDF["RatePred"] = (
    w[0] * np.log(mergedDF["price"]) +     
    w[1] *(np.exp(-np.log(mergedDF["time_passed"]))) +    
    w[2] *((mergedDF["age"] - mergedDF["age"].min()) / (mergedDF["age"].max() - mergedDF["age"].min())) +
    w[3] *(mergedDF["fashion_news_freq"])
)

# pd.plotting.scatter_matrix(mergedDF.iloc[:,[3,4,30,31,33,35]])


############### Normalizing between [0,1] the rate predictions ################
mergedDF["RatePredNormalized"] = (mergedDF["RatePred"] - mergedDF["RatePred"].min()) / (mergedDF["RatePred"].max() - mergedDF["RatePred"].min()) 


# mergedDF.to_excel("Data_Analysis.xlsx") # To check with R since it is easier.

####################### Pivot Table with Selected Features ####################

uniqueDF1 = mergedDF.groupby(['article_id','customer_id'], as_index=False)['RatePredNormalized','Rate','time_passed', 'age', 'price', 'fashion_news_freq'].mean() # Unique article - customer pairs with predicted weight
uniqueDF = uniqueDF1.dropna()

index_df = uniqueDF.groupby(['customer_id'], as_index=False)['RatePredNormalized','Rate','time_passed', 'age', 'price', 'fashion_news_freq'].mean()
index_df_art = uniqueDF.groupby(['article_id'], as_index=False)['RatePredNormalized','Rate', 'price'].mean()
index_df_time = uniqueDF.groupby(['customer_id'], as_index=False)['time_passed'].min()




R = uniqueDF.pivot(index='customer_id', columns='article_id', values='Rate').values

R_Pred = uniqueDF.pivot(index='customer_id', columns='article_id', values='RatePredNormalized').values

nan_count = np.count_nonzero(np.isnan(R))



########################## Gradient Descent ###################################

irow, jcol = np.where(~np.isnan(R)) # Nan Olmayanlar

R_Pred_c = R_Pred.copy()

w = np.random.rand(4)
total_e = 0
from tqdm import trange
alpha = 0.000001
with trange(1000) as epochs:
    for _ in epochs:
        error_prev = np.copy(total_e)
        total_e = 0
        gradient0 = 0
        gradient1 = 0
        gradient2 = 0
        for i, j in zip(irow, jcol):
            # Prediction of r_ij
            y_pred = R_Pred_c[i][j]
            e = R[i][j] - y_pred
            
            w[0] += 2 * e * np.log(index_df_art["price"].iloc[j]) * alpha
            
            w[1] += 2 * e * (np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) * alpha
            
            w[2] += 2 * e * ((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min())) * alpha          
            
            w[3] += 2 * e * (index_df["fashion_news_freq"].iloc[i]) * alpha          

            
            ###### Gradients to check if the function converges ###############
            # gradient0 += 2 * e * ((index_df_art["price"].iloc[j] - index_df_art["price"].min()) / (index_df_art["price"].max() - index_df_art["price"].min())) * alpha
            
            # gradient1 += 2 * e * (np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) * alpha
            
            # gradient2 += 2 * e * ((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min())) * alpha          
            
            
            R_Pred_c[i][j] = (
                w[0] * np.log(index_df_art["price"].iloc[j]) +     
                w[1] *(np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) +    
                w[2] *((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min()))+
                w[3] *(index_df["fashion_news_freq"].iloc[i])
            )
            total_e += e ** 2
        
        
        # print("Gradients: ", gradient0/uniqueDF.shape[0], gradient1/uniqueDF.shape[0], gradient2/uniqueDF.shape[0])
        epochs.set_description(f'Total Square Error: {total_e:.2f}')
        if abs(error_prev - total_e) <= 0.01:
            print("early stop")
            break
    
    
    

    
    
    
    
    
########### Gradient Descent for Logistic Regression Based Model ##############   Error Functionu Değiştir + Gradientleri Değiştir

irow, jcol = np.where(~np.isnan(R)) # Nan Olmayanlar

R_Pred_c = R_Pred.copy()

w = np.random.rand(4)
total_e = 0
from tqdm import trange
alpha = 0.0001
with trange(1000) as epochs:
    for _ in epochs:
        error_prev = np.copy(total_e)
        total_e = 0
        mse = 0
        for i, j in zip(irow, jcol):
            # Prediction of r_ij
            y_pred = R_Pred_c[i][j]
            y = R[i][j]
            e = -y*np.log(y_pred) - (1-y) * np.log(1-y_pred) # Cost Function
            e2 = y - y_pred
            
            w[0] += (y-y_pred) * (np.log(index_df_art["price"].iloc[j]))  * alpha
            
            w[1] += (y-y_pred) * (np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) * alpha
            
            w[2] += (y-y_pred)* ((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min())) * alpha          
            
            w[3] += (y-y_pred)* (index_df["fashion_news_freq"].iloc[i]) * alpha          

            
            P = (
                w[0] * (np.log(index_df_art["price"].iloc[j])) +     
                w[1] *(np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) +    
                w[2] *((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min()))+
                w[3] * (index_df_art["fashion_news_freq"].iloc[i])          
            )
            R_Pred_c[i][j] = 1/(1+np.exp(-P))
            total_e += e ** 2
            mse += e2 **2
        
        # print("Gradients: ", gradient0/uniqueDF.shape[0], gradient1/uniqueDF.shape[0], gradient2/uniqueDF.shape[0])
        epochs.set_description(f'Total Square Error: {total_e:.2f}')
        if abs(error_prev - total_e) <= 0.01:
            print("early stop")
            break

        
#################### Fill NA Values in R_Pred #################################


R_Pred_Final = R_Pred.copy()
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        P = (
            w[0] * (np.log(index_df_art["price"].iloc[j])) +     
            w[1] *(np.exp(-np.log(index_df_time["time_passed"].iloc[i]))) +    
            w[2] *((index_df["age"].iloc[i] - index_df["age"].min()) / (index_df["age"].max() - index_df["age"].min())) +
            w[3] * (index_df_art["fashion_news_freq"].iloc[i]) 
        )
        R_Pred_Final[i][j] = 1/(1+np.exp(-P))






