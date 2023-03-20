#!/usr/bin/env python
# coding: utf-8

# Import numpy library

# In[199]:


import numpy as np


# Import Pandas Library

# In[200]:


import pandas as pd


# Read the csv file from the location and create a pandas dataframe using the read file to analyse the data

# In[3]:


ipl_df = pd.read_csv("F:\IPL 2022 Batters.csv")


# In[4]:


ipl_df


# Use "info" function to get the idea about the type of entries and their respective counts. We can also get the idea about any null entries if any.

# In[201]:


ipl_df.info()


# Use "describe" function to get the mean, standard deviation, max, min of the given data

# In[6]:


ipl_df.describe()


# Import Matplotlib and seaborn libraries to visualise and analyse the data using graphs.

# In[202]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[203]:


ipl_df.Runs.describe()


# Setting the figure size, font size and color.

# In[204]:


matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# Here we're using renaming function to rename the column names of type int into some string type. Here we renamed 100, 50 and 4s
# column to Hundreds, Fifty and Fours.

# In[206]:


ipl_df.rename(columns = {'4s': 'Fours','50': 'Fifty','100':'Hundreds'}, inplace = True)       
ipl_df


# In this step we're using matplotlib library to plot a scatter plot between Runs and Fours to get the idea about how the two 
# items are related. The red line is the best fit line. The "corr" function is used to calculate the correlation between the 
# two quantities

# In[212]:


# plotting the data
x = ipl_df.Runs
y = ipl_df.Fours
plt.scatter(x, y)
 
# This will fit the best line into the graph
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
         (np.unique(x)), color='red')
plt.xlabel('Runs')
plt.ylabel('Fours')
ipl_df.Runs.corr(ipl_df.Fours)


# In this step we're using matplotlib library to plot a scatter plot between Strikerate and Runs to get the idea about how the 
# two items are related. The red line is the best fit line.

# In[208]:


# plotting the data
x = ipl_df.Runs
y = ipl_df.SR
plt.scatter(x, y)
 
# This will fit the best line into the graph
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
         (np.unique(x)), color='red')
plt.xlabel('Runs')
plt.ylabel('SR')
ipl_df.Runs.corr(ipl_df.SR)


# In this step we're using matplotlib library to plot a scatter plot between Stirke Rate and Fours to get the idea about how the 
# two items are related. The red line is the best fit line.

# In[209]:


# plotting the data
x = ipl_df.SR
y = ipl_df.Fours
plt.scatter(x, y)
 
# This will fit the best line into the graph
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
         (np.unique(x)), color='red')
plt.xlabel('SR')
plt.ylabel('Fours')
ipl_df.SR.corr(ipl_df.Fours)


# In this step we used seaborn library to form a correlation matrix. It shows how one quantity is related with other qauantities.
# Correlation of 1 represents that the two quantities are highly correlated or they have a stronger relationship. If correlation
# value is small it means the two quantities are weakly dependent on each other and if correlation value is high then the two 
# quantities are strongly dependent on each other.

# In[215]:


sns.heatmap(ipl_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');


# In[216]:


plt.title('Runs vs. Fours')
sns.scatterplot(data=ipl_df, x='Runs', y='Fours', alpha=0.7, s=15);


# Define a function to calculate number of fours using runs scored. Here w and b are the paramters, where w is the slope of the
# straight line and b is the intercept.

# In[218]:


def estimate_fours(Runs, w, b):
    return w * Runs + b


# In[ ]:





# In[219]:


Runs = ipl_df.Runs
estimated_fours = estimate_fours(Runs, w, b)


# In[220]:


plt.plot(Runs, estimated_fours, 'r-o');
plt.xlabel('Estimated Fours');
plt.ylabel('Runs');


# In this piece of code we're plotting the estimated value and the actual value.

# In[221]:


target = ipl_df.Fours

plt.plot(Runs, estimated_fours, 'r', alpha=0.9);
plt.scatter(Runs, target, s=8,alpha=0.8);
plt.xlabel('Estimated Fours');
plt.ylabel('Runs')
plt.legend(['Estimate', 'Actual']);


# Here we defined a "try_parameters" function to predict the results by tuning w and b. And getting the best set of values of
# w and b to get the best fitted line.

# In[223]:


def try_parameters(w, b):
    Runs = ipl_df.Runs
    target = ipl_df.Fours
    
    estimated_fours = estimate_fours(Runs, w, b)
    
    plt.plot(Runs, estimated_fours, 'r', alpha=0.9);
    plt.scatter(Runs, target, s=8,alpha=0.8);
    plt.xlabel('Runs');
    plt.ylabel('Fours')
    plt.legend(['Estimate', 'Actual']);


# In[224]:


try_parameters(0.085, 0.5)


# Here we defined root mean square loss function to get the idea about how far out results are from the actual value.

# In[90]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# In[225]:


targets = ipl_df['Fours']
predicted = estimate_fours(ipl_df.Runs, 0.09, -0.3)


# In[226]:


rmse(targets, predicted)


# Now we used scikit-learn library to use the inbuilt Linear Regression model to predict the output.

# In[168]:


from sklearn.linear_model import LinearRegression


# In[169]:


model = LinearRegression()


# In[170]:


inputs = ipl_df[['Runs']]
targets = ipl_df.Fours
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)


# In[171]:


model.fit(inputs, targets)


# In[172]:


model.predict(np.array([[863], 
                        [616], 
                        [508]]))


# In[173]:


predictions = model.predict(inputs)


# In[174]:


predictions


# In[175]:


rmse(targets, predictions)


# In[176]:


model.coef_


# In[177]:


model.intercept_


# In[178]:


try_parameters(model.coef_, model.intercept_)


# In[179]:


# Create inputs and targets
inputs, targets = ipl_df[['Runs', 'BF','Fifty','Inns']], ipl_df['Fours']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[180]:


model.coef_, model.intercept_


# In[183]:


from sklearn.model_selection import train_test_split


# In[184]:


inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)


# In[185]:


# Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)

# Generate predictions
predictions_test = model.predict(inputs_test)

# Compute loss to evalute the model
loss = rmse(targets_test, predictions_test)
print('Test Loss:', loss)


# In[186]:


# Generate predictions
predictions_train = model.predict(inputs_train)

# Compute loss to evalute the model
loss = rmse(targets_train, predictions_train)
print('Training Loss:', loss)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
