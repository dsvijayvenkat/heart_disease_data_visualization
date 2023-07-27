#!/usr/bin/env python
# coding: utf-8

# ### Heart_disease_Data_Visuaization

# #### To visualize the dataset effectively, some important requirements are:
# 
# **Data Cleaning:** Ensure the dataset is cleaned and free from missing values, inconsistencies, or duplicate records. Clean data is crucial for accurate and meaningful visualizations.
# 
# **Data Exploration:** Understand the distribution and characteristics of each feature in the dataset. This exploration will help identify outliers, anomalies, or patterns that might impact the visualizations.
# 
# **Data Type Handling:** Check the data types of each feature and make sure they are appropriate for visualization. For example, numeric data should be numeric types, and categorical data should be represented as strings or categorical types.
# 
# **Data Scaling:** If the range of values in different features varies significantly, consider scaling the data to bring them to a common scale. Standardization or normalization may be necessary depending on the algorithms or visualizations used.
# 
# **Selection:** Select relevant features for visualization based on your specific objectives. Too many features can clutter visualizations and make them less interpretable.
# 
# **Target Variable** Understanding: Understand the target variable ('target' in this case) and its distribution, especially if you are creating visualizations related to classification or regression tasks.
# 
# **Plot Selection:** Choose appropriate plot types for different types of data. For example, use bar plots for categorical data, scatter plots for numerical data, and line plots for time series data.
# 
# **Color Mapping:** Use color effectively to represent different categories or groups in the data. Ensure that the color choices are visually distinguishable and meaningful.
# 
# **Labels and Titles:** Include clear and informative labels for the axes, legends, and titles for the plots. Labels help in understanding the information presented in the visualization.
# 
# **Data Interpretation:** Always provide interpretations and context for the visualizations. A good visualization should tell a clear and concise story about the data.
# 
# Remember that visualizations are powerful tools for understanding and communicating data insights. Careful consideration of these requirements will lead to clear, informative, and meaningful visualizations that facilitate better data analysis and decision-making.

# #### Information abot the dataset:
# 
# #### Feature Information:
# 
# 1.	age
# 
# 2.	sex
# 
# 3.	chest pain type (4 values)
# 
# 4.	resting blood pressure
# 
# 5.	serum cholestoral in mg/dl
# 
# 6.	fasting blood sugar > 120 mg/dl
# 
# 7.	resting electrocardiographic results (values 0,1,2)
# 
# 8.	maximum heart rate achieved
# 
# 9.	exercise induced angina
# 
# 10.	oldpeak = ST depression induced by exercise relative to rest
# 
# 11.	the slope of the peak exercise ST segment
# 
# 12.	number of major vessels (0-3) colored by flourosopy
# 
# 13.	thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
# 
# The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
# 
# More information about dataset : https://archive.ics.uci.edu/dataset/45/heart+disease 

# #### importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# #### Data Loading from local drive

# In[2]:


data = pd.read_csv(r"C:\Users\venka\OneDrive\Documents\datasets\heartdisease.csv")
data.iloc[0:4, :]


# #### Data Cleaning

# In[3]:


# Checking for missing vlaues
data.isnull().any() # this will return boolean expression as output


# In[4]:


data.isnull().sum()


# !! Oh, Geat there are no muissing values to fill. This will help us to save more time.

# #### Data Exploration
# 
# #### Data Inspection:

# In[5]:


print("Shape of the data: ", data.shape)
print("\n")
print("Dimention of the Data: ", data.ndim)
print("\n")
print("Size of the dataset: ", data.size)
print("\n")
print("Data type of the each column: ", data.dtypes)
print("\n")
print("Information about the dataset: ", data.info())


# #### Summary Statistics

# In[6]:


# To desccribe the statistics of the sentire data
data.describe().T


# From the above information we are able to observe some statistics of each column in the datset.
# 
# To perform other statistics on data set please look at below:

# In[7]:


# Select the numerical columns for calculation
numerical_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


# In[8]:


# Calculate the mean for each numerical column
mean_values = data[numerical_columns].mean()
mean_values
# or 
# data.mean()


# In[9]:


# Calculate the median for each numerical column
median_values = data[numerical_columns].median()
median_values
# or 
# data.median()


# In[10]:


# Calculate the mode for each numerical column
mode_values = data[numerical_columns].mode().iloc[0]
mode_values
# or 
# data.mode()


# In[11]:


# Calculate the variance for each numerical column
variance_values = data[numerical_columns].var()
variance_values
# or 
# data.var()


# In[12]:


# Calculate the standard deviation for each numerical column
std_deviation_values = data[numerical_columns].std()
std_deviation_values
# or 
# data.std()


# In[13]:


# Calculate the kurtosis for each numerical column
kurtosis_values = data[numerical_columns].kurtosis()
kurtosis_values
# or
# data.kurtosis()


# In[14]:


# Create a new DataFrame to store the results
statistics_df = pd.DataFrame({
    'Mean': mean_values,
    'Median': median_values,
    'Mode': mode_values,
    'Variance': variance_values,
    'Standard Deviation': std_deviation_values,
    'Kurtosis': kurtosis_values
})

# Display the statistics DataFrame
print(statistics_df)


# #### Data Visualization

# In[15]:


# Barplot for the count of "Sex" and "age" columns
plt.figure(figsize = (12,6))
sns.countplot(x = "age", data = data, hue = "sex")
plt.title("Distribution of Age and Gender")
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.show()


# In[16]:


# Histogram of 'age' and "sex"
plt.figure(figsize = (8,4))
sns.histplot(data=data, x="age", hue = "sex", kde=True)
plt.title('Distribution of Age and Sex')
plt.xlabel('Age')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.show()


# In[17]:


# Scatter plot of 'chol' vs. 'thalach'
plt.scatter(data['chol'], data['thalach'], c = data['target'], cmap = 'coolwarm', alpha = 0.8)
plt.colorbar(label='Target')
plt.title('Cholesterol vs. Maximum Heart Rate')
plt.xlabel('Cholesterol')
plt.ylabel('Max Heart Rate')
plt.show()


# In[18]:


# Box plot for 'cp' (chest pain type) grouped by 'exang' (exercise-induced angina)
sns.boxplot(x='exang', y='cp', data=data)
plt.title('Chest Pain Type vs. Exercise-Induced Angina')
plt.xlabel('Exercise-Induced Angina (0 = No, 1 = Yes)')
plt.ylabel('Chest Pain Type')
plt.show()


# In[19]:


# Bar Plot: Count of Target Classes
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=data)
plt.title('Count of Target Classes')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()


# In[20]:


# Box Plot: Age vs. Target
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='age', data=data)
plt.title('Age vs. Target')
plt.xlabel('Target')
plt.ylabel('Age')
plt.show()


# In[21]:


# Scatter Plot: Cholesterol vs. Max Heart Rate (Thalach) with Target Color Mapping
plt.figure(figsize=(8, 6))
sns.scatterplot(x='chol', y='thalach', hue='target', data=data)
plt.title('Cholesterol vs. Max Heart Rate (Thalach) with Target')
plt.xlabel('Cholesterol')
plt.ylabel('Max Heart Rate (Thalach)')
plt.show()


# In[22]:


# Violin Plot: Age and Gender
plt.figure(figsize=(8, 6))
sns.violinplot(x='sex', y='age', data=data)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender (1 = Male, 0 = Female)')
plt.ylabel('Age')
plt.show()


# In[23]:


# Bar Plot: Chest Pain Type by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='sex', y='cp', data=data)
plt.title('Chest Pain Type by Gender')
plt.xlabel('Gender (1 = Male, 0 = Female)')
plt.ylabel('Chest Pain Type')
plt.show()


# In[ ]:





# In[24]:


# Pairplot: Scatter Plot Matrix for Selected Features
selected_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sns.pairplot(data[selected_features], diag_kind='kde')
plt.suptitle('Scatter Plot Matrix for Selected Features', y=1.02)
plt.show()


# In[25]:


# Heatmap: Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ### Data Visualization Using Plotly

# In[26]:


import plotly.express as px

# Set the theme to a template with white background
px.defaults.template = "plotly_white"


# In[27]:


# Violin Plot - Age and Gender
fig = px.violin(data, x='sex', y='age', box=True, points="all", title='Age Distribution by Gender',
                labels={'sex': 'Gender (1 = Male, 0 = Female)', 'age': 'Age'})
fig.show()


# In[28]:


# Bar Plot - Chest Pain Type by Gender
fig = px.bar(data, x='sex', y='cp', title='Chest Pain Type by Gender',
             labels={'sex': 'Gender (1 = Male, 0 = Female)', 'cp': 'Chest Pain Type'})
fig.show()


# In[29]:


# Histogram - Resting Blood Pressure (trestbps) Distribution
fig = px.histogram(data, x='trestbps', nbins=15, title='Resting Blood Pressure (trestbps) Distribution',
                   labels={'trestbps': 'Resting Blood Pressure', 'count': 'Count'})
fig.show()


# In[30]:


# Count Plot - Count of Exercise-induced Angina by Slope
fig = px.histogram(data, x='slope', color='exang', barmode='group',
                   title='Count of Exercise-induced Angina by Slope',
                   labels={'slope': 'Slope', 'exang': 'Exercise-induced Angina',
                           'count': 'Count'}, category_orders={"exang": [0, 1]})
fig.show()


# In[31]:


# Pairplot with Hue - Pairwise Scatter Plots with Target as Hue
fig = px.scatter_matrix(data, dimensions=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], color='target',
                        title='Pairwise Scatter Plots with Target as Hue', symbol='target',
                        labels={'target': 'Target', 'age': 'Age', 'trestbps': 'Resting Blood Pressure',
                                'chol': 'Cholesterol', 'thalach': 'Max Heart Rate (Thalach)',
                                'oldpeak': 'ST Depression (oldpeak)'})
fig.show()


# In[32]:


# Box Plot: Resting Blood Pressure (trestbps) by Chest Pain Type (cp)
import plotly.express as px

fig = px.box(data, x='cp', y='trestbps', title='Resting Blood Pressure (trestbps) by Chest Pain Type (cp)',
             labels={'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure'})
fig.show()


# In[33]:


# Scatter 3D Plot: Age, Cholesterol, and Max Heart Rate (Thalach)
fig = px.scatter_3d(data, x='age', y='chol', z='thalach', color='target', title='Age, Cholesterol, and Max Heart Rate (Thalach)',
                    labels={'age': 'Age', 'chol': 'Cholesterol', 'thalach': 'Max Heart Rate (Thalach)',
                            'target': 'Target'}, symbol='target')
fig.show()


# In[34]:


# Bar Plot: Count of Fasting Blood Sugar (fbs) and Exercise-induced Angina (exang)
fig = px.bar(data, x='fbs', color='exang', barmode='group',
             title='Count of Fasting Blood Sugar (fbs) and Exercise-induced Angina (exang)',
             labels={'fbs': 'Fasting Blood Sugar', 'exang': 'Exercise-induced Angina', 'count': 'Count'})
fig.show()


# In[35]:


# Violin Plot: Cholesterol Distribution by Slope and Exercise-induced Angina
fig = px.violin(data, x='slope', y='chol', box=True, points="all",
                color='exang', title='Cholesterol Distribution by Slope and Exercise-induced Angina',
                labels={'slope': 'Slope', 'chol': 'Cholesterol', 'exang': 'Exercise-induced Angina'})
fig.show()


# In[36]:


# Histogram: Target Distribution
fig = px.histogram(data, x='target', title='Target Distribution',
                   labels={'target': 'Target', 'count': 'Count'})
fig.show()


# In[37]:


# Bar Plot: Count of Chest Pain Type (cp) by Target
fig = px.bar(data, x='cp', color='target', barmode='group',
             title='Count of Chest Pain Type (cp) by Target',
             labels={'cp': 'Chest Pain Type', 'target': 'Target', 'count': 'Count'})
fig.show()


# In[38]:


# Scatter Plot: Age vs. Resting Blood Pressure (trestbps) colored by Gender
fig = px.scatter(data, x='age', y='trestbps', color='sex', title='Age vs. Resting Blood Pressure (trestbps) colored by Gender',
                 labels={'age': 'Age', 'trestbps': 'Resting Blood Pressure', 'sex': 'Gender'},
                 color_discrete_map={1: 'blue', 0: 'pink'})
fig.show()


# In[39]:


# Histogram: Distribution of Max Heart Rate (Thalach) by Target
fig = px.histogram(data, x='thalach', color='target', nbins=15,
                   title='Distribution of Max Heart Rate (Thalach) by Target',
                   labels={'thalach': 'Max Heart Rate (Thalach)', 'count': 'Count', 'target': 'Target'},
                   color_discrete_map={1: 'green', 2: 'red'})
fig.show()


# In[40]:


# Violin Plot: Cholesterol Distribution by Target and Gender
fig = px.violin(data, x='target', y='chol', box=True, points="all", color='sex',
                title='Cholesterol Distribution by Target and Gender',
                labels={'target': 'Target', 'chol': 'Cholesterol', 'sex': 'Gender'},
                color_discrete_map={1: 'purple', 2: 'orange'})
fig.show()


# In[41]:


# Pie Chart: Distribution of Exercise-induced Angina (exang)
fig = px.pie(data, names='exang', title='Distribution of Exercise-induced Angina (exang)',
             labels={'exang': 'Exercise-induced Angina'},
             color_discrete_map={0: 'lightblue', 1: 'darkblue'})
fig.show()


# These visualizations will help us to gain more insights into the relationships and patterns within the data. 
# 
# 
# ### Note : 
# 
# You have to modify the plots according to your specific analysis requirements.

# In[ ]:




