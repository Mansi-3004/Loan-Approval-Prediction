#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('LoanApprovalPrediction (1).csv')


# In[2]:


data


# In[3]:


data.info()


# In[4]:


data.isnull().sum() #missing values


# In[5]:


data.columns


# In[6]:


data.duplicated().sum()


# In[7]:


# dependent -> 12 missing values
data['Dependents']


# In[8]:


counts = data['Property_Area'].value_counts(dropna=False) # dropna = false ensures nan is counted 
counts


# In[9]:


# Identify categorical columns
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

# Set figure size
plt.figure(figsize=(18, 36))
index = 1

# Plot bar charts for each categorical column
for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11, 4, index)  # Adjust grid size based on number of categories
    plt.xticks(rotation=90)
    sns.barplot(x=y.index, y=y.values)  # Corrected x and y values
    plt.title(col)  
    index += 1

plt.tight_layout()  
plt.show()


# In[10]:


from sklearn import preprocessing

# Identify categorical columns before encoding
categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

# Dictionary to store label mappings
label_mappings = {}

# Apply Label Encoding and store mappings
for col in categorical_cols:
    label_encoder = preprocessing.LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])  # Convert categories to numbers
    label_mappings[col] = {index: label for index, label in enumerate(label_encoder.classes_)}  # Save mapping


for col, mapping in label_mappings.items():
    print(f"Label Mapping for {col}: {mapping}")


# In[11]:


# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)


# ## EDA

# In[12]:


plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), 
            cmap='RdYlGn', 
            fmt='.2f',      
            linewidths=2,   
            annot=True,     
            center=0)       

plt.show()


# In[13]:


for col in data.columns:
  data[col] = data[col].fillna(data[col].mean()) 
  
data.isna().sum()


# ## Splitting Data

# In[14]:


from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
print(X.shape,Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.4,
                                                    random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[15]:


#Algorithm to use 
#logistic regression,decision tree,random forest,support vector machine,knn


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

model = { 'Logistic Regression':LogisticRegression(), 'Decision Tree': DecisionTreeClassifier() ,
          'Random Forest' :RandomForestClassifier(), "SVC" :SVC(), "KNN":KNeighborsClassifier()}
for name, model in model.items():
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f'{name} Accuracy: {accuracy*100:.2f}%')


# In[17]:


#using LR and RF

logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)


print("Enter values for the following features:")
gender = int(input("Gender (1 for Male, 0 for Female): "))
married = int(input("Married (1 for Yes, 0 for No): "))
dependents = int(input("Number of Dependents (0, 1, 2, 3): "))
education = int(input("Education (1 for Graduate, 0 for Not Graduate): "))
self_employed = int(input("Self Employed (1 for Yes, 0 for No): "))
applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_amount_term = float(input("Loan Amount Term (in months): "))
credit_history = int(input("Credit History (1 for Yes, 0 for No): "))
property_area = int(input("Property Area (0 for Urban, 1 for Semiurban, 2 for Rural): "))


user_input = [[gender, married, dependents, education, self_employed, 
               applicant_income, coapplicant_income, loan_amount, 
               loan_amount_term, credit_history, property_area]]

# predictions
logistic_prediction = logistic_model.predict(user_input)
rf_prediction = rf_model.predict(user_input)

print(f"\nLogistic Regression Prediction: {'Approved' if logistic_prediction[0] == 1 else 'Rejected'}")
print(f"Random Forest Prediction: {'Approved' if rf_prediction[0] == 1 else 'Rejected'}")


# In[ ]:


pip install gradio


# In[ ]:


import gradio as gr

def predict_loan(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_amount_term, credit_history, property_area):
    
    user_input = [[gender, married, dependents, education, self_employed,
                   applicant_income, coapplicant_income, loan_amount,
                   loan_amount_term, credit_history, property_area]]
    
    lr_pred = logistic_model.predict(user_input)[0]
    rf_pred = rf_model.predict(user_input)[0]

    return (
        "Approved" if lr_pred == 1 else "Rejected",
        "Approved" if rf_pred == 1 else "Rejected"
    )

iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio([1, 0], label="Gender (1 = Male, 0 = Female)"),
        gr.Radio([1, 0], label="Married (1 = Yes, 0 = No)"),
        gr.Slider(0, 3, step=1, label="Number of Dependents"),
        gr.Radio([1, 0], label="Education (1 = Graduate, 0 = Not Graduate)"),
        gr.Radio([1, 0], label="Self Employed (1 = Yes, 0 = No)"),
        gr.Number(label="Applicant Income in thousands"),
        gr.Number(label="Coapplicant Income in thousands"),
        gr.Number(label="Loan Amount in thounsands"),
        gr.Number(label="Loan Amount Term (months)"),
        gr.Radio([1, 0], label="Credit History (1 = Good, 0 = Bad)"),
        gr.Radio([0, 1, 2], label="Property Area (0 = Urban, 1 = Semiurban, 2 = Rural)")
    ],
    outputs=[
        gr.Textbox(label="Logistic Regression Prediction"),
        gr.Textbox(label="Random Forest Prediction")
    ],
    title="Loan Approval Prediction",
    description="Enter your details below to see if your loan might be approved based on two machine learning models."
)

iface.launch()


# In[ ]:




