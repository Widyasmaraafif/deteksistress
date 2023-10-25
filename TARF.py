#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #Linear Algebra
import pandas as pd #Data Processing
import matplotlib.pyplot as plt #Visualisasi
import seaborn as sns #Visualisasi dalam angka
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# In[2]:


#import dataset
data = 'Sleep_health_and_lifestyle_dataset.csv'
df = pd.read_csv(data)


# In[3]:


#Melihat dimensi dataset
df.shape


# In[4]:


#Tampilkan dataset
df.head()


# In[5]:


df = df.drop(columns = 'Person ID')


# In[6]:


df['Blood Pressure'] = df['Blood Pressure'].replace({'/':'.'}, regex=True).astype('float')


# In[7]:


#Cek Missing Value
df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# # Membagi variabel kategorikal dan variabel numerik

# In[10]:


cat_col = df.select_dtypes('object')
print('Terdapat', cat_col.shape[1], 'kolom kategorikal')


# In[11]:


num_col = df.select_dtypes('number')
print('Terdapat', num_col.shape[1], 'kolom numerik')


# # Analisa variabel kategorikal

# In[12]:


fig = plt.figure(figsize=(15, 10))

for i, var in enumerate(cat_col):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=var, data= cat_col)
    plt.xticks(rotation=90)
    
plt.show()


# In[13]:


df['BMI Category'].value_counts()


# In[14]:


df['BMI Category'] = df['BMI Category'].replace({'Obese':'Overweight','Normal Weight':'Normal'})
df['BMI Category'].value_counts()


# # Analisa variabel numerik

# In[15]:


fig = plt.figure(figsize=(10, 10))

for i, var in enumerate(num_col):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=var, data=num_col)
    plt.xticks(rotation=0)

plt.show()


# In[16]:


sns.heatmap(data=df.corr(),annot=True)


# In[17]:


sns.histplot(num_col,x='Heart Rate')


# # Target analisis

# In[18]:


TARGET = 'Stress Level'


# In[19]:


df[TARGET].value_counts()


# In[20]:


batas_bin = [1,3,5,7,10]
kategori = [0, 1, 2, 3]
df[TARGET] = pd .cut(df[TARGET], bins=batas_bin, labels=kategori)


# In[21]:


df[TARGET].value_counts()


# In[22]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, hue='Gender', x=TARGET)
plt.title('Tingkat Stress Level berdasarkan Jenis Kelamin', fontsize=14)
plt.show()


# Ini menunjukkan bahwa 

# In[23]:


plt.figure(figsize=(6, 4))
sns.histplot(data=df, hue=TARGET, y='Age',
             multiple = "dodge")
plt.title('Distribution of Sleep Disorder by Age', fontsize=14)
plt.show()


# In[24]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, hue=TARGET, y='Occupation')
plt.title('Distribution of Sleep Disorder by Occupation', fontsize=14)
plt.show()


# In[25]:


df.columns


# In[52]:


df['Gender'].value_counts()


# # Preprocessing Categorical Variables

# In[27]:


df.select_dtypes('object').head(2)


# In[28]:


label_encoder = LabelEncoder()


# In[29]:


df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])


# In[30]:


df.head()


# In[31]:


df['Occupation'].value_counts()


# # Train test split

# In[32]:


X = df.drop(columns=TARGET)
y = df[TARGET]


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


y_train


# In[35]:


scaler = StandardScaler()


# In[36]:


rf = RandomForestClassifier(
            n_estimators = 100,
            criterion    = 'gini',
            max_depth    = None,
            max_features = 'auto',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )


# In[37]:


random_forest = Pipeline(steps=[
    ('scaler', scaler),
    ('model', rf)

])


# In[38]:


random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print('The model gives' ,accuracy_score(y_test, y_pred), 'accuracy on y_test')


# In[39]:


y_train.value_counts()


# In[40]:


y_test_pred = pd.DataFrame(random_forest.predict(X_test), index=y_test.index, columns=[TARGET])


# In[41]:


results_df = y_test_pred.join(y_test, lsuffix='_Prediction', rsuffix='_y_test', how='inner')

results_df.head(30)


# In[42]:


model = random_forest.named_steps['model'] # We cannot use "feature_importances_" in the pipeline, what we do is save the fit in a variable.

feature_score=pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_score,y=feature_score.index)
plt.title('Visualization Feature Importance ')
plt.show()


# In[43]:


print(classification_report(y_test, y_pred))


# In[44]:


import pickle


# In[45]:


filename = 'stress_level_trained.sav'
pickle.dump(random_forest,open(filename,'wb'))


# In[46]:


loaded_model = pickle.load(open('stress_level_trained.sav','rb'))


# In[47]:


# Evaluating

input_data = (1,27,9, 6.1,6,42,1,126.83,77,4200,1) #300
#changing the input data into numpy array
id_np_array = np.asarray(input_data)
id_reshaped = id_np_array.reshape(1,-1)

prediction = random_forest.predict(id_reshaped)
print(prediction)

if(prediction[0]==0):
    print("Stress Level: NORMAL")
elif(prediction[0]==1):
    print("Stress Level: LOW")
elif (prediction[0]==2):
    print("Stress Level: MEDIUM")
else:
    print("Stress Level: HIGH")
    
# input_data = (21.38,90.38,128) #100
# #changing the input data into numpy array
# id_np_array = np.asarray(input_data)
# id_reshaped = id_np_array.reshape(1,-1)

# prediction = random_forest.predict(id_reshaped)
# print(prediction)

# if(prediction[0]==0):
#     print("Stress Level: LOW")
# elif(prediction[0]==1):
#     print("Stress Level: MEDIUM")
# else:
#     print("Stress Level: HIGH")
    
# input_data = (25.41,94.41,167) #200
# #changing the input data into numpy array
# id_np_array = np.asarray(input_data)
# id_reshaped = id_np_array.reshape(1,-1)

# prediction = random_forest.predict(id_reshaped)
# print(prediction)

# if(prediction[0]==0):
#     print("Stress Level: LOW")
# elif(prediction[0]==1):
#     print("Stress Level: MEDIUM")
# else:
#     print("Stress Level: HIGH")


# In[58]:


import numpy as np
import pickle
import streamlit as st  

# Loading the trained model
loaded_model = pickle.load(open('stress_level_trained.sav','rb'))
# Replace path over stress_trained.sav

def stresslevel_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = loaded_model.predict(id_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "Stress Level: NORMAL"
    elif(prediction[0]==1):
        return "Stress Level: LOW"
    elif (prediction[0]==2):
        return "Stress Level: MEDIUM"
    else:
        return "Stress Level: HIGH"
    
def main():
    
    st.title('STRESS LEVEL PREDICTION WEB APP')
    
    Ocp = ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse', 'Sales Represntative', 'Salesperson', 'Scientist', 'Software Engineer', 'Teacher', 'Other']
    
    Gender = st.selectbox('Gender',['Male', 'Female'])
    Age = st.text_input('Age')
    Occupation = st.selectbox('Occupation',Ocp)
    SleepDuration = st.text_input('Sleep Duration')
    QualityofSleep = st.text_input('Quality of Sleep')
    PALevel = st.text_input('Physical Activity Level')
    BMICategory = st.selectbox('BMI Category',['Normal', 'Overweight'])
    BloodPreasure = st.text_input('Blood Preasure')
    HeartRate = st.text_input('Heart Rate')
    DailySteps = st.text_input('Daily Step')
    SleepDisorder = st.selectbox('Sleep Disorder',['Tidak', 'Insomnia', 'Sleep Apnea'])
    
    if (Gender == "Female"):
        Gender = 0
    elif (Gender == "Male"):
        Gender = 1
        
    if (Occupation == "Accountant"):
        Occupation = 0
    elif (Occupation == "Doctor"):
        Occupation = 1
    elif (Occupation == "Engineer"):
        Occupation = 2
    elif (Occupation == "Lawyer"):
        Occupation = 3
    elif (Occupation == "Manager"):
        Occupation = 4
    elif (Occupation == "Nurse"):
        Occupation = 5
    elif (Occupation == "Sales Representative"):
        Occupation = 6
    elif (Occupation == "Salesperson"):
        Occupation = 7
    elif (Occupation == "Scientist"):
        Occupation = 8
    elif (Occupation == "Software Engineer"):
        Occupation = 9
    elif (Occupation == "Teacher"):
        Occupation = 10
        
    
    if (BMICategory == "Normal"):
        BMICategory = 0
    elif (BMICategory == "Overweight"):
        BMICategory = 1
        
    if (SleepDisorder == "Insomnia"):
        SleepDisorder = 0
    elif (SleepDisorder == "Tidak"):
        SleepDisorder = 1
    elif (SleepDisorder == "Sleep Apnea"):
        SleepDisorder = 2 
        
    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = stresslevel_prediction([Gender,Age,Occupation,SleepDuration,QualityofSleep,PALevel,BMICategory,BloodPreasure,HeartRate,DailySteps,SleepDisorder])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()


# # 11. Feature Scaling

# In[ ]:


# cols = X_train.columns


# In[ ]:


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)


# In[ ]:


# X_train = pd.DataFrame(X_train, columns=[cols])


# In[ ]:


# X_test = pd.DataFrame(X_test, columns=[cols])


# In[ ]:


# X_train.describe()


# In[ ]:


# # import SVC classifier
# from sklearn.svm import SVC


# # import metrics to compute accuracy
# from sklearn.metrics import accuracy_score


# # instantiate classifier with default hyperparameters
# svc=SVC() 


# # fit classifier to training set
# svc.fit(X_train,y_train)


# # make predictions on test set
# y_pred=svc.predict(X_test)


# # compute and print accuracy score
# print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


# import pickle


# In[ ]:


# filename = 'stress_level_svm.sav'
# pickle.dump(svc,open(filename,'wb'))


# In[ ]:


# loaded_model = pickle.load(open('stress_level_svm.sav','rb'))


# In[ ]:


# # Evaluating

# input_data = (1,27,9, 6.1,6,42,1,126.83,77,4200,1) #300
# #changing the input data into numpy array
# id_np_array = np.asarray(input_data)
# id_reshaped = id_np_array.reshape(1,-1)

# prediction = svc.predict(id_reshaped)
# print(prediction)

# if(prediction[0]==0):
#     print("Stress Level: NORMAL")
# elif(prediction[0]==1):
#     print("Stress Level: LOW")
# elif (prediction[0]==2):
#     print("Stress Level: MEDIUM")
# else:
#     print("Stress Level: HIGH")


# In[ ]:




