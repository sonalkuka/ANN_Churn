import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

model=tf.keras.models.load_model('model.h5')

with open('scalar.pkl','rb') as file:
    scaler=pickle.load(file)
with open('one_hot_encoding_geo.pkl','rb') as file:
    ohe=pickle.load(file)
with open('Lable_encoder.pkl','rb') as file:
    le=pickle.load(file)

st.title("Customer Churn Prediction")

credit_score=st.number_input("Credit Score")
Geography=st.selectbox("Select Country",ohe.categories_[0])
Gender=st.selectbox("Gender",le.classes_)
Age=st.slider("Age",18,90)
Tenure=st.slider("Tenure",1,10)
Balance=st.number_input("Balance")
NumOfProducts=st.slider("Number of products",1,20)
HasCrCard=st.selectbox("Has Card",[0,1])
IsActiveMember=st.selectbox("Is active member or not",[1,0])
EstimatedSalary=st.number_input("Salary Estimated")

input_data=pd.DataFrame({
    'CreditScore':[credit_score], 
    'Gender':[le.transform([Gender])[0]],
    'Age':[Age], 
    'Tenure':[Tenure],
    'Balance':[Balance], 
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard], 
    'IsActiveMember':[IsActiveMember], 
    'EstimatedSalary':[EstimatedSalary], 
    
}

)

geo_df=ohe.transform([[Geography]])
df_geo=pd.DataFrame(geo_df.toarray(),columns=ohe.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),df_geo],axis=1)


input_data=scaler.transform(input_data)

prediction=model.predict(input_data)
prediction_proba=prediction[0][0]

st.write(f"Probability: {prediction_proba:.2f}")

if prediction_proba>0.5:
    st.write("Customer likely to churn")
else:
    st.write("Customer not likely to churn")
