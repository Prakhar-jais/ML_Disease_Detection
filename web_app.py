# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:32:07 2022

@author: pranil jaiswal
"""

import numpy as np
import pickle
import streamlit as st 
import streamlit_option_menu as om



from sklearn.preprocessing import StandardScaler
path1 = r"C:\Users\pranil jaiswal\.spyder-py3\web_app_ML\trained_model.sav"
path2 = r"C:\Users\pranil jaiswal\.spyder-py3\web_app_ML\Heart_disease_Model.sav"
model1 = pickle.load(open(path1,'rb'))
model2 = pickle.load(open(path2,'rb'))

#creating np array# side bar for navigation

with st.sidebar:
    select = om.option_menu('AI Disease Detecting System',['Diabetes Detection','HeartDiseases Detection'],icons=  ['capsule','heart-pulse-fill'],default_index=0)


if(select =='Diabetes Detection'):
    st.title('ML Diabetes Detection')
    
    def dib_pred(input_data):
        
        
        input_array = np.asarray(input_data)
        #reshae the array
        input_data_reshaped = input_array.reshape(1,-1)
        
       
        
    
        prediction = model1.predict(input_data_reshaped)
        print(prediction)
        
        if(prediction == 0):
            
            return 'person is not  diabetic'
        else:
            
            return'the person is diabetic'
            
    
    Pregnancies = st.text_input('No of Pregnancies')
    Glucose = st.text_input('Glucose Value')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction =st.text_input('Diabetes Pedigree_Function')
    Age = st.text_input('Person Age')
      
    
    diagnosis = '' 
    
    if st.button('Diabetes Diagnosis'):
        diagnosis = dib_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
    # prediction code 
    
    
    # create button for prediction
    
if(select =='HeartDiseases Detection'):
    
    st.title('ML HeartDiseases Detection ')
    


    
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Cp')
    trestbps = st.text_input('Trestbps')
    chol = st.text_input('Chol')
    fbs = st.text_input('Fbs')
    restecg = st.text_input('restecg')
    thalach = st.text_input('Thalach')
    exang = st.text_input('Exang')
    oldpeak = st.text_input('Oldpeak')
    slope = st.text_input('Slope')
    ca = st.text_input('Ca')
    thal = st.text_input('Thal')
   
    def heart_pred(input_data):
        
        input_array = np.asarray(input_data)
        #reshae the array
        input_data_reshaped = input_array.reshape(1,-1)
        
       
        
    
        prediction = model2.predict(input_data_reshaped)
        print(prediction)
        
        if(prediction == 0):
            
            return 'person has no heart disease'
        else:
            
            return'the person has heart disease'
        
    diagnose = '' 
    
    if st.button('Heart Disease Diagnosis'):
        diagnose = heart_pred([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
    
    st.success(diagnose)
    
    











        
