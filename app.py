import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pickle

st.title("Predictoma ")
st.markdown("Welcome to Predictoma- A machine learning based prognosis prediction tool for Glioblastoma patients. Please input the following values to get a prediction: ")

if st.checkbox('Demographics data'):
    Age = st.number_input('Age at diagnosis (in years)',min_value=1,max_value=120,value =10,step=1)
    Sex = st.multiselect(
      'Gender of the patient', 
       ('Male', 'Female'))
     
if st.checkbox('Medical history'):
   
   Karnofsky_Performance_Score = st.number_input(label ='Karnofsky Performance Score',min_value=20, max_value=100,step=10)

   Longest_Dimension = st.number_input('Longest Dimension of tumor (in cm)',min_value=0.3,max_value=3.0)

   Shortest_Dimension = st.number_input('Shortest Dimension of tumor (in cm)',min_value=0.025,max_value=1.00)

if st.checkbox('Genomics information'):
      Fraction_Genome_Altered = st.number_input('Fraction Genome Altered',min_value=0.00,max_value= 0.75,value =0.00)
      PTENcna = st.number_input('for PTEN',min_value=-2.00,max_value= 2.00)
      EGFRcna = st.number_input('for EGFR',min_value=-1.00,max_value= 2.00)
      TP53cna = st.number_input('for TP53',min_value=-2.00,max_value= 2.00)                
      IDH1cna = st.number_input('for IDH1',min_value=-2.00,max_value= 2.00)

    
      if Age>0: 
            data = [Age,Karnofsky_Performance_Score,Longest_Dimension, Shortest_Dimension,Fraction_Genome_Altered,PTENcna, EGFRcna, TP53cna, IDH1cna]
   
            pkl_filename = 'Lassoglioma.pkl'
            with open(pkl_filename, 'rb+') as file:
                 pickle_model = pickle.load(file)

            prediction = pickle_model.predict(np.array(data).reshape(1,-1))[0]
            mae1_pred = prediction - 5.9
            mae2_pred = prediction + 5.9

            st.write ("According to the linear regression algorithm\n")
            if mae1_pred>0:
               st.write (f"The predicted Survival is {mae1_pred:0.1f} - {mae2_pred:0.1f} months")
            else:
               st.write (f"The predicted Survival is approximately {mae2_pred:0.1f} months")

        



st.info('This tool is to help make the doctor and patient an informed decision regarding treatment strategy and is a purely informational message. The predictions are made using machine learning algorithms.')