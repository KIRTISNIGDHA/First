import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle

st.title("Predictoma ")
st.markdown("Welcome to Predictoma- A machine learning based prognosis prediction tool for Glioblastoma patients. Please input the following values to get a prediction: ")

if st.checkbox('Demographics data'):
    Age = st.number_input('Age at diagnosis (in years)',min_value=1,max_value=150,value =1,step=1)
    Sex = st.multiselect(
      'Gender of the patient', 
       ('Male', 'Female'))
     
if st.checkbox('Medical history'):
   Disease_Free_Status = st.multiselect(
                      "What is the current disease Status", 
                       ('Disease Free', 'Recurred/Progressed'))

   Disease_free_months = st.number_input('Disease Free (in months)')

   Karnofsky_Performance_Score = st.number_input(label ='Karnofsky Performance Score',min_value=0, max_value=100, value =0,step=10)



if st.checkbox('Characteristics of the tumor'):
   if st.checkbox('Physical Characteristics of the tumor'):
      Longest_Dimension = st.number_input('Longest Dimension of tumor (in cm)',value = 0.00)

      Shortest_Dimension = st.number_input('Shortest Dimension of tumor (in cm)',value = 0.00)

   if st.checkbox('Genomics information'):
      Mutation_Count = st.number_input('Mutation Count',value = 0)
      Fraction_Genome_Altered = st.number_input('Fraction Genome Altered',value = 0.0000)
      if st.checkbox('Copy Number Alteration'):
         PTENcna = st.number_input('for PTEN',min_value=-2,max_value= 2,value = 0,step=1)

         EGFRcna = st.number_input('for EGFR',min_value=-2,max_value= 2,value = 0,step=1)

         TP53cna = st.number_input('for TP53',min_value=-2,max_value= 2,value = 0,step=1)

         IDH1cna = st.number_input('for IDH1',min_value=-2,max_value= 2,value = 0,step=1)

      if st.checkbox('Methylation'):
         PTEN_Methylation = st.multiselect("PTEN Methylation", ('HM27', 'HM450','EPIC'))
         if PTEN_Methylation == 'HM27':
            PTENHM27 = st.number_input('Input value',value = 0.00)
            PTENHM450 = 0
            PTEN_EPIC =0
         elif PTEN_Methylation == 'HM450':
            PTENHM450 = st.number_input('Input value',value = 0.00)
            PTENHM27 =0
            PTEN_EPIC =0
         else:
            PTEN_EPIC = st.number_input('Input value',value = 0.00)
            PTENHM450 = 0
            PTENHM27 =0
   
         EGFR_Methylation = st.multiselect( "EGFR Methylation", ('HM27', 'HM450','EPIC'))
         if EGFR_Methylation == 'HM27':
            EGFRHM27 = st.number_input('Input EGFR methylation value',value = 0.00)
            EGFRHM450 = 0
            EGFR_EPIC =0
         elif EGFR_Methylation =='HM450':
            EGFR_HM450 = st.number_input('Input EGFR methylation value',value = 0.00)
            EGFRHM27 = 0
            EGFR_EPIC =0
         else:
            EGFR_EPIC = st.number_input('Input EGFR methylation value',value = 0.00)
            EGFRHM450 = 0
            EGFRHM27 =0


         TP53_EPIC = st.number_input('Methylation of TP53',value = 0.00)

         MGMTHM27 = st.number_input('Methylation of MGMT',value = 0.00)

         if Age>0: 
            data = [Age, Mutation_Count, Disease_free_months,Fraction_Genome_Altered, Longest_Dimension, Karnofsky_Performance_Score,
       PTENcna, EGFRcna, TP53cna, IDH1cna, PTENHM27, EGFRHM27, 
                  MGMTHM27, PTENHM450, EGFRHM450, PTEN_EPIC, EGFR_EPIC, TP53_EPIC]
   
            pkl_filename = 'SVRglioma.pkl'
            with open(pkl_filename, 'rb+') as file:
                 pickle_model = pickle.load(file)

            prediction = pickle_model.predict(np.array(data).reshape(1,-1))[0]
            mae1_pred = prediction - 8
            mae2_pred = prediction + 8

            st.write ("According to the support vector regression model\n")
            if mae1_pred>0:
               st.write (f"The predicted Survival is {mae1_pred:0.2f} - {mae2_pred:0.2f} months")
            else:
               st.write (f"The predicted Survival approximately {mae2_pred:0.2f} months")

            pkl_filename1 = 'rfglioma.pkl'
            with open(pkl_filename1, 'rb+') as file:
                 pickle_rfmodel = pickle.load(file)

            prediction_rf = pickle_rfmodel.predict(np.array(data).reshape(1,-1))[0]
            mae1_rfpred = prediction_rf - 9
            mae2_rfpred = prediction_rf + 9

            st.write ("According to the random forest regression model\n")
            if mae1_rfpred>0:
               st.write (f"The predicted Survival is {mae1_rfpred:.2f} - {mae2_rfpred:.2f} months.")
            else:
               st.write (f"The predicted Survival is approximately {mae2_rfpred:0.2f} months.")




st.info('This tool is to help make the doctor and patient an informed decision regarding treatment strategy and is a purely informational message. The predictions are made using machine learning algorithms.')