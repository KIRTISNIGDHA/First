import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle

st.title("Prediction of Glioblastoma patient survival duration using  Machine Learning Regression Algorithm")

Age = st.number_input('Age')

Sex = st.selectbox("Sex", ('Male', 'Female'))


Disease_Free_Status = st.selectbox("Disease Status", ('Disease Free', 'Recurred/Progressed'))


Disease_free_months = st.number_input('Disease Free months')

Karnofsky_Performance_Score = st.number_input('Karnofsky Performance Score')

Longest_Dimension = st.number_input('Longest Dimension of tumor')

Fraction_Genome_Altered = st.number_input('Fraction Genome Altered')
Mutation_Count = st.number_input('Input Mutation Count')

PTENcna = st.number_input('Copy Number Alteration for PTEN')

EGFRcna = st.number_input('Copy Number Alteration for EGFR')

TP53cna = st.number_input('Copy Number Alteration for TP53')


IDH1cna = st.number_input('Copy Number Alteration for IDH1')

PTEN_Methylation = st.selectbox("PTEN Methylation", ('HM27', 'HM450','EPIC'))
if PTEN_Methylation == 'HM27':
   PTENHM27 = st.number_input('Input value')
   PTENHM450 = 0
   PTEN_EPIC =0
elif PTEN_Methylation == 'HM450':
   PTENHM450 = st.number_input('Input value')
   PTENHM27 =0
   PTEN_EPIC =0
else:
   PTEN_EPIC = st.number_input('Input value')
   PTENHM450 = 0
   PTENHM27 =0
   
EGFR_Methylation = st.selectbox("EGFR Methylation", ('HM27', 'HM450','EPIC'))
if EGFR_Methylation == 'HM27':
   EGFRHM27 = st.number_input('Input EGFR methylation value')
   EGFRHM450 = 0
   EGFR_EPIC =0
elif EGFR_Methylation =='HM450':
   EGFR_HM450 = st.number_input('Input EGFR methylation value')
   EGFRHM27 = 0
   EGFR_EPIC =0
else:
   EGFR_EPIC = st.number_input('Input EGFR methylation value')
   EGFRHM450 = 0
   EGFRHM27 =0


TP53_EPIC = st.number_input('Methylation of TP53')

MGMTHM27 = st.number_input('Methylation of MGMT')

data = [Age, Mutation_Count, Disease_free_months,Fraction_Genome_Altered, Longest_Dimension, Karnofsky_Performance_Score,
       PTENcna, EGFRcna, TP53cna, IDH1cna, PTENHM27, EGFRHM27, 
                  MGMTHM27, PTENHM450, EGFRHM450, PTEN_EPIC, EGFR_EPIC, TP53_EPIC]


pkl_filename = 'SVRglioma.pkl'
with open(pkl_filename, 'rb+') as file:
     pickle_model = pickle.load(file)


prediction = pickle_model.predict(np.array(data).reshape(1,-1))[0]

mae1_pred = prediction - 8
mae2_pred = prediction + 8

if Age>0 :
   st.write ("According to the support vector regression model\n")
   if mae1_pred>0:
      st.write ("The predicted Survival is ", mae1_pred ,'-',mae2_pred ,"months")
   else:
      st.write ("The predicted Survival approximately",mae2_pred ,"months")


pkl_filename1 = 'rfglioma.pkl'
with open(pkl_filename1, 'rb+') as file:
     pickle_rfmodel = pickle.load(file)

prediction_rf = pickle_rfmodel.predict(np.array(data).reshape(1,-1))[0]

mae1_rfpred = prediction_rf - 9
mae2_rfpred = prediction_rf + 9


if Age>0 :
   st.write ("According to the random forest regression model\n")

   if mae1_rfpred>0:
      st.write ("The predicted Survival is ", mae1_rfpred ,'-',mae2_rfpred ,"months")
   else:
      st.write ("The predicted Survival approximately",mae2_rfpred ,"months")

st.info('This is a purely informational message')

