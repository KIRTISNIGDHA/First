import streamlit as st
import numpy as np
import pandas as pd

st.title("Prediction of Glioblastoma patient survival duration using  Machine Learning Regression Algorithm")

Age = st.number_input('Age')

Sex = st.selectbox("Sex", ('Male', 'Female'))


Disease_Free_Status = st.selectbox("Disease Status", ('Disease Free', 'Recurred/Progressed'))


Disease_free_months = st.number_input('Disease Free months')

Karnofsky_Performance_Score = st.number_input('Karnofsky Performance Score')

Longest_Dimension = st.number_input('Longest Dimension of tumor')

Fraction_Genome_Altered = st.number_input('Fraction Genome Altered')

PTENcna = st.number_input('Copy Number Alteration for PTEN')

EGFRcna = st.number_input('Copy Number Alteration for EGFR')

TP53cna = st.number_input('Copy Number Alteration for TP53')

IDH1cna = st.number_input('Copy Number Alteration for IDH1')

PTEN_Methylation = st.selectbox("PTEN Methylation", ('HM27', 'HM450','EPIC'))
if PTEN_Methylation == 'HM27':
   PTENHM27 = st.number_input('Input value')
elif PTEN_Methylation == 'HM450':
    PTENHM450 = st.number_input('Input value')
else:
   PTEN_EPIC = st.number_input('Input value')

   
EGFR_Methylation = st.selectbox("EGFR Methylation", ('HM27', 'HM450','EPIC'))

if EGFR__Methylation == 'HM27':
   EGFR_HM27 = st.number_input('Input value')
elif EGFR__Methylation == 'HM450':
    EGFR_HM450 = st.number_input('Input value')
else:
   EGFR__EPIC = st.number_input('Input value')


TP53_EPIC = st.number_input('Methylation of TP53')

MGMTHM27 = st.number_input('Methylation of MGMT')

num_data = [Age,Disease_free_months,Fraction_Genome_Altered]
cat_data = [Sex,Disease_Free_Status,Gene_Expression_Subtype,GCIMP_Methylation,methylation_Status,IDH1_Mutation,MGMT_Status]




if Age >0 :
   if Sex=="Male":
      st.write ("The predicted Survival is 5 months")
   else:
      st.write ("The predicted Survival is 7 months")


st.info('This is a purely informational message')

