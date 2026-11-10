import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Credit Risk Studio", page_icon="📊", layout="wide")
st.title("📊 Credit Risk Studio")
st.caption("Educational credit-risk classification demo. Not a lending decision system.")
DATA=Path(__file__).parent/'data/credit_applicants.csv'
df=pd.read_csv(DATA)
features=['age','monthly_income','monthly_debt','late_payments_12m','credit_history_years']
X=df[features]; y=df['credit_risk']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('classifier',RandomForestClassifier(n_estimators=250,max_depth=6,random_state=42,class_weight='balanced'))])
model.fit(X_train,y_train); pred=model.predict(X_test)
a,b,c=st.columns(3); a.metric('Applicants',len(df)); b.metric('Test accuracy',f'{accuracy_score(y_test,pred):.1%}'); c.metric('High-risk share',f'{(y=="High Risk").mean():.1%}')
st.subheader('Applicant data')
st.dataframe(df,use_container_width=True,hide_index=True)
st.subheader('Try a sample applicant')
cols=st.columns(5)
age=cols[0].number_input('Age',18,90,32); income=cols[1].number_input('Monthly income',1000,1000000,65000,1000); debt=cols[2].number_input('Monthly debt',0,1000000,18000,500); late=cols[3].number_input('Late payments',0,30,1); years=cols[4].number_input('Credit history years',0,60,5)
if st.button('Estimate risk',type='primary'):
    result=model.predict(pd.DataFrame([[age,income,debt,late,years]],columns=features))[0]
    st.success(f'Model estimate: {result}') if result=='Low Risk' else st.warning(f'Model estimate: {result}')
st.subheader('Evaluation')
st.write('Confusion matrix'); st.dataframe(pd.DataFrame(confusion_matrix(y_test,pred),index=['Actual High Risk','Actual Low Risk'],columns=['Predicted High Risk','Predicted Low Risk']))
st.code(classification_report(y_test,pred),language='text')
st.info('The included 100-row dataset is synthetic. It is designed for code practice and must not be used for real credit decisions.')
