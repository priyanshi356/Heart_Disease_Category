import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('practice_test_pr.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('CLASSIFICATION DATASET.csv')
X= dataset.iloc[:, :-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
imputer = imputer.fit(X[:, np.r_[1:5, 7:13]])  
X[:, np.r_[1:5, 7:13]]= imputer.transform(X[:, np.r_[1:5, 7:13]])

#GENDER
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
imputer = imputer.fit(X[:, 5:6]) 
X[:, 5:6]= imputer.transform(X[:, 5:6])

#GEOGRAPHY
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Spain', verbose=1, copy=True)  
imputer = imputer.fit(X[:, 6:7]) 
X[:, 6:7]= imputer.transform(X[:, 6:7])

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])

labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

def predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,
                                          thalach,exang,oldpeak,slope,ca,thal):
  output= model.predict(sc_X.transform([[age,cp,trestbps,chol,fbs,Gender,Geography,restecg,
                                          thalach,exang,oldpeak,slope,ca,thal]]))
  print("heart disease", output)
  if output==[0]:
    print("it is heart disease category 0")
  elif output==[1]:
    print("it is heart disease category 1")
  elif output==[2]:
    print("it is heart disease category 2")
  elif output==[3]:
    print("it is heart disease category 3")
  else:
    print("it is heart disease category 4")
  print(output)
  return output
  
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart Diesease using decision tree classifier")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    age = st.number_input('Insert a Age')
    cp = st.number_input('Insert cp')
    trestbps = st.number_input('Insert trestbps')
    chol = st.number_input('Insert chol')
    fbs = st.number_input('Insert fbs')
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Geography = st.number_input('Insert Geography Spain:1 France:0')
    restecg = st.number_input('Insert restecg')
    thalach = st.number_input('Insert thalach')
    oldpeak = st.number_input('Insert oldpeak')
    exang = st.number_input('Insert exang')
    slope = st.number_input('Insert slope')
    ca = st.number_input('Insert ca')
    thal = st.number_input('Insert thal')
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,
                                          thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted category {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Priyanshi Agrawal")
      st.subheader("Student, Department of Computer Engineering")

if __name__=='__main__':
  main()
