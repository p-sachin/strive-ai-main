import streamlit as st
import pandas as pd
import numpy as np
from streamlit.proto.Markdown_pb2 import Markdown
from without_aug import *
from sklearn.ensemble import AdaBoostClassifier
from PIL import Image

sb = st.sidebar
menu = sb.selectbox("Navigation", ('Home', 'Diagnostics'))
if menu == "Home":
    st.header('Team Robert Koch')
    st.markdown(
        '''<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">''',
        unsafe_allow_html=True)
    st.markdown('''
    <div class="list-group">
  <a href="#" class="list-group-item list-group-item-action active" aria-current="true">
    Team members
  </a>
  <a href="#" class="list-group-item list-group-item-action">Sachin</a>
  <a href="#" class="list-group-item list-group-item-action">Abdullah</a>
  <a href="#" class="list-group-item list-group-item-action">Saidalikhon</a>

</div>''', unsafe_allow_html=True)
elif menu == 'Diagnostics':

    ######################################################################################################
    #################################     H E A R T   D I S E A S E   ####################################
    ######################################################################################################

    sb.subheader('insert patient data')
    age = sb.number_input('Age of the person:', min_value=1, max_value=100, value=40)
    sex_text = sb.selectbox('Gender of the person:', ('male', 'female'))
    if sex_text == 'male':
        sex = 1
    elif sex_text == 'female':
        sex = 0
    cp = sb.selectbox('Chest Pain type chest pain type', (0, 1, 2, 3))
    trtbps = sb.number_input('resting blood pressure (in mm Hg)', min_value=50, max_value=210, value=120)
    chol = sb.number_input('cholestoral in mg/dl fetched via BMI sensor', min_value=100, max_value=700, value=200)
    fbs = sb.selectbox('(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)', (0, 1))
    restecg = sb.selectbox('resting electrocardiographic results', (0, 1, 2))
    thalachh = sb.number_input('maximum heart rate achieved', min_value=50, max_value=300, value=120)
    exng = sb.selectbox('exercise induced angina (1 = yes; 0 = no)', (1, 0))
    oldpeak = sb.number_input('Previous peak', min_value=0.0, max_value=7.0, value=0.0, step=0.1)
    slp = sb.selectbox('Slope', (0, 1, 2))
    caa = sb.selectbox('number of major vessels (0-3)', (0, 1, 2, 3))
    thall = sb.selectbox('Thal rate', (0, 1, 2, 3))

    option = st.checkbox('show the result')

    if option:

        patient = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall
        }
        x_patient = pd.DataFrame([patient])
        print(x_patient)

        x_patient_transformed = column_transformer(X_train, x_patient)[1]

        model = AdaBoostClassifier()

        model.fit(X_train_scaled, y_train)
        prediction = model.predict(x_patient_transformed)
        if prediction == 1:
            st.error('There is a high probability that you \'ll get heart attact !!!')
        else:
            st.success('No worries, you are fine.')
