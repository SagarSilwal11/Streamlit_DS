import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#uploading the data using Hardcored File path
st.title('Dataset head:')
st.divider()
df = pd.read_excel(r'F:\DS_project_org\Classification_handedness\survey.xls')  # Provide the correct filename with the extension
st.write(df.head())  # Display the DataFrame
st.write(f'Shape:{df.shape}')
st.write(f'Columns:{df.columns}')
st.write(f"Classify:{df['Handedness'].unique()}")


##Data Visualization Part
st.title('Data Visualization:')
st.divider()
#plot Nationality chart
st.write('### Nationality')
nation=df['Nationality'].value_counts()
st.bar_chart(nation)
#plot Sex chart
st.write('### Sex') 
sex=df['Sex'].value_counts()
st.bar_chart(sex)
st.write('### Nationality and Handedness')
st.divider()
crosstab=pd.crosstab(df['Nationality'],df['Handedness'])
st.bar_chart(crosstab)
st.write('### Sex and Handedness')
crosstab=pd.crosstab(df['Sex'],df['Handedness'])
st.bar_chart(crosstab)


#Peprocessing
st.title('Data Preprocessing')
st.divider()
new_df=df.copy()
st.write(new_df)
label_encoding_cols=['Nationality','Sex','Handedness']
le_nation=LabelEncoder()
le_sex=LabelEncoder()
le_hand=LabelEncoder()
new_df['Nationality']=le_nation.fit_transform(df['Nationality'])
new_df['Sex']=le_sex.fit_transform(df['Sex'])
new_df['Handedness']=le_hand.fit_transform(df['Handedness'])
st.write('Labels')
le_nation_map=dict(zip(le_nation.classes_,le_nation.transform(le_nation.classes_)))
le_sex_map=dict(zip(le_sex.classes_,le_sex.transform(le_sex.classes_)))
le_hand_map=dict(zip(le_hand.classes_,le_hand.transform(le_hand.classes_)))
st.write(le_nation_map,le_sex_map,le_hand_map)
st.write('Label_dataframe')
st.write(new_df)
scale=MinMaxScaler()
new_df['Age']=scale.fit_transform(new_df[['Age']])
st.text('Processed_data')
st.write(new_df)


# Model Training
st.title('Model_fiting_area')
st.divider()
X=new_df.iloc[:,1:4]
y=new_df.iloc[:,4]
model_name=st.selectbox('Select Classifier',('LogisticRegression','RandomForest','DecisionTree'))

def take_parameters_ui(model_name):
    params={}
    if model_name=='LogisticRegression':
        C=st.sidebar.slider('C',0.01,10.0,0.05)
        params['C']=C
        penalty=st.sidebar.selectbox('Penalty',['l2','l1'])
        params['Penalty']=penalty
        solver=st.sidebar.selectbox('solver',['lbfgs','sag','saga','liblinear'])
        params['solver']=solver
    elif model_name=='RandomForest':
        max_depth=st.sidebar.slider("max_depth", 2, 15) 
        params["max_depth"] = max_depth 
    else:
        split=st.sidebar.selectbox('splitter',['best','random'])
        params['splitter']=split
        crite=st.sidebar.selectbox('criterion',['gini','entropy','log_loss'])
        params['criterion']=crite
    return params

params=take_parameters_ui(model_name)


def get_model(model_name,params):
    model=None
    if model_name=='LogisticRegression':
        model=LogisticRegression(C=params['C'],penalty=params['Penalty'],solver=params['solver'])
    elif model_name=='RandomForest':
        model=RandomForestClassifier(max_depth=params['max_depth'])
    else:
        model=DecisionTreeClassifier(splitter=params['splitter'],criterion=params['criterion'])
    return model


model=get_model(model_name,params)

model.fit(X,y)
y_pred=model.predict(X)
score=model.score(X,y)

st.write(f'Model={model_name}')
st.write(f'Accuracy={score}')

