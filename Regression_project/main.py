import streamlit as st
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,LinearRegression

# Set background color using custom CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("solar_pics.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add your Streamlit app content here
st.markdown("""
# GO GREEN PROJECT
#### SOLAR POWER TO AC_POWER PREDICTION
""")
st.divider()
st.image(r"solar_pics.jpg",caption='SolarPanel')


def load_csv():
    df=pd.read_csv('process_data.csv')
    # st.write(df)
    return df
data=load_csv()


def split_fit_predict(data):
    X=data.drop(['AC_POWER'],axis=1)
    y=data['AC_POWER']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
model=split_fit_predict(data)


model_name=st.sidebar.selectbox('Select Regression',('LinearRegression','Lasso',"Ridge"))

def parameter(model_name):
    params={}
    if model_name=='Lasso' or model_name=='Ridge':
        alpha=st.sidebar.slider("Alpha",1,100)
        params['Alpha']=alpha
    return params

params=parameter(model_name)


def model(model_name,params):
    X_train,X_test,y_train,y_test=split_fit_predict(data)
    if model_name=='LinearRegression':
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        y_pred=lr.predict(X_test)
        train_score=lr.score(X_train,y_train)
        test_score=lr.score(X_test,y_test)
        st.markdown(f'#### The training score is{train_score} and testing score is {test_score}')
    elif model_name=='Lasso':
        lasso=Lasso(alpha=params['Alpha'])
        lasso.fit(X_train,y_train)
        train_score=lasso.score(X_train,y_train)
        test_score=lasso.score(X_test,y_test)
        st.markdown(f'#### The training score is{train_score} and testing score is {test_score}')
    else:
        ridge=Ridge(alpha=params['Alpha'])
        ridge.fit(X_train,y_train)
        train_score=ridge.score(X_train,y_train)
        test_score=ridge.score(X_test,y_test)
        st.markdown(f'#### The training score is{train_score} and testing score is {test_score}')

model_s=model(model_name,params)
st.divider()
st.title('User Interface')

#Using the UI based for input the data and predict

def get_data():
    
    source_key = st.selectbox('source_key', (
        '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD', '7JYdWkrLSPkdwr4',
        'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE',
        'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB', 'bvBOhCH3iADSZry', 'iCRJl6heRkivqQ3',
        'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv',
        'uHbuxQJl8lW7ozc', 'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY',
        'zVJPv84UY57bAof', 'YxYtjZvoooNbGkE'
    ))
    daily_yield = st.number_input("Enter Daily Yield:",min_value=0)
    total_yield = st.number_input("Enter Total Yield:",min_value=0)
    ambient_temperature = st.number_input("Enter Ambient Temperature:",min_value=0)
    irradiation = st.number_input("Enter Irradiation:",min_value=0)
    hour = st.number_input("Enter Hour:",min_value=0,max_value=23)
    day = st.number_input("Enter Day:",min_value=0)
    minute = st.number_input("Enter Minute:",min_value=0,max_value=45)
    data={'SOURCE_KEY': [source_key],
        'DAILY_YIELD': [daily_yield],
        'TOTAL_YIELD': [total_yield],
        'AMBIENT_TEMPERATURE': [ambient_temperature],
        'IRRADIATION': [irradiation],
        'Hour': [hour],
        'Day': [day],
        'Minute': [minute]
    }
    return data


data=get_data()
def dataframe(data):
    df=pd.DataFrame(data)
    st.subheader('Dataframe')
    st.write(df)
    return df 

df=dataframe(data)

def preprocessing(df):
    with open('preprocessing.pki','rb') as f:
        encoder,scaler=pickle.load(f)
    
    df['SOURCE_KEY']=encoder.fit_transform(df['SOURCE_KEY'])
    source_map=dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
    st.write('label_encoding')
    st.write(source_map)
    st.write("Scaling the data:")
    scale_df=scaler.fit_transform(df)
    st.write(df)
    processed_df=df.copy()
    return processed_df

processed_df=preprocessing(df)


def model_prediction(processed_df):
    
    with open('models.pki','rb') as f:
        lr,lasso,ridge=pickle.load(f)
    if model_name=='LinearRegression':
        predicted_data=lr.predict(processed_df)
        st.write('The predicted AC value generated is',predicted_data)
    elif model_name=='Lasso':
        predicted_data=lasso.predict(processed_df)
        st.write('The predicted AC value generated is',predicted_data)
    else:
        predicted_data=ridge.predict(processed_df)
        st.write('The predicted AC value generated is',predicted_data)

model_user=model_prediction(processed_df)