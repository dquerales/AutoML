import streamlit as st
from pycaret.classification import setup, pull, compare_models, save_model, evaluate_model, predict_model
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://images.unsplash.com/photo-1582571352032-448f7928eca3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=992&q=80")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    #st.info("This project application helps you build and explore your data.")
    st.info('This webapp was made by Daniel Querales (d.querales@gmail.com) using **Streamlit**.')
    #st.image('https://streamlit.io/images/brand/streamlit-mark-color.png', width=50)

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        col1, col2 = st.columns(2)
        col1.metric('Rows', df.shape[0])
        col2.metric('Columns', df.shape[1])

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    export = profile_df.to_html()
    st.download_button(label="Download Report", data=export, file_name='profile_report.html', use_container_width=True)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling', use_container_width=True): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        # evaluate_model(best_model)
        predict_model(best_model)
        predict_df = pull()
        st.dataframe(predict_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl", use_container_width=True)
