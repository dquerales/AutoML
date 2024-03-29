import streamlit as st
from pycaret.classification import setup, pull, compare_models, save_model, evaluate_model, predict_model
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os 

st.set_page_config(page_title="AutoML & Explorer App", page_icon="🚗")

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("🚗 AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling"])
    st.markdown('Made by [Daniel Querales](https://www.linkedin.com/in/danielquerales/)')

if choice == "Upload":
    st.title("✏️ Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        col1, col2 = st.columns(2)
        col1.metric('Rows', df.shape[0])
        col2.metric('Columns', df.shape[1])

if choice == "Profiling": 
    st.title("🔎 Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    export = profile_df.to_html()
    st.download_button(label="Download Report", data=export, file_name='profile_report.html', use_container_width=True)

if choice == "Modelling": 
    st.title("🕹️ Train your Model")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling', use_container_width=True): 
        setup(df, target=chosen_target, session_id=1)
        setup_df = pull()
        st.header('Setup')
        st.dataframe(setup_df, use_container_width=True)
        st.header('Compare Models')
        best_model  = compare_models(budget_time = 0.5)
        compare_df = pull()
        st.dataframe(compare_df, use_container_width=True)
        evaluate_model(best_model)
        st.header('Test Data')
        predict_model(best_model)
        predict_df = pull()
        st.dataframe(predict_df,  use_container_width=True)
        st.header('Download your model')
        save_model(best_model, 'best_model')
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl", use_container_width=True)
