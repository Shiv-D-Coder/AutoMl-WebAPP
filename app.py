from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

st.set_page_config(
    page_title="AutoMl APP",
    page_icon=":bar_chart:"
)

with st.sidebar:
    st.image("img.png")
    st.title("AutoMlapp")
    choise = st.radio("select from below", [
                      "Uplode", "Profiling", "ML", "Downlode"])
    st.info("This is StremLit Based webapp for AutoML")

if os.path.exists("submission_data.csv"):
    submission_df = pd.read_csv('submission_data.csv', index_col=None)

if choise == "Uplode":
    st.title("Uplode your Data for Further Processing :point_down:")

    uploded_file = st.file_uploader("Uplode your .csv file here")
    if uploded_file:
        df = pd.read_csv(uploded_file, index_col=None)
        df.to_csv("submission_data.csv", index=None)
        st.dataframe(df)
        st.success("Your Data has Successfully Uploaded :white_check_mark:")
        st.snow()

if choise == "Profiling":
    st.title("Exploratry Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choise == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        st.success("Your model is Successfully Trained")
        st.info("Plese Click to Download for installing your trained model")

if choise == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

    st.balloons()
