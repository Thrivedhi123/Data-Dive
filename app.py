import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup, compare_models, pull, save_model, load_model

import os
def main():
    
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv',encoding='unicode_escape')
        
    activities =["Home","EDA","Plots","Modelling"]
    choice = st.sidebar.selectbox("Select Page",activities)
    if choice =="Home":
        st.title("Data Dive")
        st.markdown("""
Toss your data in 📊, Data Dive will uncover hidden secrets with fancy charts 📈 and even predict the future with some ML magic ✨!
""")     
        st.header("About this Project")
        st.markdown("""
🚀 Dive into data exploration like a pro with our EDA project! Here's what you can do:

🔍 **Explore Data**: Throw your data in and watch the magic happen!  
📊 **Visualize**: Get ready for some eye candy with fancy charts and plots!  
🤖 **Modeling**: Let's make predictions! Our ML magic is here to help you build models with ease!  
🌟 **Excitement**: It's not just analysis, it's an adventure! Get ready to uncover hidden insights and make data-driven decisions like never before!
""")

    if choice == "EDA":
        st.header("Exploratory Data Analysis")
        data = st.file_uploader("Upload a dataset",type=['text','csv'])
        if data is not None:
            df=pd.read_csv(data,encoding="unicode_escape")
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)
            c= st.selectbox("",["Show Shape","Show Summary","Show Columns"])
            if c=="Show Shape":
                st.write(df.shape)
            
            if c =="Show Summary":
                st.write(df.describe())
                
            all_columns = df.columns.to_list()

            if c =="Show Columns":
                st.write(all_columns)
            
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

                
    if choice =="Plots":
        st.header("Data Visulaization")
        st.markdown("""
    📊 Welcome to the Data Visualization page! Here, you can explore your dataset through various plots and charts.
    
    **Select Type**: Choose the type of plot you want to generate from the dropdown menu.
    
    **Select Columns for Plotting**: Pick the columns you want to visualize from your dataset.
    
    After making your selections, hit the "Generate Plot" button to see your customizable plot!
    """)
        all_column_names = df.columns.tolist()
        type_of_plot = st.selectbox("select Type",["Area","bar","line","hist","box","kde"])
        selected_column_names = st.multiselect("Select required Columns for Analysis for plotting",all_column_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_column_names))
            
            if type_of_plot == 'Area':
                cust_data = df[selected_column_names]
                st.area_chart(cust_data)
            
            elif type =="bar":
                cust_data = df[selected_column_names]
                st.bar_chart(cust_data)
                
            elif type_of_plot == 'line':
                cust_data = df[selected_column_names]
                st.line_chart(cust_data)

				# Custom Plot 
            elif type_of_plot:
                cust_plot= df[selected_column_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()
    
    if choice == "Modelling": 
            st.header("Modelling")
            st.markdown("""
    🤖 Welcome to the Modelling page! Here, you can build and compare machine learning models for predictive analysis.
    
    **Choose the Target Column**: Select the target variable you want to predict from your dataset.
    
    Click the "Run Modelling" button to initiate the model training process. Once completed, you'll see the ML experiment settings and the best performing model along with its evaluation metrics.
    """)
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'): 
                  setup(df, target=chosen_target,verbose=False)
                  setup_df = pull()
                  st.info("This is ML experiment settings")
                  st.dataframe(setup_df)
                  best_model = compare_models()
                  compare_df = pull()
                  st.dataframe(compare_df)
                  best_model
                  save_model(best_model, 'best_model')
                  
main()