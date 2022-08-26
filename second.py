import pandas as pd
import numpy as np
import neattext.functions as nfx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


def display(df_data) :
    df_data['Cleantext'] = df_data['Summary'].apply(str)
    #punctuations 
    df_data['Cleantext'] = df_data['Cleantext'].apply(nfx.remove_punctuations)
    # Stopwords
    df_data['Cleantext'] = df_data['Cleantext'].apply(nfx.remove_stopwords)

    #Dataframe
    st.markdown("<h1 style='text-align: left; </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; '>Dataset Used</h2>", unsafe_allow_html=True)
    df_disp = df_data.sample(100)
    st.dataframe(df_disp,width=750,height= 200)

    #Data Visualization
    st.markdown("<h1 style='text-align: left; </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; '>Visualization of Dataset :</h2>", unsafe_allow_html=True)

    #Scatterplot
    st.markdown("<h1 style='text-align: left; </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; '>Scatterplot :</h3>", unsafe_allow_html=True)
    df_scatterplot = df_data.sample(100)
    df_scatterplot['Id'] = np.arange(len(df_scatterplot)) 
    fig = px.scatter(df_scatterplot,x="Id", y="Score", color = "Score",hover_data=['Cleantext'])
    fig.update_layout(title_text='Scatterplot', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    #Countplot
    st.markdown("<h3 style='text-align: left; </h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; '>Countplot :</h3>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df_data['Score'])  
    plt.title("Countplot of Sentiments")    
    st.pyplot(fig)
    
    #Distplot
    st.markdown("<h3 style='text-align: left; </h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; '>Distplot :</h3>", unsafe_allow_html=True)
    plt.title("Distplot")
    fig = sns.displot(df_data, x=df_data['Score'])
    st.pyplot(fig)

    #Histogram
    st.markdown("<h3 style='text-align: left; </h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; '>Histogram :</h3>", unsafe_allow_html=True)
    fig = sns.pairplot(df_data , x_vars="Id", y_vars="Score",kind="kde")
    st.pyplot(fig)