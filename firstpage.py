import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
import time
import second
import third
import fourth
import third

df_data= pd.read_csv("Reviews.csv")
st.markdown("<h1 style='text-align: center; '>Sentiment Analysis Model</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; '>Artifically Intelligent model for sentiment analysis model</h4>", unsafe_allow_html=True)
st.markdown("<h6>Description: </h6>", unsafe_allow_html=True)
st.write("Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive,",
" negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor.", "It plans to concentrate on individuals'",
 "viewpoints, sentiments, and perspectives about subjects, occasions, issues, elements, people, and their",
  "qualities in web-based entertainment (e.g., person to person communication destinations, discussions, web journals, and so forth)",
   "communicated by either message surveys or remarks. Amazon is an illustration of the world's biggest web-based retailer that permits"
   , "its clients to rate its items and openly compose surveys.")

r =st.sidebar.radio("Index",["Main page","Dataset","About"])

if r == "Dataset":
    st.markdown("<h1 style='text-align: left; </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; '> Dataset</h1>", unsafe_allow_html=True)
    second.display(df_data)
    url = 'C:/Users/91982/Desktop/NTCC/Reviews.html';
    if st.button('View Summary DataSet'):
        webbrowser.open_new_tab(url)

if r == "About" :
    third.aboutdisp()