import streamlit as st
import time
from turtle import width
import maincode


def customeval() :
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: black;'>Custom Prediction :</h3>", unsafe_allow_html=True)\

    placeholder = st.empty()
    inp = placeholder.text_input('Please enter the text here', key=1)
    col1, col2, col3 = st.columns([3, 3, 3])
    showpred  = col1.button('Predict', key=4)
    clickclear = col3.button('Refresh', key=3)

    if clickclear:
        inp = placeholder.text_input('Sample text here', value='', key=2)

    if showpred :
        x = {inp}
        start = time.time()
        res = maincode.modeleval(inp)
        # prediction = maincode.modeleval(x)
        end = time.time()
        diff=(end-start)
        diffms = diff*1000
        st.write(res)