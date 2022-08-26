import streamlit as st

def aboutdisp () :
    st.markdown("<h1 style='text-align: Left; '>About the model </h1>", unsafe_allow_html=True)
    st.write("This model is analysing sentiments of Amazon review data.",
            "It is based on data science and AI, it has Logistic regression as its algorithm. The model is predicting the sentiments via AI",
            "and telling the results as whether the user is satisfied or unsatisfied.")
    st.markdown("<h1 style='text-align: Left; '>About the Author </h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: Left; '>Spraha Kumawat </h4>", unsafe_allow_html=True)
    st.write("I was born in year 2002 In Jaipur district of Rajasthan, brought up and schooled in Jaipur only.",
    "I am a student at Amity Univeristy, Noida","I have keen interest in data science and AIML and wish to go ahead in this field as a career.")