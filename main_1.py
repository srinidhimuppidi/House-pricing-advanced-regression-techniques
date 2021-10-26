import streamlit as st

st.title("GUI for Machine Learning Model")

st.subheader("Welcome to the sample GUI")
st.radio("Select the option",[1,2,3])
st.text_input("Enter some value")
st.button("Hit me")

st.checkbox("Check this out")

st.slider("Slide the bar",min_value=0,max_value=10)

st.selectbox("Select the option wanted",("Cat","Dog"))

st.sidebar.button("Check-in","Checkout")

