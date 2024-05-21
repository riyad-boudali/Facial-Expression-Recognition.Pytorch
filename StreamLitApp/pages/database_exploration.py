import streamlit as st
import pickle
import pandas as pd
import chardet 

st.set_page_config(
  page_title="Apeep",
  page_icon="H",
  layout="wide",
  initial_sidebar_state="expanded"
)

st.header("Data exploration")

uploaded_file = st.file_uploader("Upload you dataframe in csv")
if uploaded_file is not None:
  # Detect file encoding
  rawdata = uploaded_file.read()
  result = chardet.detect(rawdata)
  encoding = result['encoding']

  # Use the detected encoding to read the file
  uploaded_file.seek(0)  # Reset the file pointer to the beginning
  df = pd.read_csv(uploaded_file, encoding=encoding)

  st.dataframe(df, use_container_width=True)

  st.subheader("Dataframe description")
  st.write(df.describe())