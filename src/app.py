import tiktoken
import streamlit as st
import pandas as pd
import plotly.express as px


# setup
embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)


# app
st.title('Tokens Counter for OpenAI embeddings')
st.write("This application calculates the number of tokens in a given text or multiple texts passed in a CSV file. Counting the number of tokens is essential when working with OpenAI's API.")
st.markdown(
    "The maximum number of tokens is limited to **8191** as of **2023-04-09**")

# create subheader
st.subheader("Text tokens counter")
text = st.text_area("Enter text here")
# add button
if st.button("Count tokens (text)"):
    n_tokens = len(encoding.encode(text))
    st.info(f"Required tokens: {n_tokens:,.0f}")

# add file uploader
st.subheader("Multi-text tokens counter")
uploaded_file = st.file_uploader("Choose an CSV file, with column 'text'")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# add button
if st.button("Count tokens (file)"):
    data['n_tokens'] = data['text'].apply(lambda x: len(encoding.encode(x)))
    # plot histogram of number of tokens per text using plotly
    fig = px.histogram(data, x="n_tokens", nbins=100,
                       title="Histogram of required tokens")
    fig.update_xaxes(range=[0, max(data['n_tokens'])*1.1])
    st.plotly_chart(fig)
    mean_n_tokens = data['n_tokens'].mean()
    st.info(f"Mean required tokens: {mean_n_tokens}")
