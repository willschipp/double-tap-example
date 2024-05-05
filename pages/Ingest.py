import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from io import BytesIO
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions


BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

st.header("Ingest")

text = '''
    ### Process  
    1. a source file (PDF) gets uploaded and split  
    2. each 'split' then has a set of vectors created for it via 'embedding'
    3. the vector is used as an index, and stored along side the 'split'
'''

st.markdown(text)

# setup other components
embedding_function = embedding_functions.OllamaEmbeddingFunction(url=BASE_URL + "/api/embeddings",model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path="./vectorstore/chroma/")
collection = client.get_or_create_collection(name="double-tap",embedding_function=embedding_function)

uploaded = st.file_uploader(label="Choose a PDF...",type='pdf')

st.session_state.df = pd.DataFrame([])

if uploaded:
    with st.spinner('Uploading the file...'):
        # convert it to a dataframe
        pdf_bytes = uploaded.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        i = 0
        while (i<len(reader.pages)):
            page = reader.pages[i]
            i += 1
            content = page.extract_text()
            pageObj = {"number":i,
                       "content":content,
                       "length":len(content)}
            pages.append(pageObj)

    st.session_state.df = pd.DataFrame(pages)


def save_df():
    df = st.session_state.df
    if len(df) > 0:
        with st.spinner("Saving..."):
            collection.add(
                documents=df['content'].tolist(),
                embeddings=df['vectors'].tolist(),
                ids=list(map(str,df['number'].tolist()))
            )
                


#now that we have the collection, make it visible
with st.form("df_form"):
    #always at the top
    submitted = st.form_submit_button("Create Vectors")

    # df = pd.DataFrame(pages)
    df = st.session_state.df
    df_display = st.dataframe(df)
    
    if submitted:
        if len(df) > 0:                
            with st.spinner("Generating vectors..."):
                vectors_array = []
                for idx, row in df.iterrows():
                    content = row['content']
                    vectors = ollama.embeddings(model=EMBED_MODEL,prompt=content)
                    vectors_array.append(vectors['embedding'])
            df_display.empty()
            df.insert(3,"vectors",vectors_array)
            df_display = st.dataframe(df)


st.button("Save",on_click=save_df)


    # if st.button("Create Vectors"):
    #     with st.spinner("Generating vectors..."):
    #         vectors_array = []
    #         # go through the data frame
    #         for idx, row in df.iterrows():
    #             # get the content
    #             content = row['content']
    #             # encode
    #             vectors = ollama.embeddings(model=EMBED_MODEL,prompt=content)
    #             vectors_array.append(vectors['embedding'])
    #     dfdisplay.empty()
    #     df.insert(3,"vectors",vectors_array)
    #     dfdisplay = st.dataframe(df)

    #     numbers = list(map(str,df['number'].tolist()))
    #     st.write(numbers)

    #     collection.add(
    #         documents=df['content'].tolist(),
    #         ids=numbers
    #     )

    #     st.write("Done. Now you can go query it.")

