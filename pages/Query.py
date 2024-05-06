import streamlit as st
import ollama

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer ..."}

def query_huggingface(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


st.header("Query")

text = '''
    ### Process  
    1. The query is turned into a set of vectors  
    2. The vector is then used to search against the vector database
    3. The search results are then added into the prompt
    3. And the prompt is then sent to the query LLM
'''

prompt = '''
    Answer the question using only the provided context.  
    If you don't know the answer, just reply with you don't know, don't try and make up a response.

    {context}

    Question: {question}
'''

st.markdown(text)

# setup other components
embedding_function = embedding_functions.OllamaEmbeddingFunction(url=BASE_URL + "/api/embeddings",model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path="./vectorstore/chroma/")
collection = client.get_or_create_collection(name="double-tap",embedding_function=embedding_function)

query = st.text_input("Enter your query")
st.divider()
vectors = []

if (query):
    # convert to a vector
    with st.spinner("Working..."):
        vectors = ollama.embeddings(model=EMBED_MODEL,prompt=query)        
    #write it out
    st.subheader("Vectorized query")
    vector_text_content = str(vectors['embedding'])[0:255] + "...]"
    vector_text = st.write(vector_text_content)
    st.write("Performed on the local CPU")

    st.divider()

    #now query the database for the text
    if st.button("Run the Query"):
        with st.spinner("Querying..."):
            # results = collection.get()
            results = collection.query(
                query_texts=query,
                n_results=10
                )
        final_prompt = prompt.format(context=results["documents"][0],question=query)
        # now show a prompt
        st.subheader("Now ask the complete query")
        # st.markdown(final_prompt)
        # st.divider()
        # now LLM submission
        with st.spinner("Asking the LLM..."):
            # result = ollama.generate(model="mistral",prompt=final_prompt,stream=False)
            result = query_huggingface({
                "inputs": final_prompt,
                })
        st.subheader("The Answer...")
        st.write(result[0]['generated_text'])
        st.write("Performed on a remote GPU")


