import streamlit as st
import ollama

import chromadb
import chromadb.utils.embedding_functions as embedding_functions


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
vectors = []

if (query):
    # convert to a vector
    with st.spinner("Working..."):
        vectors = ollama.embeddings(model=EMBED_MODEL,prompt=query)        
    #write it out
    vector_text_content = str(vectors['embedding'])[0:255]
    vector_text = st.write(vector_text_content)

    #now query the database for the text
    if st.button("Run the Query"):
        with st.spinner("Querying..."):
            # results = collection.get()
            results = collection.query(
                query_texts=query,
                n_results=10
                )
        st.write(results["documents"])
        final_prompt = prompt.format(context=results["documents"],question=query)
        # now show a prompt
        st.write("Now the prompt")
        st.markdown(final_prompt)
        st.write(len(final_prompt))



#TODO - include LLM submission