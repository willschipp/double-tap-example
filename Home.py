import streamlit as st

st.title("The RAG 'Double Tap'")

text = '''
    ### What is it?

    Retrieval Augmented Generation (RAG) is using data to add 'context' to an LLM question.  
    Questions can be things like 'what is the definition of an app' with pages from the dictionary added as 'context'.  
    This helps the LLM answer questions with your specific 'context' in mind.  
      
    To build a 'context', a vector database can be used that contains 'embeddings'.  
    'Embeddings' are numeric representations of text and are called 'vectors'.  
    By using these vectors, a query on a database can figure out if you meant the word 'play' as in 'I want to see a play',
    or 'play' as in 'I want to go play'.  The word 'play' will have a different embedding in each of those two sentences.  

    'Double Tap' is a part of RAG apps where an LLM is asked to process two different sets of data in sequence.  
    1. to convert the question into an 'embedding' for use in the datatabase
    2. to answer the original question, plus the database results  
    
    It becomes a 'double tap' because there's two different models used, one after the other, to complete the users original
    request.


    ### This Demo App...

    The app enables you to upload a PDF and then ask questions of it, illustrating the 'double tap' scenario.  
    To use the app, start with the 'Ingest' page and upload a PDF.  
    Once that's complete, go to the 'Query' page and ask questions of the PDF.
'''

st.markdown(text)