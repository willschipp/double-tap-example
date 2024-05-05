
from ollama import embeddings
import logging
import time
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "test_documents"

#class to do the embedding side
class Embed:

    embedding = None
    chroma_client = None
    collection = None
    ef = None

    def __init__(self):
        # setup the ollama embedding process
        logging.basicConfig(level=logging.INFO)
        self.chroma_client = chromadb.Client()
        self.ef = embedding_functions.OllamaEmbeddingFunction(url=BASE_URL + "/api/embeddings",
                                                                model_name=EMBED_MODEL)
        self.collection = self.chroma_client.create_collection(name=COLLECTION,embedding_function=self.ef)

    def get_vectors(self,text):
        logging.info("starting embed...")
        starttime = time.time()
        os.environ["OLLAMA_HOST"] = BASE_URL
        results = embeddings(model=EMBED_MODEL,prompt=text)
        endtime = time.time()
        duration = endtime - starttime
        results_obj = {"results":results,
                       "duration":duration}
        logging.info("finished embed")
        logging.info(duration)
        return results_obj

    def add_document(self,document):
        self.collection.add(
            documents=[document['document']],
            metadatas=[document['metadata']],
            ids=[document['id']]
        )        

    def add_document_vectors(self,document):
        self.collection.add(
            embeddings=[document['vectors']],
            documents=[document['document']],
            metadatas=[document['metadata']],
            ids=[document['id']]
        )        

    def quick_query(self,query):
        starttime = time.time()
        results = self.collection.query(
            query_texts=[query],
            n_results=10,
        )
        endtime = time.time()
        logging.info(endtime - starttime)
        return results
    
    def collection_count(self):
        return self.collection.count()
    
    


