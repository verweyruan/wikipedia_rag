import os

import streamlit as st
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv()


INDEX_DIR = 'wiki_rag'
PAGES = [
    "Machine learning",
    "Artificial intelligence",
    "Software engineering",
    "Saas",
    "Software Company",
    "Vibe Coding"
]


@st.cache_resource
def get_index():
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    embedding_model = OpenAIEmbedding('text-embedding-3-small')
    index = VectorStoreIndex.from_documents(docs, embedding_model=embedding_model)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    return index

@st.cache_resource
def get_query_engine():
    index = get_index()

    llm = OpenAI(model='gpt-4o-mini', temperature=0)

    return index.as_query_engine(llm=llm, similarity_top_k=3)

