import os

import streamlit as st
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings

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
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()

    llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    return index.as_query_engine(llm=llm, similarity_top_k=3)


def main():
    st.title('Wikipedia RAG Application')

    question = st.text_input('Ask a Question')
    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa = get_query_engine()
            response = qa.query(question)

        st.subheader('Answer')
        st.write(response.response)
        st.subheader('Retrieved context')

        for src in response.source_nodes:
            st.markdown(src.node.get_content())


if __name__ == '__main__':
    main()
