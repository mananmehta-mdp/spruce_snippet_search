import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from pymilvus import connections, Collection

# Load environment variables
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Connect to Milvus
connections.connect("default", host="10.16.31.196", port="19530", user=os.getenv("milvus_user"), password=os.getenv("milvus_password"))
collection = Collection("Spruce_text_snippets")

search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 256}
}

def semantic_search(query, k=10):
    query_vector = embeddings.embed_query(query)
    results = collection.search([query_vector], "embedding", search_params, limit=k, output_fields=["text_chunk"])
    
    answers = []
    for hit in results[0]:
        answer = {
            "URL": hit.id,
            "Distance": hit.distance,
            "Text": hit.entity.get("text_chunk")
        }
        answers.append(answer)
    
    return answers

# Streamlit UI
st.title("Semantic Search with Milvus")

# Get user input
query = st.text_input("Enter your query:", "")
k_value = st.slider("Number of results:", 1, 100, 10)

# If the user has entered a query, perform the search and display the results
if query:
    results = semantic_search(query, k=k_value)
    
    for result in results:
        st.markdown(f"**URL:** {result['URL']}<br>**Distance:** {result['Distance']}<br>**Text:** {result['Text']}", unsafe_allow_html=True)
        st.write("---")

if __name__ == '__main__':
    st.write("App is running!")
