"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

import pickle

import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Sustainable Data Day - ESG Data Bot retriever", page_icon=":robot:")


# Load the LangChain.
@st.cache_resource
def load_chain():
    index = faiss.read_index("docs.index")

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index

    chat = ChatOpenAI(temperature=0)

    chain = VectorDBQAWithSourcesChain.from_llm(llm=chat, vectorstore=store)
    return chain


chain = load_chain()

# From here down is all the StreamLit UI.

st.header("Sustainable Data Day ESG - Data Bot Retriever")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

with st.form(key="form"):
    user_input = st.text_input("You: ", "Hello, who are you?", key="input")
    submit_button_pressed = st.form_submit_button("Submit to Bot")

if submit_button_pressed:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
