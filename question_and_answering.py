import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import os
from langchain.chains.question_answering import load_qa_chain

class QandA:
    def __init__(self):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['API_TOKEN']
        pdfreader = PdfReader('Research_Paper_on_Artificial_Intelligence.pdf')

        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings()
        self.document_search = FAISS.from_texts(texts, embeddings)

    def loadLLM(self):
        llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain

    # query = "What does who indicate?"
    def get_answers(self, query):
        chain = self.loadLLM()
        docs = self.document_search.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
    

question_and_answer = QandA()

# App title
st.set_page_config(page_title="Lexical Wizards")
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = question_and_answer.get_answers(prompt)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)