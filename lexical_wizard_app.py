import os

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')
#add_replicate_api = os.environ['REPLICATE_API_TOKEN']
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


#######################################################
#     :: LEXICAL WIZARD ::
#       ================
#       Description:
#           -   This project is aimed to perform Information Retrieval from Uploaded Document
#                       and provide answers to custom user queries in the real time.
#           -   Components:
#                -   PDFLoader : To load the PDF
#                -   CharacterTextSplitter : To split the text from loaded PDF in chunks
#                -   HuggingFaceEmbeddings : To convert text chunks in to embeddings
#                -   FAISS : Local in-memory Vector store to store indexed embeddings
#                -   LlaMA2 : Large Language Model with 7B parameters (One of the most powerful
#                               Open Source model currently).
#                -   Replicate : Helps in invoking LLM models through API

#       Team Members :
#           -   Kumar, Amrit  (amritk2@illinois.edu)
#           -   Madhavan, Siddharth (sm120@illinois.edu)
#           -   Gattu, Sudha Mrudula  (sudhamg2@illinois.edu)
#           -   Awachat, Ishika   (awachat2@illinois.edu)
#######################################################

# --------------------  [Business Logic implementations- STARTS here] -------------#
def get_pdf_text(pdf_docs):
    """
        Reads Uploaded PDF and generate text
    :param pdf_docs: List of PDF Documents
    :return:
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Takes the input text generated from the source PDF/other sources and generate smaller chunks
    :param text: large Text from PDF
    :return: Chunks after splitting the input text from the PDF
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Generates Embeddings for the chunks and stores in the FAISS - in memory vector store
    :param text_chunks:
    :return: vector store
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    """
    -   Creates the conversational LLM chain based on the 'LlaMA2 70B chat' model using the in memory vector store.
    -   Calls the LLM model via Replicate API.
    -   Subsequent chat queries are answered based on the vector embeddings and the current chat history context.
    :param vector_store:
    :return: Conversational Chain
    """
    load_dotenv()

    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1})
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    """
    Handles conversation between user and the LexicalWizard Chatbot.
    :param user_question:
    :return: Generated output
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:

            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def clear_chat_history():
    memory.clear()
    st.session_state.chat_history = []
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today"}]
# --------------------  [Business Logic implementations- Ends here] -------------#

# --------------------  [Streamlit UI code - STARTS here] -------------#
def main():
    load_dotenv()
    st.set_page_config("Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Lexical Wizard !!")
    st.subheader(
        "Information Retrieval using Large Language Models!! ::books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    user_question = st.chat_input("Ask a question from your documents")

    # st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    # if st.button("Clear Chat"):
    #     clear_chat_history()
    #     st.text("Chat cleared.")

    if user_question:
        handle_user_input(user_question)
    with st.sidebar:

        st.title("Chat based Information Retrieval from PDF.")
        st.header("Powered by Llama-2-70b Chat Model")
        st.subheader("Chat with PDF 💬")
        pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner("Processing"):
                # Extract Text from PDF
                raw_text = get_pdf_text(pdf_docs)
                # Split the Text into Chunks
                text_chunks = get_text_chunks(raw_text)
                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Done!")


        st.header("References")
        st.markdown('''
                - [LLAMA2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
                - [Streamlit](https://streamlit.io/)
                - [Replicate](https://replicate.com/)
                - [HuggingFace Tokens](https://huggingface.co/settings/tokens) 
                ''')

        st.header("Team: Lexical Wizard")
        st.markdown('''
                        - Amrit Kumar
                        - Siddharth Madhavan
                        - Sudha Mrudula Gattu
                        - Ishika Awachat
                        ''')
# --------------------  [Streamlit UI code - Ends here] -------------#

if __name__ == "__main__":
    main()
