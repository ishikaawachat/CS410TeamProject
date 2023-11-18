from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import os
from langchain.chains.question_answering import load_qa_chain

class QandA:
    def __init__(self):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
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
print(question_and_answer.get_answers("What does who indicate?"))