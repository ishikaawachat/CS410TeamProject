#!/usr/bin/env python
# coding: utf-8

# #**Llama 2+ Pinecone + LangChain**

# ##**Step 1: Install All the Required Pakages**

# In[4]:


# get_ipython().system('pip install langchain')
# get_ipython().system('pip install pypdf')
# get_ipython().system('pip install unstructured')
# get_ipython().system('pip install sentence_transformers')
# get_ipython().system('pip install pinecone-client')
# get_ipython().system('pip install llama-cpp-python')
# get_ipython().system('pip install huggingface_hub')


# In[22]:



# get_ipython().system('pip install --ignore-installed PyYAML')
#
# get_ipython().system('pip install pinecone-client')


# #**Step 2: Import All the Required Libraries**

# In[7]:


from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
#import pinecone
import os


# #**Step 3: Load the Data**

# In[41]:


#get_ipython().system('gdown "https://drive.google.com/uc?id=15hUEJQViQDxu_fnJeO_Og1hGqykCmJut&confirm=t"')
#content = 'content/'

# In[9]:


#ls


# In[10]:


#loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

# #**Step 3: Load the Data**
loader = PyPDFLoader("content/The-Field-Guide-to-Data-Science.pdf")

data = loader.load()


# In[12]:


data


# #**Step 4: Split the Text into Chunks**
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

docs=text_splitter.split_documents(data)



print(f"docs length: {len(docs)}")


# In[20]:



print(f"docs 0: {docs[0]}")

# #**Step 5: Setup the Environment**

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '2e4b8c70-7a88-4bae-b8d1-5b92af6639d3')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')


# #**Step 6: Downlaod the Embeddings**


embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# #**Step 7: Initializing the Pinecone**

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "cs410lexicalwizard" # put in the name of your pinecone index here


# #**Step 8: Create Embeddings for Each of the Text Chunk**
docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

# # If pinecone index are already present, it can be resused as below

#docsearch = Pinecone.from_existing_index(index_name, embeddings)


# #**Step 9: Similarity Search**
query="YOLOv7 outperforms which models"  # Sample query to be searched in the loaded doc/PDF

docs=docsearch.similarity_search(query)
print(f"docs retrieved after Similarity Search:\n {docs}")



# #**Step 9: Query the Docs to get the Answer Back (Llama 2 Model)**

# In[24]:


#get_ipython().system('CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose')


# #Import All the Required Libraries

# In[25]:


from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain


# In[26]:


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager


# #  Quantized Models from the Hugging Face Community

# The Hugging Face community provides quantized models, which allow us to efficiently and effectively utilize the model on the T4 GPU. It is important to consult reliable sources before using any model.
# 
# There are several variations available, but the ones that interest us are based on the GGLM library.
# 
# We can see the different variations that Llama-2-13B-GGML has [here](https://huggingface.co/models?search=llama%202%20ggml).
# 
# 
# 
# In this case, we will use the model called [Llama-2-13B-chat-GGML](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML).

#  Quantization reduces precision to optimize resource usage.

# Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer ( int8 ) instead of the usual 32-bit floating point ( float32 ).

# In[27]:


model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format


# In[28]:


model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)


# In[29]:


n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=False,
)


# In[30]:


chain=load_qa_chain(llm, chain_type="stuff")


# In[31]:


query="YOLOv7 outperforms which models"
docs=docsearch.similarity_search(query)


# In[33]:


print(f"docs :{docs}")



# In[32]:


chain.run(input_documents=docs, question=query)


# In[34]:


query="YOLOv7 trained on which dataset"
docs=docsearch.similarity_search(query)


# In[35]:


chain.run(input_documents=docs, question=query)


# #**Step 10: Query the Docs to get the Answer Back (Hugging Face Model)**

# In[ ]:


from langchain.llms import HuggingFaceHub


# In[ ]:


llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})


# In[ ]:


chain=load_qa_chain(llm, chain_type="stuff")


# In[ ]:


query="What are examples of good data science teams?"
docs=docsearch.similarity_search(query)


# In[ ]:


chain.run(input_documents=docs, question=query)


# In[ ]:




