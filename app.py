from flask import Flask, render_template,jsonify,request # type: ignore
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_pinecone import PineconeVectorStore
import pinecone

app= Flask(__name__) 
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index="medical-chatbot"
print("Downloading Model")
embeddings = download_hugging_face_embeddings()

print("Initializing the Pinecone")
#Initializing the Pinecone
docsearch = PineconeVectorStore(embedding=embeddings, index_name=index)

print("loading the index")
#loading the index
docsearch = PineconeVectorStore.from_existing_index(index,embeddings)

print("calling chain")
PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"promt":PROMPT}

print("loading the model")
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    print("Loading the model...")
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.8}
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load the model: {e}")



qa =RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
print("initializing home page")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=['GET','POST'])
def chat():
    msg=request.form["msg"]
    input=msg
    print(msg)
    result=qa.invoke({"query":input})
    print("Response : ", result["result"])
    return str(result['result'])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)
