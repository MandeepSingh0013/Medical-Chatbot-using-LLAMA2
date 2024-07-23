from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print("Extracting Data")
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("Downloading the hugging face")
embeddings = download_hugging_face_embeddings()
print("uploading Start")
#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index= pc.Index("medical-chatbot")

embedding= [embeddings.embed_query(t.page_content) for t in text_chunks]
#Creating Embeddings for Each of The Text Chunks & storing
vectors=[{
    "id": f"vec{i+1}",
    "values": embedding[i],
    "metadata" : {"text": str(text_chunks[i].page_content)}
}
 for i in range(0,1000)#len(text_chunk)
]
namespace = "medical_chatbot"


index.upsert(
    vectors = vectors,
    namespace= namespace
)
print("vector Uploaded")