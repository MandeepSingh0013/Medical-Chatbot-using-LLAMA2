from langchain.llms import CTransformers
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