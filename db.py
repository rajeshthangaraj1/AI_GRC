# from qdrant_client import QdrantClient

# # in-memory DB
# client = QdrantClient(path="./data")

# print("Qdrant running ✅")


# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="BAAI/bge-large-en-v1.5",
#     local_dir="./models/bge-large",
#     token=True  # uses your env token
# )

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-large",
    local_dir="./models/bge-reranker-large",
    token=True 
)


# from langchain_community.llms import Ollama

# llm = Ollama(model="mistral-small3.2")

# response = llm.invoke("Generate ISO 42001 checklist")

# print(response)