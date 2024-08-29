import faiss
import datasets
import csv
import transformers
import torch
import numpy as np

from datasets import load_dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import warnings
warnings.filterwarnings("ignore")

rag_data = "datasets"

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

llm_model = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": "../model"},
    device_map="auto",
)

def get_index(rag_data_path):
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=rag_data_path: True
    # dataset will be downloaded at the first time of running this code
    ds = load_dataset(path='wiki_dpr', name='psgs_w100.multiset.compressed', split='train', cache_dir=rag_data_path)
    faiss_index = ds.get_index('embeddings').faiss_index

    return ds, faiss_index

class RAG:
    def __init__(self, llm, rag_data_path): 
        self.llm = llm  
        self.ds, self.index = get_index(rag_data_path)

    def embed_text(self, text):
        embeded_text = q_encoder(**q_tokenizer(text, return_tensors="pt"))[0][0].detach().numpy()
        return embeded_text

    def retrieve_documents(self, query_embedding, k=5):
        # Retrieve top k similar documents for the query
        indices = self.index.search(np.array([query_embedding]), k)
        indices = np.array(indices)
        return indices.flatten().tolist()[5:]

    def generate_response(self, constrain, claim, retrieved_doc_ids):
        try:
            # Combine retrieved documents with the query
            references = self.prepare_input(claim, retrieved_doc_ids)
            messages=[
                {"role": "system", "content": constrain},
                {"role": "system", "content": references},
                {"role": "user", "content": claim}
            ]  
            # print(messages)                    
            # Generate a response
            response = self.llm(
                messages,
                pad_token_id=128001,
                max_new_tokens=256,
            )  
            return response[0]["generated_text"][-1]['content']
        except Exception as e:
            return str(e)

    def prepare_input(self, query, doc_ids):
        # Retrieve and combine document texts based on their IDs with the query
        docs = []
        for id in doc_ids: 
            docs.append(self.ds[int(id)]['text'])
        references = " ".join(docs)
        return references

    def answer_query(self, constrain, claim):
        claim_embedding = self.embed_text(claim)
        retrieved_doc_ids = self.retrieve_documents(claim_embedding)
        response = self.generate_response(constrain, claim, retrieved_doc_ids)
        print(response)

        return response

def rag(constrain, claim, rag_data_path):
    rag = RAG(llm_model, rag_data_path)
    response = rag.answer_query(constrain, claim)
    return response
