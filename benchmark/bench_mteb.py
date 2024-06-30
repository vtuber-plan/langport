from typing import Any
import mteb
import numpy as np
import torch
from tqdm import tqdm

# Define the sentence-transformers model name
model_name = "gte-Qwen2-7B-instruct"

import openai

client = openai.OpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key="",
)

def batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

class EmbeddingModel():
    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        ret = []
        for sent in tqdm(batched(sentences, 16), total=len(sentences)//16):
            response = client.embeddings.create(
                model=model_name,
                input=sent,
                encoding_format="float",
            )
            for embed_data in response.data:
                embed_final = embed_data.embedding
                ret.append(embed_final)
        return ret

model = EmbeddingModel()
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"mteb_results/{model_name}")