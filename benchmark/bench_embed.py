import argparse
import random
import time
import openai
import threading
import tqdm
from concurrent.futures import ThreadPoolExecutor

def start_session(i: int, url: str, model: str) -> str:
  try:
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = url

    # create a chat completion
    response = openai.Embedding.create(
      model=model,
      input="Hello! What is your name?",
    )
    # print the completion
    out = response.data[0].embedding
    total_tokens = response.usage.total_tokens
    completion_tokens = 0
  except Exception as e:
    print(e)
    return "", 0, 0
  
  return out, total_tokens, completion_tokens

def main(args):
  tik = time.time()
  tasks = []
  with ThreadPoolExecutor(max_workers=args.n_thread) as t:
    for i in range(args.total_task):
      task = t.submit(start_session, i=i, url=args.url, model=args.model_name)
      tasks.append(task)
    
    results = []
    for task in tqdm.tqdm(tasks):
      results.append(task.result())

  n_tokens = sum([ret[1] for ret in results])
  n_queries = sum([1 for ret in results if ret[1] != 0])
  time_seconds = time.time() - tik
  print(
      f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
      f"throughput: {n_tokens / time_seconds} tokens/s."
      f"QPS: {n_queries / time_seconds} queries/s."
  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default="vicuna")
    parser.add_argument("--total-task", type=int, default=200)
    parser.add_argument("--n-thread", type=int, default=32)
    args = parser.parse_args()

    main(args)