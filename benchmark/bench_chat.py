import argparse
import random
import time
import openai
import threading
import tqdm
from concurrent.futures import ThreadPoolExecutor

def start_session(i: int, url: str, model: str, stream: bool=False, max_tokens: int=2048, random_len: int=0) -> str:
  try:
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = url

    if random_len != 0 :
      messages = [{"role": "user", "content": "Hello! What is your name?" + "a" * random.randint(1, random_len)}]
    else:
      messages = [{"role": "user", "content": "Hello! What is your name?"}]
    # create a chat completion
    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      stream=stream,
      max_tokens=max_tokens,
    )
    # print the completion
    if stream:
      out = ""
      for chunk in response:
        out += str(chunk)
    else:
      out = response.choices[0].message.content
      total_tokens = response.usage.total_tokens
      completion_tokens = response.usage.completion_tokens
  except Exception as e:
    print(e)
    return "", 0, 0
  
  return out, total_tokens, completion_tokens

def main(args):
  tik = time.time()
  tasks = []
  with ThreadPoolExecutor(max_workers=args.n_thread) as t:
    for i in range(args.total_task):
      task = t.submit(start_session, i=i, url=args.url, model=args.model_name, stream=False, max_tokens=args.max_tokens, random_len=args.random_len)
      tasks.append(task)
    
    results = []
    for task in tqdm.tqdm(tasks):
      results.append(task.result())

  n_tokens = sum([ret[2] for ret in results])
  n_queries = sum([1 for ret in results if ret[2] != 0])
  time_seconds = time.time() - tik
  print(
      f"Successful number: {n_queries} / {args.total_task}. "
      f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
      f"throughput: {n_tokens / time_seconds} tokens/s."
      f"QPS: {n_queries / time_seconds} queries/s."
  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default="vicuna")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--total-task", type=int, default=200)
    parser.add_argument("--n-thread", type=int, default=32)
    parser.add_argument("--random-len", type=int, default=0)
    args = parser.parse_args()

    main(args)