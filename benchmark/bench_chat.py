import time
import openai
import threading
import tqdm
from joblib import Parallel, delayed

def start_session(i: int):
  try:
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = "http://localhost:8000/v1"

    model = "7b-bf16"
    # create a chat completion
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": "Hello! What is your name?"}]
    )
    # print the completion
    # print(completion.choices[0].message.content)
  except Exception as e:
    return 0
  
  return 1

if __name__ == "__main__":
  start_time = time.time()

  test_num = 100
  results = Parallel(n_jobs=64)(delayed(start_session)(i) for i in tqdm.tqdm(range(test_num), total=test_num))

  end_time = time.time()
  print(f"Successed samples: {sum(results)}")
  print(f"Speed: {sum(results) / (end_time - start_time)}")