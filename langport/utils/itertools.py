

def batched(data, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]