import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = ['shediao', 'shendiao', 'tianlong']

### TODO: read data from ([name]/input.txt for name in names)
### TODO: combine multiple books into one single data file
### TODO: split data for train(0.9) and valid (0.1)

train_data, val_data = [], []

for name in names:
    file_path = os.path.join(r"D:\Program\Git\hw2\MiniGLM\data", name, "input.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        book_content = file.read()
        train_end_index = int(len(book_content) * 0.9)
        train_data.append(book_content[:train_end_index])
        val_data.append(book_content[train_end_index:])

###

### TODO: tokenize raw data with tiktoken encoder

train_tokens = [enc.encode_ordinary(text) for text in train_data]
val_tokens = [enc.encode_ordinary(text) for text in val_data]

### TODO: transform to numpy array

train_ids, val_ids = None, None

train_ids = np.hstack(train_tokens)
val_ids = np.hstack(val_tokens)

###

# save numpy array to file [name]/train.bin and [name]/val.bin
os.makedirs("processed_pretrain", exist_ok=True)
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
