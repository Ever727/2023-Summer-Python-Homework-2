### TODO: Implement metrics Perplexity, Rouge-L, etc.
###
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
encoded_sequence = enc.encode_ordinary('A')
converted_sequence = np.array(encoded_sequence, dtype=np.uint16)

print(converted_sequence)