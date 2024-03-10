import os
import sys

name = sys.argv[1]
assert name in ["shediao", "shendiao", "tianlong"], "Unidentified dataset name!"

links = {
    "shendiao": "https://github.com/loyalpartner/jywxFenxi/raw/master/神雕侠侣.txt",
    "shediao": "https://github.com/loyalpartner/jywxFenxi/raw/master/射雕英雄传.txt",
    "tianlong": "https://github.com/loyalpartner/jywxFenxi/raw/master/天龙八部.txt"
}

os.makedirs(name, exist_ok=True)
os.system(f"wget --no-check-certificate {links[name]} -O {os.path.join(name, 'input.txt')}")
