"""Download the tiny-shakespeare corpus to input.txt (about 1 MB).
Run once:  python prepare_data.py
Any plain-text file saved as input.txt works too."""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(__file__), "input.txt")

if os.path.exists(OUT):
    print("input.txt already present")
else:
    try:
        urllib.request.urlretrieve(URL, OUT)
        print(f"downloaded {os.path.getsize(OUT)} bytes -> input.txt")
    except Exception as e:
        print("download failed:", e)
        print("Manually save any .txt corpus as input.txt in this folder.")
