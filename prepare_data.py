"""Fetch a corpus to input.txt.

    python prepare_data.py                    # tiny-shakespeare, ~1.1 MB (default)
    python prepare_data.py --dataset shakespeare   # complete works, ~5.4 MB

  tiny        - the original tiny-shakespeare.
  shakespeare - the COMPLETE works: ~5x more text from the SAME domain, which is
                the clean way to test "more data" without also changing the
                distribution. tiny-shakespeare is a subset of this.

Any plain-text file saved as input.txt also works."""
import argparse
import os
import urllib.request

DATASETS = {
    "tiny": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "shakespeare": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
}
OUT = os.path.join(os.path.dirname(__file__), "input.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="tiny")
    args = ap.parse_args()
    req = urllib.request.Request(DATASETS[args.dataset], headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
        with open(OUT, "wb") as f:
            f.write(data)
        print(f"{args.dataset}: {len(data):,} bytes -> input.txt")
    except Exception as e:
        print("download failed:", e)
        print("Manually save any .txt corpus as input.txt in this folder.")


if __name__ == "__main__":
    main()
