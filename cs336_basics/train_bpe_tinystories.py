import cProfile
import os
import time
from cs336_basics.tokenizer.bpe import train_bpe


def main():
    input_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/TinyStoriesV2-GPT4-train.txt")
        # os.path.join(os.path.dirname(__file__), "../data/TinyStoriesV2-GPT4-train.txt")
    )
    print(input_path)
    VOCAB_SIZE = 10000

    print("[INFO] Training BPE on TinyStories")
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        num_processes=128,
    )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"[INFO] Took {execution_time / 1000}s")


if __name__ == "__main__":
    cProfile.run("main()")
