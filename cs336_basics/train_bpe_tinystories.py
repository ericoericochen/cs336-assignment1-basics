import cProfile
import os
import time

from cs336_basics.tokenizer.bpe import train_bpe


def main():
    input_path = os.path.abspath(
        # os.path.join(
        #     os.path.dirname(__file__), "../tests/fixtures/tinystories_sample_5M.txt"
        # )
        os.path.join(os.path.dirname(__file__), "../data/TinyStoriesV2-GPT4-train.txt")
        # os.path.join(os.path.dirname(__file__), "../data/TinyStoriesV2-GPT4-train.txt")
    )
    print(input_path)
    VOCAB_SIZE = 10000

    # print("[INFO] Training Slow BPE on TinyStories")
    # start_time = time.time()
    # vocab, merges = train_bpe_slow(
    #     input_path=input_path,
    #     vocab_size=VOCAB_SIZE,
    #     special_tokens=["<|endoftext|>"],
    #     num_processes=128,
    # )
    # end_time = time.time()

    # slow_execution_time = end_time - start_time
    # print(f"[INFO] Took {slow_execution_time / 1000}s")

    print("[INFO] Training Fast BPE on TinyStories")
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        num_processes=128,
    )
    end_time = time.time()

    fast_execution_time = end_time - start_time
    print(f"[INFO] Took {fast_execution_time}s")


if __name__ == "__main__":
    # cProfile.run("main()")
    main()
