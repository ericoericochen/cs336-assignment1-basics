import cProfile
import time


from .tokenizer.bpe import train_bpe, train_bpe_slow


NUM_ITERATIONS = 3


def main():
    print("=" * 10, "Slow BPE Train", "=" * 10)
    start_time = time.time()
    for i in range(NUM_ITERATIONS):
        vocab, merges = train_bpe_slow(
            input_path="/Users/ericchen/Eric/cs336/cs336-assignment1-basics/tests/fixtures/corpus.en",
            vocab_size=500,
            special_tokens=["<|endoftext|>"],
        )
    end_time = time.time()

    slow_avg_time = (end_time - start_time) / 3
    print(f"Slow Execution Time: {slow_avg_time}ms")

    print("=" * 10, "Fast BPE Train", "=" * 10)
    start_time = time.time()
    for i in range(NUM_ITERATIONS):
        vocab, merges = train_bpe(
            input_path="/Users/ericchen/Eric/cs336/cs336-assignment1-basics/tests/fixtures/corpus.en",
            vocab_size=500,
            special_tokens=["<|endoftext|>"],
        )
    end_time = time.time()

    fast_avg_time = (end_time - start_time) / 3
    print(f"Fast Execution Time: {fast_avg_time}ms")

    print(f"Diff: {slow_avg_time - fast_avg_time}ms")


if __name__ == "__main__":
    # cProfile.run("main()")
    main()
