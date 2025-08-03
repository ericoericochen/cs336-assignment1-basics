import os
import regex as re
from tqdm import tqdm
from typing import BinaryIO
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


input_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../tests/fixtures/tinystories_sample_5M.txt"
    )
)
# input_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../../data/TinyStoriesV2-GPT4-train.txt")
# )

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_text(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    freq: dict[tuple[bytes], int] = defaultdict(int)
    chunks = re.split(
        "|".join([re.escape(special_token) for special_token in special_tokens]), text
    )

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken = match.group(0).encode("utf-8")
            freq[pretoken] += 1

    return freq


def stream_file_chunks(input_path: str, num_chunks: int):
    with open(input_path, mode="rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def pretokenize(
    input_path: str, special_tokens: list[str], num_processes: int = 1
) -> dict[bytes, int]:
    """
    Pretokenize the corpus at `input_path` with `num_processes` and returns
    a dict containing the pretokens and their frequency. The special tokens
    are removed from the corpus before pretokenization.
    """

    freq: dict[tuple[bytes], int] = defaultdict(int)

    worker = partial(pretokenize_text, special_tokens=special_tokens)
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    with ProcessPoolExecutor() as executor:
        # parallelize pretokenization by dividing corpus into chunks
        # then merging the pretoken frequency together
        filestream = stream_file_chunks(input_path, num_processes)
        for mini_freq in executor.map(worker, filestream):
            for k, v in mini_freq.items():
                freq[k] += v

    return freq
