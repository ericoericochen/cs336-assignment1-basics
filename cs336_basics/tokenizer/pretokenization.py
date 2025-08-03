import os
import regex as re
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def read_delimited_chunks_from_file(
    input_path: str, special_tokens: list[str], chunk_size: int = 4096
):
    with open(input_path) as f:
        prev_chunk = ""
        while True:
            chunk = f.read(chunk_size)
            mini_chunks = re.split("|".join(special_tokens), chunk)

            for mini_chunk in mini_chunks[:-1]:
                yield mini_chunk

            break


def pretokenize(input_path: str, special_tokens: list[str], num_workers: int = 1):
    for x in read_delimited_chunks_from_file(input_path, special_tokens=special_tokens):
        print(x)
        print("=" * 100)


if __name__ == "__main__":
    pretokens = pretokenize(input_path, special_tokens=["<|endoftext|>"], num_workers=2)
    print(pretokens)


text = """u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>


Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
"Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
"No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
"Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
"Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
"We're okay, Mommy. But our toys are broken," Lily said.
"I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
"Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
<|endoftext|>"""


import regex as re

print(text.count("<|endoftext|>"))

yo = re.split(r"<\|endoftext\|>", text)
print(yo)
print(len(yo))

print(re.escape("<|endoftext|>"))
