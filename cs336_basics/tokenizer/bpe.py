import regex as re
from collections import defaultdict


class Vocabulary:
    """Mapping of token ids (ints) to tokens (sequence of utf-8 bytes)"""

    def __init__(self, special_tokens: list[str] = []):
        """
        Args
            special_tokens: list of special tokens to include in the vocabulary
        """
        # initial tokens in vocab - special tokens + byte strings 0 - 255
        tokens = [*special_tokens, *(chr(i) for i in range(256))]

        self.tokens_to_token_ids: dict[bytes, int] = {}
        self.token_ids_to_tokens: dict[int, bytes] = {}

        for i, token in enumerate(tokens):
            token_bytes = token.encode("utf-8")
            self.tokens_to_token_ids[token_bytes] = i
            self.token_ids_to_tokens[i] = token_bytes

    def __len__(self):
        return len(self.tokens_to_token_ids)

    def get_token(self, token_id: int):
        return self.token_ids_to_tokens.get(token_id, None)

    def get_token_id(self, token: bytes):
        return self.tokens_to_token_ids.get(token, None)

    def add(self, token: bytes):
        pass

    def as_dict(self):
        return {**self.token_ids_to_tokens}


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train Byte Pair Encoding (BPE) on the text in `input_path`"""
    print("training bpe tokenizer")

    vocab = Vocabulary(special_tokens)
    merges = []

    with open(input_path) as f:
        dataset = f.read()

    frequency: dict[tuple[bytes], int] = defaultdict(int)

    # count frequency of special token
    for special_token in special_tokens:
        count = dataset.count(special_token)
        pretoken = special_token.encode("utf-8")
        if count > 0:
            frequency[(pretoken,)] = count

        # replace special token
        dataset = dataset.replace(special_token, "")

    # pretokens = re.findall(PAT, dataset)
    pretokens = dataset.split()

    # count frequency of each pretoken
    for pretoken in pretokens:
        token_bytes = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
        frequency[token_bytes] += 1

    num_merges = vocab_size - len(vocab)

    for m in range(num_merges):
        # count up frequency of pairwise tokens
        pairwise_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pretoken, freq in frequency.items():
            if len(pretoken) > 1:
                for i in range(len(pretoken) - 1):
                    pairwise_tokens = (pretoken[i], pretoken[i + 1])
                    pairwise_freq[pairwise_tokens] += freq

        # get pairwise tokens with the highest frequency
        max_pairwise_count = max(pairwise_freq.values())
        highest_freq_pairwise_tokens = max(
            [k for k, v in pairwise_freq.items() if v == max_pairwise_count]
        )

        # merge pairwise token into one token and add to vocab

        # merge

        break

    return None, merges


vocab, merges = train_bpe(
    input_path="/Users/ericchen/Eric/cs336/cs336-assignment1-basics/cs336_basics/test.txt",
    vocab_size=256 + 1 + 6,
    special_tokens=["<|endoftext|>"],
)
