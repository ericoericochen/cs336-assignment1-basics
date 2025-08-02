import regex as re
from collections import defaultdict


class Vocabulary:
    """Mapping of token ids (ints) to tokens (sequence of utf-8 bytes)"""

    def __init__(self, special_tokens: list[str] = []):
        """
        Args
            special_tokens: list of special tokens to include in the vocabulary
        """
        # add special tokens and first 256 utf-8 bytes to the vocab
        self.tokens_to_token_ids: dict[bytes, int] = {}
        self.token_ids_to_tokens: dict[int, bytes] = {}

        for i, token in enumerate(special_tokens):
            token_bytes = token.encode("utf-8")
            self.add(token_bytes)

        for i in range(256):
            token_bytes = bytes([i])
            self.add(token_bytes)

    def __len__(self):
        return len(self.tokens_to_token_ids)

    def get_token(self, token_id: int):
        return self.token_ids_to_tokens.get(token_id, None)

    def get_token_id(self, token: bytes):
        return self.tokens_to_token_ids.get(token, None)

    def add(self, token: bytes):
        assert token not in self.tokens_to_token_ids
        vocab_size = len(self)
        self.token_ids_to_tokens[vocab_size] = token
        self.tokens_to_token_ids[token] = vocab_size

    def as_dict(self):
        return {**self.token_ids_to_tokens}


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_vocab(
    pair: tuple[bytes, bytes], vocab: Vocabulary, frequency: dict[tuple[bytes], int]
):
    """Merge pair of tokens into one token, add to vocab, and update frequency with new tokens"""
    merged_frequency: dict[tuple[bytes], int] = defaultdict(int)
    merged_token = pair[0] + pair[1]
    vocab.add(merged_token)

    # update each word with new merged token
    pretokens = list(frequency.keys())
    for pretoken in pretokens:
        new_tokens = pretoken
        if len(pretoken) > 1:
            new_tokens = [pretoken[0]]
            for i in range(1, len(pretoken)):
                token = pretoken[i]
                curr_pair = (new_tokens[-1], token)
                if curr_pair == pair:
                    new_tokens.pop()
                    new_tokens.append(merged_token)
                else:
                    new_tokens.append(token)

            new_tokens = tuple(new_tokens)

        freq = frequency[pretoken]
        merged_frequency[new_tokens] = freq

    return vocab, merged_frequency


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train Byte Pair Encoding (BPE) on the text in `input_path`"""
    print("training bpe tokenizer")

    vocab = Vocabulary(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path) as f:
        dataset = f.read()

    frequency: dict[tuple[bytes], int] = defaultdict(int)

    # split by special tokens and get pretokens
    chunks = re.split("|".join(special_tokens), dataset)

    # pretokens = dataset.split()

    pretokens = []
    for chunk in chunks:
        chunk_pretokens = re.findall(PAT, chunk)
        pretokens.extend(chunk_pretokens)

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
        pair = max([k for k, v in pairwise_freq.items() if v == max_pairwise_count])

        # merge pairwise token into one token, add to vocab, and update freq
        merges.append(pair)
        vocab, frequency = merge_vocab(pair, vocab, frequency)

    return vocab.as_dict(), merges


# vocab, merges = train_bpe(
#     input_path="/Users/ericchen/Eric/cs336/cs336-assignment1-basics/cs336_basics/test.txt",
#     vocab_size=256 + 1 + 6,
#     special_tokens=["<|endoftext|>"],
# )
