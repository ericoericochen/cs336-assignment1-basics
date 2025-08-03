import regex as re
from tqdm import tqdm
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

PairwiseFrequency = dict[tuple[bytes, bytes], int]
PretokenFrequency = dict[tuple[bytes], int]


def bpe_merge(
    pair: tuple[bytes, bytes],
    vocab: Vocabulary,
    freq: PretokenFrequency,
    pairwise_freq: PairwiseFrequency,
):
    """
    Executes one merge step of BPE training.
    - Merge pair of tokens into one token
    - Add new token to vocab
    - Update pretokens using the new token
    - Update pairwise frequency to include new token

    Mutates `vocab`, `frequ`, and `pairwise_freq`.
    """
    merged_token = pair[0] + pair[1]
    vocab.add(merged_token)

    # update each word with new merged token
    pretokens = list(freq.keys())
    for pretoken in pretokens:
        count = freq[pretoken]
        new_pretoken = pretoken
        if len(pretoken) > 1:
            new_pretoken = [pretoken[0]]
            for i in range(1, len(pretoken)):
                token = pretoken[i]
                curr_pair = (new_pretoken[-1], token)

                # found the pair of tokens to merge
                if curr_pair == pair:
                    new_pretoken.pop()

                    # check if the token in a pair with the first token in our merged pair
                    if len(new_pretoken) > 0:
                        pairwise_freq[(new_pretoken[-1], pair[0])] -= count
                        pairwise_freq[(new_pretoken[-1], merged_token)] += count

                    new_pretoken.append(merged_token)

                    # decrement prev pair of tokens
                    pairwise_freq[pair] -= count
                else:
                    # check if the curr token is in a pair with the last token in our merged pair
                    if len(new_pretoken) > 0 and new_pretoken[-1] == merged_token:
                        pairwise_freq[(pair[1], token)] -= count
                        pairwise_freq[(merged_token, token)] += count
                    new_pretoken.append(token)

            new_pretoken = tuple(new_pretoken)

        # update freq with new pretoken
        count = freq[pretoken]
        del freq[pretoken]
        freq[new_pretoken] = count

    return vocab, freq, pairwise_freq


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    show_tqdm: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train Byte Pair Encoding (BPE) on the text in `input_path`"""

    vocab = Vocabulary(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # parallized pretokenization

    with open(input_path) as f:
        dataset = f.read()

    print(dataset)
    raise RuntimeError

    freq: dict[tuple[bytes], int] = defaultdict(int)

    # protokenization: split by special tokens and get pretokens
    chunks = re.split("|".join(special_tokens), dataset)

    texts = []
    for chunk in chunks:
        chunk_texts = re.findall(PAT, chunk)
        texts.extend(chunk_texts)

    # count frequency of each pretoken
    # each pretoken is a sequence of utf-8 bytes, initially a sequence of 1 bytes which
    # are merged during bpe training
    for text in texts:
        pretoken_bytes = tuple(bytes([byte]) for byte in text.encode("utf-8"))
        freq[pretoken_bytes] += 1

    num_merges = vocab_size - len(vocab)

    # count up frequency of pairwise tokens
    pairwise_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pretoken, count in freq.items():
        if len(pretoken) > 1:
            for i in range(len(pretoken) - 1):
                pairwise_tokens = (pretoken[i], pretoken[i + 1])
                pairwise_freq[pairwise_tokens] += count

    merges_iter = range(num_merges)
    if show_tqdm:
        merges_iter = tqdm(merges_iter, desc="Merges")

    for m in merges_iter:
        # get pairwise tokens with the highest frequency
        max_pairwise_count = max(pairwise_freq.values())
        pair = max([k for k, v in pairwise_freq.items() if v == max_pairwise_count])

        # bpe merge step
        merges.append(pair)
        vocab, freq, pairwise_freq = bpe_merge(pair, vocab, freq, pairwise_freq)

    return vocab.as_dict(), merges
