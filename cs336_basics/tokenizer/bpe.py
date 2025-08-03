import os
import regex as re
from tqdm import tqdm
from collections import defaultdict

from .pretokenization import pretokenize, pretokenize_text


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

PairwiseFrequency = dict[tuple[int, int], int]
PretokenFrequency = dict[bytes, int]
PretokenToTokens = dict[bytes, tuple[int, ...]]


def bpe_merge(
    pair: tuple[int, int],  # pair of token ids
    vocab: Vocabulary,
    freq: PretokenFrequency,
    pretoken_to_tokens: PretokenToTokens,
    pairwise_freq: PairwiseFrequency,
):
    """
    Executes one merge step of BPE training.
    - Merge pair of tokens into one token
    - Add new token to vocab
    - Update pretokens using the new token
    - Update pairwise frequency to include new token

    Mutates `vocab`, `freq`, and `pairwise_freq`.
    """
    merged_token = pair[0] + pair[1]
    vocab.add(merged_token)

    # update each word with new merged token
    for pretoken, count in freq.items():
        tokens = pretoken_to_tokens[pretoken]
        new_tokens = tokens
        if len(tokens) > 1:
            new_tokens = [tokens[0]]
            for i in range(1, len(tokens)):
                token = tokens[i]
                curr_pair = (new_tokens[-1], token)

                # found the pair of tokens to merge
                if curr_pair == pair:
                    new_tokens.pop()

                    # check if the token in a pair with the first token in our merged pair
                    if len(new_tokens) > 0:
                        pairwise_freq[(new_tokens[-1], pair[0])] -= count
                        pairwise_freq[(new_tokens[-1], merged_token)] += count

                    new_tokens.append(merged_token)

                    # decrement prev pair of tokens
                    pairwise_freq[pair] -= count
                else:
                    # check if the curr token was in a pair with the last token in our merged pair
                    if len(new_tokens) > 0 and new_tokens[-1] == merged_token:
                        pairwise_freq[(pair[1], token)] -= count
                        pairwise_freq[(merged_token, token)] += count
                    new_tokens.append(token)

            new_tokens = tuple(new_tokens)

        # update freq with new pretoken
        pretoken_to_tokens[pretoken] = new_tokens


# def bpe_merge(
#     pair: tuple[bytes, bytes],
#     vocab: Vocabulary,
#     freq: PretokenFrequency,
#     pairwise_freq: PairwiseFrequency,
# ):
#     """
#     Executes one merge step of BPE training.
#     - Merge pair of tokens into one token
#     - Add new token to vocab
#     - Update pretokens using the new token
#     - Update pairwise frequency to include new token

#     Mutates `vocab`, `frequ`, and `pairwise_freq`.
#     """
#     merged_token = pair[0] + pair[1]
#     vocab.add(merged_token)

#     # update each word with new merged token
#     pretokens = list(freq.keys())
#     for pretoken in pretokens:
#         count = freq[pretoken]
#         new_pretoken = pretoken
#         if len(pretoken) > 1:
#             new_pretoken = [pretoken[0]]
#             for i in range(1, len(pretoken)):
#                 token = pretoken[i]
#                 curr_pair = (new_pretoken[-1], token)

#                 # found the pair of tokens to merge
#                 if curr_pair == pair:
#                     new_pretoken.pop()

#                     # check if the token in a pair with the first token in our merged pair
#                     if len(new_pretoken) > 0:
#                         pairwise_freq[(new_pretoken[-1], pair[0])] -= count
#                         pairwise_freq[(new_pretoken[-1], merged_token)] += count

#                     new_pretoken.append(merged_token)

#                     # decrement prev pair of tokens
#                     pairwise_freq[pair] -= count
#                 else:
#                     # check if the curr token is in a pair with the last token in our merged pair
#                     if len(new_pretoken) > 0 and new_pretoken[-1] == merged_token:
#                         pairwise_freq[(pair[1], token)] -= count
#                         pairwise_freq[(merged_token, token)] += count
#                     new_pretoken.append(token)

#             new_pretoken = tuple(new_pretoken)

#         # update freq with new pretoken
#         count = freq[pretoken]
#         del freq[pretoken]
#         freq[new_pretoken] = count

#     return vocab, freq, pairwise_freq


CPU_COUNT = os.cpu_count() or 1


# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
#     num_processes: int = CPU_COUNT,
#     show_tqdm: bool = True,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Train Byte Pair Encoding (BPE) on the text in `input_path`"""

#     vocab = Vocabulary(special_tokens)
#     merges: list[tuple[bytes, bytes]] = []

#     # parallized pretokenization - pretoken to their frequency
#     freq = pretokenize(
#         input_path, special_tokens=special_tokens, num_processes=num_processes
#     )

#     num_merges = vocab_size - len(vocab)

#     # count up frequency of pairwise tokens
#     pairwise_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
#     for pretoken, count in freq.items():
#         if len(pretoken) > 1:
#             for i in range(len(pretoken) - 1):
#                 pairwise_tokens = (pretoken[i], pretoken[i + 1])
#                 pairwise_freq[pairwise_tokens] += count

#     merges_iter = range(num_merges)
#     if show_tqdm:
#         merges_iter = tqdm(merges_iter, desc="Merges")

#     for m in merges_iter:
#         # get pairwise tokens with the highest frequency
#         max_pairwise_count = max(pairwise_freq.values())
#         pair = max([k for k, v in pairwise_freq.items() if v == max_pairwise_count])

#         # bpe merge step
#         merges.append(pair)
#         vocab, freq, pairwise_freq = bpe_merge(pair, vocab, freq, pairwise_freq)

#     return vocab.as_dict(), merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = CPU_COUNT,
    show_tqdm: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train Byte Pair Encoding (BPE) on the text in `input_path`"""

    vocab = Vocabulary(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # parallized pretokenization - pretoken to their frequency
    freq = pretokenize(
        input_path, special_tokens=special_tokens, num_processes=num_processes
    )

    # mapping from pretoken to tokens
    pretoken_to_tokens: PretokenToTokens = {
        pretoken: tuple(bytes([byte]) for byte in pretoken) for pretoken in freq.keys()
    }

    num_merges = vocab_size - len(vocab)

    # count up frequency of pairwise tokens
    pairwise_freq: PairwiseFrequency = defaultdict(int)
    for pretoken, count in freq.items():
        if len(pretoken) > 1:
            for i in range(len(pretoken) - 1):
                token1 = bytes([pretoken[i]])
                token2 = bytes([pretoken[i + 1]])

                # pairwise_tokens = (pretoken[i], pretoken[i + 1])
                pairwise_tokens = (token1, token2)
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
        bpe_merge(pair, vocab, freq, pretoken_to_tokens, pairwise_freq)

    return vocab.as_dict(), merges


# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
#     num_processes: int = CPU_COUNT,
#     show_tqdm: bool = True,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Train Byte Pair Encoding (BPE) on the text in `input_path`"""

#     vocab = Vocabulary(special_tokens)
#     merges: list[tuple[bytes, bytes]] = []

#     # parallized pretokenization - pretoken to their frequency
#     freq = pretokenize(
#         input_path, special_tokens=special_tokens, num_processes=num_processes
#     )

#     # mapping from pretoken to tokens
#     pretoken_to_tokens = {
#         pretoken: tuple(bytes([byte]) for byte in pretoken) for pretoken in freq.keys()
#     }

#     num_merges = vocab_size - len(vocab)

#     # count up frequency of pairwise tokens
#     pairwise_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
#     for pretoken, count in freq.items():
#         if len(pretoken) > 1:
#             for i in range(len(pretoken) - 1):
#                 pairwise_tokens = (pretoken[i], pretoken[i + 1])
#                 pairwise_freq[pairwise_tokens] += count

#     merges_iter = range(num_merges)
#     if show_tqdm:
#         merges_iter = tqdm(merges_iter, desc="Merges")

#     for m in merges_iter:
#         # get pairwise tokens with the highest frequency
#         max_pairwise_count = max(pairwise_freq.values())
#         pair = max([k for k, v in pairwise_freq.items() if v == max_pairwise_count])

#         # bpe merge step
#         merges.append(pair)
#         vocab, freq, pairwise_freq = bpe_merge(pair, vocab, freq, pairwise_freq)

#     return vocab.as_dict(), merges
