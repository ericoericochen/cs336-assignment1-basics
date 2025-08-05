from typing import Iterable, Iterator

from .pretokenization import get_pretokens


def get_pairs(tokens: tuple[bytes]) -> set[tuple[bytes]]:
    """Get all pairwise tokens from a sequence of tokens"""
    all_pairs = set()

    for i in range(len(tokens) - 1):
        all_pairs.add((tokens[i], tokens[i + 1]))

    return all_pairs


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.token_to_id = {t: i for i, t in vocab.items()}
        self.merges = merges
        self.merges_to_idx = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        # print("encoding pretoken")
        # print(pretoken)

        tokens = tuple(bytes([token]) for token in pretoken)

        # repeat until no merges can be applied
        while True:
            # get all pairwise tokens
            pairs = get_pairs(tokens)

            # find applicable merges
            applicable_merges = [pair for pair in pairs if pair in self.merges_to_idx]
            # print("applicable merges")
            if len(applicable_merges) == 0:
                break

            # apply the merge that is first to be created
            merge_to_apply = min(applicable_merges, key=lambda p: self.merges_to_idx[p])
            new_tokens = [tokens[0]]

            for i in range(1, len(tokens)):
                pair = (new_tokens[-1], tokens[i])
                if pair == merge_to_apply:
                    new_tokens.pop()
                    new_tokens.append(pair[0] + pair[1])
                else:
                    new_tokens.append(tokens[i])

            tokens = tuple(new_tokens)

        # print(tokens)
        ids = [self.token_to_id[token] for token in tokens]
        # print(ids)

        # raise RuntimeError

        return ids

    def encode(self, text: str) -> list[int]:
        # print("encoding yo")
        # print(text)

        pretokens = get_pretokens(text, self.special_tokens)
        ids = []

        for pretoken in pretokens:
            pretoken_ids = self._encode_pretoken(pretoken)
            ids.extend(pretoken_ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        decoded = ""

        for id in ids:
            token = self.vocab[id].decode("utf-8")
            decoded += token

        return decoded
