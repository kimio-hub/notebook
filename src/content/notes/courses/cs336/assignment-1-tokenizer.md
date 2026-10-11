---
title: "CS336 Assignment 1：BPE 分词器实现"
description: "Byte-level BPE 的训练与编解码流程：special token 切分、regex 预分词、merge 学习，以及对应的实现代码。"
tags:
  - "cs336"
  - "tokenizer"
  - "bpe"
category: "courses"
series: "CS336"
date: "2026-07-22"
updated: "2026-07-22"
---

## tokenizer
  
```text

训练：

语料文本 -> special token 切分 -> regex 预分词 -> UTF-8 bytes

         -> 统计相邻 pair -> 学习 merges -> vocab

  

编码：

输入文本 -> special token 切分 -> regex 预分词 -> UTF-8 bytes

         -> 按训练顺序应用 merges -> token IDs

  

解码：

token IDs -> vocab 中的 bytes -> 拼接 -> UTF-8 字符串

```

导入、正则 和类型别名
```python
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path

import regex

PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
Pair = tuple[bytes, bytes]
Pieces = tuple[bytes, ...]
```

清理special token 
```python
def _specials(tokens: Sequence[str] | None) -> list[str]:
    result = list(dict.fromkeys(tokens or []))
    if "" in result:
        raise ValueError("special tokens cannot contain an empty string")
    return result
```
构造special token正则
```python
def _special_pattern(tokens: Sequence[str]) -> regex.Pattern[str] | None:
    if not tokens:
        return None
    return regex.compile("|".join(map(regex.escape, sorted(tokens, key=len, reverse=True))))
```
从文本中排除special token
```python
def _segments(text: str, pattern: regex.Pattern[str] | None) -> Iterator[str]:
    if pattern is None:
        yield text
        return
    start = 0
    for match in pattern.finditer(text):
        if start < match.start():
            yield text[start : match.start()]
        start = match.end()
    if start < len(text):
        yield text[start:]
```
进行一次merge
```python
def _merge(items: Pieces, pair: Pair) -> Pieces:
    result: list[bytes] = []
    index = 0
    while index < len(items):
        if index + 1 < len(items) and items[index : index + 2] == pair:
            result.append(pair[0] + pair[1])
            index += 2
        else:
            result.append(items[index])
            index += 1
    return tuple(result)
```

训练bpe
```python
def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    specials = _specials(special_tokens)
    if vocab_size < 256 + len(specials):
        raise ValueError("vocab_size is too small for bytes and special tokens")

    vocab = {index: bytes([index]) for index in range(256)}
    for token in specials:
        vocab[len(vocab)] = token.encode()

    counts: Counter[Pieces] = Counter()
    text = Path(input_path).read_text(encoding="utf-8")
    for segment in _segments(text, _special_pattern(specials)):
        for match in PATTERN.finditer(segment):
            counts[tuple(bytes([value]) for value in match.group().encode())] += 1
```
建立统计索引
```python
    merges: list[Pair] = []
    words = list(counts)
    frequencies = [counts[word] for word in words]
    word_pairs: list[Counter[Pair]] = []
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[int]] = {}

    for word_index, (word, frequency) in enumerate(zip(words, frequencies)):
        local_counts = Counter(zip(word, word[1:]))
        word_pairs.append(local_counts)
        for pair, occurrences in local_counts.items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words.setdefault(pair, set()).add(word_index)
```
每一轮 bpe merge
```python
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        for word_index in pair_to_words.get(best, set()).copy():
            frequency = frequencies[word_index]
            for pair, occurrences in word_pairs[word_index].items():
                pair_counts[pair] -= occurrences * frequency
                pair_to_words[pair].discard(word_index)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            words[word_index] = _merge(words[word_index], best)
            local_counts = Counter(zip(words[word_index], words[word_index][1:]))
            word_pairs[word_index] = local_counts
            for pair, occurrences in local_counts.items():
                pair_counts[pair] += occurrences * frequency
                pair_to_words.setdefault(pair, set()).add(word_index)
    return vocab, merges
```
tokenizer 初始化
```python
class Tokenizer:
    """Byte-level BPE tokenizer compatible with the CS336 tokenizer API."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[Pair],
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = _specials(special_tokens)
        self.token_to_id = {value: key for key, value in self.vocab.items()}

        next_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            value = token.encode()
            if value not in self.token_to_id:
                self.vocab[next_id] = value
                self.token_to_id[value] = next_id
                next_id += 1
        if any(bytes([value]) not in self.token_to_id for value in range(256)):
            raise ValueError("vocab must contain all 256 byte tokens")

        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_ids = {
            token: self.token_to_id[token.encode()] for token in self.special_tokens
        }
        self.special_pattern = _special_pattern(self.special_tokens)
        self._encode_piece.cache_clear()
```

编码pre-token
```python
    @lru_cache(maxsize=65_536)
    def _encode_piece(self, text: str) -> tuple[int, ...]:
        pieces: Pieces = tuple(bytes([value]) for value in text.encode())
        while len(pieces) > 1:
            pair = min(
                set(zip(pieces, pieces[1:])),
                key=lambda item: self.ranks.get(item, float("inf")),
            )
            if pair not in self.ranks:
                break
            pieces = _merge(pieces, pair)
        return tuple(self.token_to_id[piece] for piece in pieces)
```
文本编码
```python
    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in PATTERN.finditer(text):
            yield from self._encode_piece(match.group())

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))
        ids: list[int] = []
        start = 0
        for match in self.special_pattern.finditer(text):
            ids.extend(self._encode_ordinary(text[start : match.start()]))
            ids.append(self.special_ids[match.group()])
            start = match.end()
        ids.extend(self._encode_ordinary(text[start:]))
        return ids
```
流式编码和接码:
```python
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Sequence[int]) -> str:
        try:
            data = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error
        return data.decode("utf-8", errors="replace")
```
保存和加载:
```python
    def save(self, filepath: str | Path) -> None:
        data = {
            "vocab": {str(key): list(value) for key, value in self.vocab.items()},
            "merges": [[list(left), list(right)] for left, right in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: str | Path) -> Tokenizer:
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        vocab = {int(key): bytes(value) for key, value in data["vocab"].items()}
        merges = [(bytes(left), bytes(right)) for left, right in data["merges"]]
        return cls(vocab, merges, data["special_tokens"])
```
便携训练:
```python
class BPETokenizer(Tokenizer):
    def __init__(
        self,
        vocab_size: int = 5_000,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        specials = _specials(special_tokens)
        vocab = {index: bytes([index]) for index in range(256)}
        for token in specials:
            vocab[len(vocab)] = token.encode()
        if vocab_size < len(vocab):
            raise ValueError("vocab_size is too small")
        super().__init__(vocab, [], specials)

    def train(self, input_path: str | Path) -> None:
        vocab, merges = train_bpe(input_path, self.vocab_size, self.special_tokens)
        Tokenizer.__init__(self, vocab, merges, self.special_tokens)
```
