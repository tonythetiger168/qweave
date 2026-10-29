"""
QL5 data: knowledge + reasoning mixture for calibration and heal
================================================================
QL1.5 calibrated and healed on general web text only, and the compressed
model kept commonsense but lost knowledge (MMLU -13 pt) and multi-step math
(GSM8K -34 pt) first (F4). QL5 makes the data that decides *which* singular
directions survive -- and that the heal restores -- contain those skills:

  source     dataset (train splits only)                      default weight
  fineweb    HuggingFaceFW/fineweb-edu  sample-10BT                 0.50
  owm        open-web-math/open-web-math                            0.15
  metamath   meta-math/MetaMathQA      (GSM8K/MATH *train* rewrites) 0.15
  gsm8k      openai/gsm8k main train   (question + worked answer)    0.05
  mmlu_aux   cais/mmlu auxiliary_train (ARC/OBQA/RACE/... train)     0.15

Contamination guard: every document is checked against 13-gram hashes of
the GSM8K and MMLU *test* sets (`Decontam`) and dropped on any hit; the
report records how many were dropped. Windows are packed per source so a
window carries one source id, which lets the heal up-weight reasoning
sources in the loss (`--source-weight`).

Without network (`synthetic=True`) every source becomes a seeded random
token stream: the CPU smoke test exercises the same packing code.
"""

import hashlib, random, re
from typing import Dict, Iterator, List, Optional, Tuple

DEFAULT_MIX = "fineweb:0.5,owm:0.15,metamath:0.15,gsm8k:0.05,mmlu_aux:0.15"
NGRAM = 13


def parse_mix(spec: str) -> Dict[str, float]:
    out = {}
    for part in spec.split(','):
        if part.strip():
            k, v = part.split(':')
            out[k.strip()] = float(v)
    tot = sum(out.values())
    assert tot > 0, spec
    return {k: v / tot for k, v in out.items()}


# ----------------------------------------------------------------------
def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_hashes(text: str, n: int = NGRAM):
    w = _words(text)
    for i in range(len(w) - n + 1):
        yield hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest()


class Decontam:
    """13-gram overlap filter against evaluation test sets."""

    def __init__(self, texts=()):
        self.h = set()
        for t in texts:
            self.add(t)
        self.checked = self.dropped = 0

    def add(self, text: str):
        self.h.update(_ngram_hashes(text))

    def clean(self, text: str) -> bool:
        self.checked += 1
        if self.h and any(g in self.h for g in _ngram_hashes(text)):
            self.dropped += 1
            return False
        return True

    @classmethod
    def from_eval_sets(cls):
        """GSM8K test + MMLU test questions (with choices)."""
        from datasets import load_dataset
        d = cls()
        for r in load_dataset("openai/gsm8k", "main", split="test"):
            d.add(r["question"] + " " + r["answer"])
        for r in load_dataset("cais/mmlu", "all", split="test"):
            d.add(r["question"] + " " + " ".join(r["choices"]))
        return d


# ----------------------------------------------------------------------
LETTERS = "ABCDEFGH"


def _fmt_mc(r) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
    return f"{r['question']}\n{choices}\nAnswer: {LETTERS[int(r['answer'])]}"


def _hf_docs(name: str, seed: int, rank: int, world: int) -> Iterator[str]:
    """Infinite iterator of documents from one source (streams, shuffled,
    split across data-parallel ranks)."""
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    spec = {
        'fineweb': (("HuggingFaceFW/fineweb-edu", "sample-10BT"), lambda r: r["text"]),
        'owm': (("open-web-math/open-web-math",), lambda r: r["text"]),
        'metamath': (("meta-math/MetaMathQA",),
                     lambda r: f"Question: {r['query']}\nAnswer: {r['response']}"),
        'gsm8k': (("openai/gsm8k", "main"),
                  lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}"),
        'mmlu_aux': (("cais/mmlu", "auxiliary_train"),
                     lambda r: _fmt_mc(r.get("train", r))),
    }[name]
    epoch = 0
    while True:
        ds = load_dataset(*spec[0], split="train", streaming=True)
        ds = ds.shuffle(seed=seed + 1000 * epoch, buffer_size=10_000)
        if world > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world)
        for r in ds:
            yield spec[1](r)
        epoch += 1


class MixturePacker:
    """Yields (window[seq+1] token list, source name). Each source keeps its
    own token buffer; the next window's source is drawn by weight."""

    def __init__(self, tokenizer, mix: Dict[str, float], seq: int, seed: int = 0,
                 rank: int = 0, world: int = 1, decontam: Optional[Decontam] = None,
                 synthetic: bool = False, vocab: int = 0):
        self.tok, self.mix, self.seq = tokenizer, mix, seq
        self.rng = random.Random(seed * 7919 + rank)
        self.decontam = decontam
        self.synthetic, self.vocab = synthetic, vocab
        self.names = list(mix)
        self.w = [mix[k] for k in self.names]
        self.buf = {k: [] for k in self.names}
        self.docs = {}
        self.counts = {k: 0 for k in self.names}
        self.eos = getattr(tokenizer, 'eos_token_id', None) if tokenizer else None
        for i, k in enumerate(self.names):
            if synthetic:
                self.docs[k] = None
            else:
                self.docs[k] = _hf_docs(k, seed + i, rank, world)

    def _refill(self, k):
        if self.synthetic:          # source-dependent token range -> distinct stats
            r = random.Random(self.rng.random())
            lo = (self.names.index(k) * self.vocab) // (2 * len(self.names))
            self.buf[k].extend(r.randrange(lo, self.vocab) for _ in range(4 * self.seq))
            return
        while True:
            text = next(self.docs[k])
            if self.decontam is None or self.decontam.clean(text):
                break
        ids = self.tok(text).input_ids
        self.buf[k].extend(ids + ([self.eos] if self.eos is not None else []))

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[int], str]:
        k = self.rng.choices(self.names, weights=self.w)[0]
        while len(self.buf[k]) < self.seq + 1:
            self._refill(k)
        win, self.buf[k] = self.buf[k][:self.seq + 1], self.buf[k][self.seq + 1:]
        self.counts[k] += 1
        return win, k
