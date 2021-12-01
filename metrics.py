"""
Evaluation metrics for tag annotations.

Implemented metrics:

    Average precision
    Normalized Discounted cumulative gain
    R-precision
    Precision @ 5, 15, 10, 20

See `https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)` for details.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class RefTag:
    name: str
    is_important: bool = True
    canonical_name: str = None
    alternatives: List[str] = field(default_factory=list)
    partial_alternatives: List[str] = field(default_factory=list)


@dataclass
class RefTagset:
    tag2pos: Dict[str, int]
    partial_tag2pos: Dict[str, int]
    tag_names: List[str]
    n: int


class Metrics:
    def __init__(self, case_sensitive=False, only_important=False, partial_score=0.5):
        self.case_sensitive = case_sensitive
        self.only_important = only_important
        self.partial_score = partial_score

    def compute(self, ref_tag_list: List[RefTag], pred_tags: List[str]):
        if self.only_important:
            ref_tag_list = [t for t in ref_tag_list if t.is_important]
        if not self.case_sensitive:
            pred_tags = [t.lower() for t in pred_tags]
        ref_tagset = self.preprocess_ref_tags(ref_tag_list)
        return OrderedDict({
            'r_p': self.r_precision_score(ref_tagset, pred_tags, self.partial_score),
            'p': self.precision_score(ref_tagset, pred_tags, self.partial_score, k=len(pred_tags)),
            'p@5': self.precision_score(ref_tagset, pred_tags, self.partial_score, k=5),
            'p@10': self.precision_score(ref_tagset, pred_tags, self.partial_score, k=10),
            'p@15': self.precision_score(ref_tagset, pred_tags, self.partial_score, k=15),
            'p@20': self.precision_score(ref_tagset, pred_tags, self.partial_score, k=20),
            'ap': self.ap(ref_tagset, pred_tags, self.partial_score),
            'dcg-bin': self.dcg(ref_tagset, pred_tags, self.partial_score, binary=True),
            'dcg-ord': self.dcg(ref_tagset, pred_tags, self.partial_score, binary=False),
            'ndcg-bin': self.ndcg(ref_tagset, pred_tags, self.partial_score, binary=True),
            'ndcg-ord': self.ndcg(ref_tagset, pred_tags, self.partial_score, binary=False),
        })

    def preprocess_ref_tags(self, ref_tags: List[RefTag]) -> RefTagset:
        tag2pos = {}
        partial_tag2pos = {}
        norm_case = lambda t: t if self.case_sensitive else t.lower()
        for i, t in enumerate(ref_tags):
            tag2pos[norm_case(t.name)] = i
            for tag in t.alternatives:
                tag2pos[norm_case(tag)] = i
            for tag in t.partial_alternatives:
                partial_tag2pos[norm_case(tag)] = i
        tag_names = [norm_case(t.name) for t in ref_tags]
        ref_inst = RefTagset(tag2pos, partial_tag2pos, tag_names, len(ref_tags))
        return ref_inst

    @staticmethod
    def r_precision_score(ref_tags: RefTagset, predicted_tags: List[str], partial_weight: float):
        """Computes r-precision score"""
        hits = [0.] * ref_tags.n
        predicted_tags = predicted_tags[:ref_tags.n]
        for tag in predicted_tags:
            if tag in ref_tags.partial_tag2pos:
                hits[ref_tags.partial_tag2pos[tag]] = partial_weight
        for tag in predicted_tags:
            if tag in ref_tags.tag2pos:
                hits[ref_tags.tag2pos[tag]] = 1.
        return sum(hits) / float(ref_tags.n)

    @staticmethod
    def precision_score(ref_tagset: RefTagset, predicted_tags: List[str], partial_weight: float, k: int):
        """Computes precision @ k"""
        if k == 0:
            return 0.0
        predicted_tags = predicted_tags[:k]
        hits = [0.] * ref_tagset.n
        for tag in predicted_tags:
            if tag in ref_tagset.partial_tag2pos:
                hits[ref_tagset.partial_tag2pos[tag]] = partial_weight
        for tag in predicted_tags:
            if tag in ref_tagset.tag2pos:
                hits[ref_tagset.tag2pos[tag]] = 1.
        return sum(hits) / float(k)

    @staticmethod
    def dcg(ref_tagset: RefTagset, predicted_tags: List[str], partial_score: float, binary: bool):
        """Discounted cumulative gain"""
        ref_hits = set()
        scores = np.zeros(len(predicted_tags), dtype=float)
        for i, tag in enumerate(predicted_tags):
            if tag in ref_tagset.tag2pos:
                pos = ref_tagset.tag2pos[tag]
                if pos in ref_hits:
                    continue
                else:
                    scores[i] = 1. if binary else 1. / (pos + 1)
                    ref_hits.add(pos)
            elif tag in ref_tagset.partial_tag2pos:
                pos = ref_tagset.partial_tag2pos[tag]
                if pos in ref_hits:
                    continue
                else:
                    scores[i] = partial_score if binary else partial_score / (pos + 1)
                    ref_hits.add(pos)
        discounts = np.log2(np.arange(len(scores)) + 2)
        return np.sum(scores / discounts)

    @staticmethod
    def ndcg(ref_tagset: RefTagset, predicted_tags: List[str], partial_score: float, binary: bool):
        """Normalized discounted cumulative gain"""
        dcg = Metrics.dcg(ref_tagset, predicted_tags, partial_score, binary)
        idcg = Metrics.dcg(ref_tagset, ref_tagset.tag_names, partial_score, binary)
        return dcg / idcg

    @staticmethod
    def ap(ref_tagset: RefTagset, predicted_tags, partial_weight):
        """Computes average precision score.
        For details check https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision"""
        # compute average precision
        score = 0.
        num_hits = 0.
        used_pos_set = set()
        for i, tag in enumerate(predicted_tags, start=1):
            if tag in ref_tagset.tag2pos:
                pos = ref_tagset.tag2pos[tag]
                if pos not in used_pos_set:
                    num_hits += 1.0
                    used_pos_set.add(pos)
                else:
                    continue
            elif tag in ref_tagset.partial_tag2pos:
                pos = ref_tagset.partial_tag2pos[tag]
                if pos not in used_pos_set:
                    num_hits += partial_weight
                    used_pos_set.add(pos)
                else:
                    continue
            else:
                continue

            pk = num_hits / i
            score += pk
        return score / ref_tagset.n
