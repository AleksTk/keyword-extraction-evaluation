from pprint import pprint
from typing import List, Dict

from metrics import Metrics, RefTag

metrics = Metrics(case_sensitive=False, only_important=False, partial_score=0.5)


def eval_file_tags(reference_tag_list: List[RefTag], predicted_tags: List[str]) -> Dict[str, float]:
    return metrics.compute(reference_tag_list, predicted_tags)


reference_tag_list = [RefTag(name="innovation",
                             is_important=True,
                             alternatives=["innovation services"],
                             partial_alternatives=[]),
                      RefTag(name="digital technologies",
                             is_important=True,
                             alternatives=["Digital Transformation Imperative", "Digital transformation"],
                             partial_alternatives=["technology"]),
                      RefTag(name="artificial intelligence",
                             is_important=True,
                             alternatives=["embedded artificial intelligence"],
                             partial_alternatives=[]),
                      ]

predicted_tags = ["innovation", "artificial intelligence"]

file_scores = eval_file_tags(reference_tag_list, predicted_tags)
pprint(file_scores)
