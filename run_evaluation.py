"""
Usage: python run_evaluation.py <REFERENCE-DIR> <PREDICTIONS-DIR> <OUTPUT_FILE>
"""
import sys
import os
import json
from glob import glob

import pandas as pd

import metrics


def read_file_reference_tags(filename):
    tags_json = json.load(open(filename))['tags']
    tags = []
    for tag_entry in tags_json:
        r_a = tag_entry['representative_alternatives'] if tag_entry['representative_alternatives'] else []
        v_a = tag_entry['visual_alternatives'] if tag_entry['visual_alternatives'] else []
        alternatives = list(set(r_a + v_a))

        p_a = tag_entry['partial_alternatives'] if tag_entry['partial_alternatives'] else []

        tags.append(metrics.RefTag(name=tag_entry['tag'],
                                   is_important=bool(tag_entry['importance']),
                                   canonical_name=tag_entry['canonical_name'],
                                   alternatives=alternatives,
                                   partial_alternatives=p_a))
    return tags


def main():
    ref_dir = sys.argv[1]
    pred_dir = sys.argv[2]
    output_file = sys.argv[3]

    m = metrics.Metrics(case_sensitive=False, only_important=False, partial_score=0.5)

    scores_list = []
    for i, ref_file in enumerate(glob(f"{ref_dir}/*.json")):
        ref_tags = read_file_reference_tags(ref_file)

        pred_file = os.path.join(pred_dir, os.path.basename(ref_file).replace('.json', '.mp4.txt'))
        predicted_tags = open(pred_file).read().strip().split(';')
        if predicted_tags[-1] == "":
            predicted_tags = predicted_tags[:-1]

        scores = m.compute(ref_tags, predicted_tags)

        scores['file'] = os.path.basename(ref_file)
        scores['ref_tags'] = ','.join([t.name for t in ref_tags])
        scores['pred_tags'] = ','.join(predicted_tags)

        scores_list.append(scores)

        print(i, "File:", scores['file'])
        print("reference tags:", scores['ref_tags'])
        print("predicted tags:", scores['pred_tags'])
        print(pd.DataFrame([scores]).iloc[:, :-3].to_string())
        print()

    df = pd.DataFrame(scores_list)

    print("Mean scores:")
    print(df.mean(0).to_string())

    df.to_csv(output_file, index=None)
    print("Saved file scores to", output_file)


if __name__ == '__main__':
    main()
