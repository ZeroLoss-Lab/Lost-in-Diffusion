# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import List

from utils import lm

def save_eval_raw(
        raw_eval_list: List[str], 
        output_file):
    # write raw evals to jsonl
    with open(output_file, "w") as f:
        for r in raw_eval_list:
            f.write(json.dumps({"eval_res": r}) + "\n")

def jsonify_ans(
        raw_responses: List[str],
        eval_prompts: List[str], 
        evaluator_model: str, 
        key: str):

    def check_validity(gen):
        gen_nospace = gen.replace(" ", "")
        if '{{"{}":false}}'.format(key) in gen_nospace:
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen_nospace:
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []
    for r, p in zip(raw_responses, eval_prompts):
        
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = r.split("\n")[0]
            try:
                jsonifyed_res.append(json.loads(r))
            except:
                print(f"Error in eval_answer: {r}")
                jsonifyed_res.append({key: True})
    return jsonifyed_res

def check_hallucination(hallu_responses):
    hallu_evals = []
    for x in hallu_responses:
        if 'INCORRECT' in x:
            hallu_evals.append("INCORRECT")
        elif 'CORRECT' in x:
            hallu_evals.append("CORRECT")
        elif 'UNVERIFIABLE' in x:
            hallu_evals.append("UNVERIFIABLE")
        else:
            hallu_evals.append("INCORRECT")
    return hallu_evals

    