import os
import argparse
import glob

from domainbed import datasets
from domainbed.lib import reporting

import collections

import json
import os
import csv

import tqdm

from domainbed.lib.query import Q

def load_records(path):
    records = []
    for i, outer_dir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        outer_path = os.path.join(path, outer_dir)
        for j, inner_dir in tqdm.tqdm(list(enumerate(os.listdir(outer_path))),
                            ncols=80,
                            leave=False):
            results_path = os.path.join(outer_path, inner_dir, "results.jsonl")
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass
    return Q(records)



def get_results(records, dataset, algorithm):
    envs = range(datasets.num_environments(dataset))
    def get_accs(train_env, env_records):
        #test_envs = [env for env in envs if env != train_env]
        results = [0] * len(envs)
        for test_env in envs:
            test_key = "env%d_out_acc" % test_env
            val_envs = [env for env in envs if env != train_env and env != test_env]
            val_keys=",".join(["env%d_in_acc" % env for env in val_envs])
            val_fct = lambda step, step_group: {"val_acc": step_group.select(val_keys).mean(), "test_acc": step_group.select(test_key).mean()}
            val_accs = env_records.group_map("step", val_fct)
            results[test_env] = val_accs.argmax("val_acc")["test_acc"]
        return results
    #assuming for now we have data for just one algorithm and dataset
    filtered_records = records.filter_equals("dataset,algorithm", (dataset, algorithm))
    return filtered_records.group_map("args.train_env", get_accs)







    


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default="results/summary")
parser.add_argument("--dataset", type=str, default="VLCS")
parser.add_argument("--algorithm", type=str, default="ERM")

args = parser.parse_args()

records = reporting.load_records(args.result_dir)
results = get_results(records, args.dataset, args.algorithm)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

outfile = os.path.join(args.result_dir, "%s_%s_summary.csv" % (args.dataset, args.algorithm))
with open(outfile, 'w') as writer:
    csvwriter = csv.writer(writer, delimeter=',')
    for record in results:
        csvwriter.writerow(record)








