import json, logging, itertools
import requests
import numpy as np
import pandas as pd
from operator import itemgetter as at
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from joblib import delayed, Parallel
from encoders import parse_schema
from similarity_helpers import parse_server_name

def index_one_partition(p, model_dir, sname, **sim_params):
    with p.with_suffix('.meta').open('r') as f:
        meta = json.load(f)
    Index = parse_server_name(sname)
    index_instance = Index(meta["metric"], meta["dim"], **sim_params)
    try:
        index_instance.load_index(str(model_dir/str(meta["index_num"])))
    except:
        pass
    arr=np.load(p)
    ids=np.arange(meta["start_num_idx"],meta["start_num_idx"]+meta["size"])
    if len(arr)==0:
        return []
    index_instance.add_items(arr,ids)
    index_instance.save_index(str(model_dir/str(meta["index_num"])))
    return [(int(i),str(l)) for i,l in zip(ids,meta["ids"])]


@delayed
def index_several_partition(paths, model_dir, sname, **sim_params):
    ret = []
    for p in paths:
        ret.extend(index_one_partition(p, model_dir, sname, **sim_params))
    return ret


def main(args):
    with Path(args.config_file).open('r') as f:
        config = json.load(f)
    Index = parse_server_name(config["similarity_engine"])
    sim_params=config[config["similarity_engine"]]
    partition_dir = Path(args.partition_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    logging.debug("Copy Schema")
    with (partition_dir/"schema.json").open('r') as i:
        with (model_dir/"schema.json").open('w') as o:
            json.dump(json.load(i),o)

    logging.debug("Start Index")
    start = datetime.now()
    #index_labels = Parallel(-1)([index_one_partition(p,model_dir,config["similarity_engine"],**config[config["similarity_engine"]]) for p in partition_dir.glob("*.npy")])
    partition_data = sorted([(p.name.rsplit(args.partition_sep,1)[0],p) for p in partition_dir.glob("*.npy")])
    partition_data = [[p for kk, p in grp] for k,grp in itertools.groupby(partition_data, key=at(0))]
    index_labels = Parallel(-1)([index_several_partition(ps,model_dir,config["similarity_engine"],**config[config["similarity_engine"]]) for ps in partition_data])
    index_labels = sorted(sum(index_labels,[]))
    index_labels = [l for i,l in index_labels]
    logging.debug("Save labels")
    with (model_dir/"index_labels.json").open('w') as f:
        json.dump(index_labels,f)

    end = datetime.now()
    logging.debug("Took {s} seconds to index".format(s=(end-start).seconds))
    return 0


if __name__=="__main__":
    import sys
    from argparse import ArgumentParser
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    argparse = ArgumentParser()
    argparse.add_argument('--config_file',   default=str(Path(__file__).absolute().parent.parent / "data" / "config.json")  ,type=str, help='config file')
    argparse.add_argument('--model_dir',     default=str(Path(__file__).absolute().parent.parent / f"models/test-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"), type=str, help='model dir')
    argparse.add_argument('--partition_dir', default=str(Path(__file__).absolute().parent.parent / "data/ny/partitioned")   ,type=str, help='partition dir')
    argparse.add_argument('--partition_sep', default='~'   ,type=str, help='partition separator')
    sys.exit(main(argparse.parse_args()))

