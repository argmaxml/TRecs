import json, itertools, collections
from operator import itemgetter as at
import numpy as np
from pathlib import Path
import encoders
from similarity_helpers import parse_server_name, FlatFaiss

model_dir = Path(__file__).absolute().parent.parent / "models"
partitions, schema, Index, sim_params = None, None, None, None
index_labels = []


def init_schema(schema_dict, config=None):
    global schema, partitions, Index, sim_params
    if config is None:
        Index = FlatFaiss
        sim_params = {}
    else:
        sim_params=config[config["similarity_engine"]]
        Index = parse_server_name(config["similarity_engine"])
    schema = encoders.parse_schema(schema_dict)
    partitions = [Index(schema["metric"], schema["dim"], **sim_params) for _ in schema["partitions"]]
    enc_sizes = {k:len(v) for k,v in schema["encoders"].items()}
    return schema["partitions"], enc_sizes

def index(data):
    errors = []
    vecs = []
    for datum in data:
        try:
            vecs.append((schema["index_num"](datum), schema["encode_fn"](datum), datum["id"]))
        except KeyError as e:
            errors.append((datum, str(e)))
    vecs = sorted(vecs, key=at(0))
    affected_partitions = 0
    labels = set(index_labels)
    for idx, grp in itertools.groupby(vecs, at(0)):
        _, items, ids = zip(*grp)
        for id in ids:
            if id not in labels:
                labels.add(id)
                index_labels.append(id)
        affected_partitions += 1
        num_ids = list(map(index_labels.index, ids))
        partitions[idx].add_items(items, num_ids)
    return errors, affected_partitions

def query(data, k, explain=False):
    try:
        idx = schema["index_num"](data)
    except Exception as e:
        raise Exception("Error in partitioning: " + str(e))
    try:
        vec = schema["encode_fn"](data)
    except Exception as e:
        raise Exception("Error in encoding: " + str(e))
    try:
        vec = vec.reshape(1,-1).astype('float32') # for faiss
        distances, num_ids = partitions[idx].search(vec, k=k)
    except Exception as e:
        raise Exception("Error in querying: " + str(e))
    if len(num_ids) == 0:
        labels, distances = [], []
    else:
        labels = [index_labels[n] for n in num_ids[0]]
        distances = [float(d) for d in distances[0]]
    if not explain:
        return labels,distances, []

    vec = vec.reshape(-1)
    explanation = []
    X = partitions[idx].get_items(num_ids[0])
    first_sim = None
    for ret_vec in X:
        start=0
        explanation.append({})
        for col,enc in schema["encoders"].items():
            if enc.column_weight==0:
                explanation[-1][col] = float(enc.column_weight)
                continue
            end = start + len(enc)
            ret_part = ret_vec[start:end]
            query_part =   vec[start:end]
            if schema["metric"]=='l2':
                # The returned distance from the similarity server is not squared
                dst=((ret_part-query_part)**2).sum()
            else:
                sim=np.dot(ret_part,query_part)
                # Correct dot product to be ascending
                if first_sim is None:
                    first_sim = sim
                    dst = 0
                else:
                    dst = 1-sim/first_sim
            explanation[-1][col]=float(dst)
            start = end
    return labels,distances, explanation


def save_model(model_name, schema_dict):
    (model_dir/model_name).mkdir(parents=True, exist_ok=True)
    with (model_dir/model_name/"index_labels.json").open('w') as f:
        json.dump(index_labels,f)
    with (model_dir/model_name/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    saved=0
    for i,p in enumerate(partitions):
        fname = str(model_dir/model_name/str(i))
        try:
            p.save_index(fname)
            saved+=1
        except:
            continue
    return {"status": "OK", "saved_indices": saved}

def load_model(model_name):
    global index_labels, partitions, schema
    with (model_dir/model_name/"schema.json").open('r') as f:
        schema_dict=json.load(f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [Index(schema["metric"], schema["dim"], **sim_params) for _ in partitions]
    (model_dir/model_name).mkdir(parents=True, exist_ok=True)
    with (model_dir/model_name/"index_labels.json").open('r') as f:
        index_labels=json.load(f)
    loaded = 0
    for i,p in enumerate(partitions):
        fname = str(model_dir/model_name/str(i))
        try:
            p.load_index(fname)
            loaded+=1
        except:
            continue
    return loaded, schema_dict

def list_models():
    ret = [d.name for d in model_dir.iterdir() if d.is_dir()]
    ret.sort()
    return ret

def fetch(lbls):
    found = set(lbls)&set(index_labels)
    ids = [index_labels.index(l) for l in found]
    ret = collections.defaultdict(list)
    for p,pn in zip(partitions,schema["partitions"]):
        try:
            ret[pn].extend([tuple(float(v) for v in vec) for vec in p.get_items(ids)])
        except:
            # not found
            continue
    ret = map(lambda k,v: (k[0],v) if len(k)==1 else (str(k), v),ret.keys(), ret.values())
    ret = dict(filter(lambda kv: bool(kv[1]),ret))
    return ret

def encode(data):
    return schema["encode_fn"](data)

def schema_initialized():
    return (schema is not None)

def get_partition_stats():
    display = lambda t: str(t[0]) if len(t)==1 else str(t)
    max_elements  = {display(p):partitions[i].get_max_elements()  for i,p in enumerate(partitioner.get_partitions())}
    element_count = {display(p):partitions[i].get_current_count() for i,p in enumerate(partitioner.get_partitions())}
    return {"max_elements": max_elements, "element_count":element_count, "n": len(partitions)}

def get_partitions():
    return schema["partitions"]

def get_embedding_dimension():
    return schema["dim"]

def get_total_items():
    return len(index_labels)
