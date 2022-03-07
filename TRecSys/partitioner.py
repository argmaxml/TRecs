import json, itertools, collections, sys
from operator import itemgetter as at
import numpy as np
from pathlib import Path
src = Path(__file__).absolute().parent
sys.path.append(str(src))
from encoders import PartitionSchema
from similarity_helpers import parse_server_name, FlatFaiss

class Partitioner:
    __slots__ = ["schema", "partitions","index_labels", "model_dir", "IndexEngine", "engine_params"]
    def __init__(self, config=None):
        if config is None:
            self.IndexEngine = FlatFaiss
            self.engine_params = {}
        else:
            self.engine_params=config[config["similarity_engine"]]
            self.IndexEngine = parse_server_name(config["similarity_engine"])
        self.model_dir = Path(__file__).absolute().parent.parent / "models"
        self.partitions = None
        self.schema =  None
        self.index_labels = []


    def init_schema(self, encoders, filters, metric):
        self.schema = PartitionSchema(encoders, filters, metric)
        self.partitions = [self.IndexEngine(self.schema.metric, self.schema.dim, **self.engine_params) for _ in self.schema.partitions]
        enc_sizes = {k:len(v) for k,v in self.schema.encoders.items()}
        return self.schema.partitions, enc_sizes

    def index(self, data):
        errors = []
        vecs = []
        for datum in data:
            try:
                vecs.append((self.schema.partition_num(datum), self.schema.encode(datum), datum["id"]))
            except KeyError as e:
                errors.append((datum, str(e)))
        vecs = sorted(vecs, key=at(0))
        affected_partitions = 0
        labels = set(self.index_labels)
        for partition_num, grp in itertools.groupby(vecs, at(0)):
            _, items, ids = zip(*grp)
            for id in ids:
                if id in labels:
                    errors.append((datum, f"{id} already indexed."))
                    continue # skip labels that already exists
                else:
                    labels.add(id)
                    self.index_labels.append(id)
            affected_partitions += 1
            num_ids = list(map(self.index_labels.index, ids))
            self.partitions[partition_num].add_items(items, num_ids)
        return errors, affected_partitions

    def query_by_partition_and_vector(self, partition_num, vec, k, explain=False):
        try:
            vec = vec.reshape(1,-1).astype('float32') # for faiss
            distances, num_ids = self.partitions[partition_num].search(vec, k=k)
        except Exception as e:
            raise Exception("Error in querying: " + str(e))
        if len(num_ids) == 0:
            labels, distances = [], []
        else:
            labels = [self.index_labels[n] for n in num_ids[0]]
            distances = [float(d) for d in distances[0]]
        if not explain:
            return labels,distances, []

        vec = vec.reshape(-1)
        explanation = []
        X = self.partitions[partition_num].get_items(num_ids[0])
        first_sim = None
        for ret_vec in X:
            start=0
            explanation.append({})
            for col,enc in self.schema.encoders.items():
                if enc.column_weight==0:
                    explanation[-1][col] = float(enc.column_weight)
                    continue
                end = start + len(enc)
                ret_part = ret_vec[start:end]
                query_part =   vec[start:end]
                if self.schema.metric=='l2':
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

    def query(self, data, k, explain=False):
        try:
            partition_nums = self.schema.partition_num(data)
        except Exception as e:
            raise Exception("Error in partitioning: " + str(e))
        try:
            vec = self.schema.encode(data)
        except Exception as e:
            raise Exception("Error in encoding: " + str(e))
        if type(partition_nums)!=list:
            return self.query_by_partition_and_vector(partition_nums, vec, k, explain)
        # Aggregate results if multiple partitions are returned:
        labels,distances,explanation = [], [], []
        for partition_num in partition_nums:
            l,d,e = self.query_by_partition_and_vector(partition_num, vec, k, explain)
            labels.extend(l)
            distances.extend(d)
            #TODO: explanation is not supported when having multiple filters
        labels,distances = zip(*sorted(zip(labels,distances), key=at(1)))
        return labels,distances,explanation



    def save_model(self, model_name):
        (self.model_dir/model_name).mkdir(parents=True, exist_ok=True)
        with (self.model_dir/model_name/"index_labels.json").open('w') as f:
            json.dump(self.index_labels, f)
        with (self.model_dir/model_name/"schema.json").open('w') as f:
            json.dump(self.schema.to_dict(), f)
        saved=0
        for i,p in enumerate(self.partitions):
            fname = str(self.model_dir/model_name/str(i))
            try:
                p.save_index(fname)
                saved+=1
            except:
                continue
        return {"status": "OK", "saved_indices": saved}

    def load_model(self, model_name):
        with (self.model_dir/model_name/"schema.json").open('r') as f:
            schema_dict=json.load(f)
        schema = PartitionSchema(*schema_dict)
        partitions = [self.IndexEngine(schema.metric, schema.dim, **self.engine_params) for _ in self.partitions]
        (self.model_dir/model_name).mkdir(parents=True, exist_ok=True)
        with (self.model_dir/model_name/"index_labels.json").open('r') as f:
            self.index_labels=json.load(f)
        loaded = 0
        for i,p in enumerate(partitions):
            fname = str(self.model_dir/model_name/str(i))
            try:
                p.load_index(fname)
                loaded+=1
            except:
                continue
        return loaded, schema_dict

    def list_models(self):
        ret = [d.name for d in self.model_dir.iterdir() if d.is_dir()]
        ret.sort()
        return ret

    def fetch(self, lbls, numpy=False):
        sil = set(self.index_labels)
        found = [l for l in lbls if l in sil]
        ids = [self.index_labels.index(l) for l in found]
        ret = collections.defaultdict(list)
        for p,pn in zip(self.partitions, self.schema.partitions):
            for id in ids:
                try:
                    if numpy:
                        ret[pn].extend(p.get_items([id]))
                    else:
                        ret[pn].extend([tuple(float(v) for v in vec) for vec in p.get_items([id])])
                except Exception as e:
                    # not found
                    pass
        ret = map(lambda k,v: (k[0],v) if len(k)==1 else (str(k), v),ret.keys(), ret.values())
        ret = dict(filter(lambda kv: bool(kv[1]),ret))
        return ret

    def encode(self, data):
        return self.schema.encode(data)

    def schema_initialized(self):
        return (self.schema is not None)

    def get_partition_stats(self):
        display = lambda t: str(t[0]) if len(t)==1 else str(t)
        max_elements  = {display(p):self.partitions[i].get_max_elements()  for i,p in enumerate(self.get_partitions())}
        element_count = {display(p):self.partitions[i].get_current_count() for i,p in enumerate(self.get_partitions())}
        return {"max_elements": max_elements, "element_count":element_count, "n": len(self.partitions)}

    def get_partitions(self):
        return self.schema.partitions

    def get_embedding_dimension(self):
        return self.schema.dim

    def get_total_items(self):
        return len(self.index_labels)

class AvgUserPartitioner(Partitioner):
    def user_partition_mapping(self, user_metadata):
        # Assumes same features as the item
        return self.schema.partition_num(user_metadata)

    def user_query(self, user_metadata, user_histories, k):
        user_partition_num = self.user_partition_mapping(user_metadata)
        labels,distances = [], []
        for user_history in user_histories:
            vec = np.mean([v for vs in self.fetch(user_history, numpy=True).values() for v in vs], axis=0)
            item_labels,item_distances,_ = self.query_by_partition_and_vector(user_partition_num, vec, k)
            labels.extend(item_labels)
            distances.extend(item_distances)
        return labels,distances

