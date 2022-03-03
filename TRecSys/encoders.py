import json, re, itertools, collections, os
from copy import deepcopy as clone
from operator import itemgetter as at
import numpy as np
from tree_helpers import lowest_depth, get_values_nested
import requests
from smart_open import open

class PartitionSchema:
    __slots__=["encoders", "filters", "partitions", "dim", "metric"]
    def __init__(self, schema):
        if hasattr(schema, 'read'):
            schema = schema.read()
        if type(schema) == str:
            schema = json.loads(schema)
        assert type(schema) == dict, "Schema type should be a dict"
        assert "filters" in schema, "filters not in schema"
        assert "encoders" in schema, "encoders not in schema"
        self.metric = schema["metric"]
        self.filters = [f["field"] for f in schema["filters"]]
        self.partitions = list(itertools.product(*[f["values"] for f in schema["filters"]]))
        encoder = dict()
        for enc in schema["encoders"]:
            if enc["type"] in ["onehot", "one_hot", "one hot", "oh"]:
                encoder[enc["field"]] = OneHotEncoder(column=enc["field"], column_weight=enc["weight"],
                                                    values=enc["values"])
            elif enc["type"] in ["strictonehot", "strict_one_hot", "strict one hot", "soh"]:
                encoder[enc["field"]] = StrictOneHotEncoder(column=enc["field"], column_weight=enc["weight"],
                                                            values=enc["values"])
            elif enc["type"] in ["num", "numeric"]:
                encoder[enc["field"]] = NumericEncoder(column=enc["field"], column_weight=enc["weight"],
                                                    values=enc["values"])
            elif enc["type"] in ["ordinal", "ordered"]:
                encoder[enc["field"]] = OrdinalEncoder(column=enc["field"], column_weight=enc["weight"],
                                                    values=enc["values"], window=enc["window"])
            elif enc["type"] in ["bin", "binning"]:
                encoder[enc["field"]] = BinEncoder(column=enc["field"], column_weight=enc["weight"], values=enc["values"])
            elif enc["type"] in ["bino", "bin_ordinal", "bin ordinal", "ord bin"]:
                encoder[enc["field"]] = BinOrdinalEncoder(column=enc["field"], column_weight=enc["weight"], values=enc["values"],window=enc["window"])
            elif enc["type"] in ["hier", "hierarchy", "nested"]:
                encoder[enc["field"]] = HierarchyEncoder(column=enc["field"], column_weight=enc["weight"],
                                                        values=enc["values"],
                                                        similarity_by_depth=enc["similarity_by_depth"])
            elif enc["type"] in ["numpy", "np", "embedding"]:
                encoder[enc["field"]] = NumpyEncoder(column=enc["field"], column_weight=enc["weight"],
                                                            values=enc["values"], url=enc["url"])
            elif enc["type"] in ["JSON", "json", "js"]:
                encoder[enc["field"]] = JSONEncoder(column=enc["field"], column_weight=enc["weight"],
                                                            values=enc["values"], length=enc["length"])
            elif enc["type"] in ["qwak"]:
                encoder[enc["field"]] = QwakEncoder(column=enc["field"], column_weight=enc["weight"],
                                                            length=enc["length"], entity_name=enc["entity"],
                                                            feature_name=enc["feature"], environment=enc["environment"])
            else:
                raise TypeError("Unknown type {t} in field {f}".format(f=enc["field"], t=enc["type"]))
        self.encoders = encoder
        self.dim = sum(map(len, filter(lambda e: e.column_weight!=0, encoder.values())))
    def encode(self, x):
        return np.concatenate([e(x[f]) for f, e in self.encoders.items() if e.column_weight!=0])
    def partition_num(self, x):
        t = at(*(self.filters))(x)
        if type(t)!=tuple:
            t=(t,)
        return self.partitions.index(t)


class BaseEncoder:

    def __init__(self, **kwargs):
        self.column = ''
        self.column_weight = 1
        self.nonzero_elements=1
        self.__dict__.update(kwargs)

    def __len__(self):
        raise NotImplementedError("len is not implemented")

    def __call__(self, value):
        return self.column_weight * self.encode(value) * np.ones(len(self)) * (1/np.sqrt(self.nonzero_elements))

    def encode(self, value):
        raise NotImplementedError("encode is not implemented")

class CachingEncoder(BaseEncoder):
    cache_max_size=1024

    def __init__(self, **kwargs):
        # defaults:
        self.column = ''
        self.column_weight = 1
        self.values = []
        self.nonzero_elements=1
        # override from kwargs
        self.__dict__.update(kwargs)
        #caching
        self.cache={}
        self.cache_hits=collections.defaultdict(int)

    def __call__(self, value):
        """Calls encode, multiplies by weight, cached"""
        if value in self.cache:
            self.cache_hits[value]+=1
            return self.cache[value]
        ret = self.encode(value)*self.column_weight*np.ones(len(self)) * (1/np.sqrt(self.nonzero_elements))
        if (self.cache_max_size is None) or (len(self.cache)<self.cache_max_size):
            self.cache[value]=ret
            return ret
        # cache cleanup
        min_key = sorted([(v,k) for k,v in self.cache_hits.items()])[0][1]
        del self.cache_hits[min_key]
        del self.cache[min_key]
        self.cache[value]=ret
        return ret

    def flush_cache(self,new_size=1024):
        self.cache_max_size=new_size
        self.cache={}
        self.cache_hits=collections.defaultdict(int)



class NumericEncoder(BaseEncoder):
    def __len__(self):
        return 1
    def encode(self, value):
        return np.array([value])

class OneHotEncoder(CachingEncoder):

    def __len__(self):
        return len(self.values) + 1

    def encode(self, value):
        vec = np.zeros(1 + len(self.values))
        try:
            vec[1 + self.values.index(value)] = 1
        except ValueError:  # Unknown
            vec[0] = 1
        return vec

class StrictOneHotEncoder(CachingEncoder):
    def __len__(self):
        return len(self.values)

    def encode(self, value):
        vec = np.zeros(len(self.values))
        try:
            vec[self.values.index(value)] = 1
        except ValueError:  # Unknown
            pass
        return vec

class OrdinalEncoder(OneHotEncoder):
    def __init__(self, column, column_weight, values, window):
        super().__init__(column = column, column_weight=column_weight, values=values, window = window)
        self.window = window
        self.nonzero_elements=len(window)

    def encode(self, value):
        assert len(self.window) % 2 == 1, f"Window size should be odd: window: {self.window}, value: {value}"
        vec = np.zeros(1 + len(self.values))
        try:
            ind = self.values.index(value)
        except ValueError:  # Unknown
            vec[0] = 1
            return vec
        vec[1 + ind] = self.window[len(self.window) // 2 + 1]
        for offset in range(len(self.window) // 2):
            if ind - offset >= 0:
                vec[1 + ind - offset] = self.window[len(self.window) // 2 - offset]
            if ind + offset < len(self.values):
                vec[1 + ind + offset] = self.window[len(self.window) // 2 + offset]

        return vec


class BinEncoder(CachingEncoder):

    def __len__(self):
        return len(self.values) + 1

    def encode(self, value):
        vec = np.zeros(1 + len(self.values))
        i = 0
        while i < len(self.values) and value > self.values[i]:
            i += 1
        vec[i] = 1


class BinOrdinalEncoder(BinEncoder):
    def __init__(self, column, column_weight, values, window):
        super().__init__(column = column, column_weight=column_weight, values=values, window = window)
        self.window = window
        self.nonzero_elements=len(window)

    def encode(self, value):
        vec = np.zeros(1 + len(self.values))
        ind = 0
        while ind < len(self.values) and value > self.values[ind]:
            ind += 1
        vec[ind] = self.window[len(self.window) // 2 + 1]
        for offset in range(len(self.window) // 2):
            if ind - offset >= 0:
                vec[ind - offset] = self.window[len(self.window) // 2 - offset]
            if ind + offset < len(self.values):
                vec[ind + offset] = self.window[len(self.window) // 2 + offset]



class HierarchyEncoder(CachingEncoder):
    #values = {'a': ['a1', 'a2'], 'b': ['b1', 'b2'], 'c': {'c1': ['c11', 'c12']}}
    similarity_by_depth = [1, 0.5, 0]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Currently this approximation of the nonzero elements assumes:
        # (1) Two level hirearchy
        # (2) Approximately equal size of each hierarchy/category
        # {"A":[1,2,3], "B":[4,5,6]}
        inner_values = len(get_values_nested(self.values))
        outer_values = len(self.values.keys())
        self.nonzero_elements = inner_values / outer_values

    def __len__(self):
        inner_values = get_values_nested(self.values)
        return (1 + len(inner_values))

    def encode(self, value):
        # TODO: very inefficient: move to constructor
        inner_values = get_values_nested(self.values)
        vec = np.zeros(1 + len(inner_values))
        try:
            for other_value in inner_values:
                depth = lowest_depth(self.values, value, other_value)
                if depth >= len(self.similarity_by_depth):
                    # defaults to zero
                    continue
                vec[1 + inner_values.index(other_value)] = self.similarity_by_depth[depth]
        except ValueError:  # Unknown
            vec[0] = 1
        return vec

class NumpyEncoder(BaseEncoder):
    def __init__(self, column, column_weight, values, url):
        super().__init__(column = column, column_weight=column_weight, values=values, url=url)
        with open(url, 'rb') as f:
            data = np.load(f)
            self.ids = list(data["ids"])
            self.embedding = data["embedding"]
        assert self.embedding.shape[0]==len(self.ids), "Dimension mismatch between ids and embedding"
        self.nonzero_elements=self.embedding.shape[1]

    def __len__(self):
        return self.embedding.shape[1]

    def encode(self, value):
        try:
            idx = self.ids.index(value)
        except ValueError:
            return np.zeros(self.embedding.shape[1])
        return self.embedding[idx,:]

class JSONEncoder(CachingEncoder):

    def __len__(self):
        return self.length

    def encode(self, value):
        val = value.translate({ord('('):'[',ord(')'):']'})
        val = json.loads(val)
        if type(val)==dict: # sparse vector
            vec = np.zeros(len(self))
            for k,v in val.items():
                vec[int(k)]=v
        elif type(val)==list:
            vec = np.array(val)
        else:
            raise TypeError(str(type(val))+ " is not supported")
        return vec

class QwakEncoder(BaseEncoder):
    def __init__(self, column, column_weight, environment, length, entity_name, api_key=None):
        super().__init__(column = column, column_weight=column_weight, length=length,
            entity_name=entity_name, environment=environment)
        if api_key is None:
            api_key = os.environ.get("QWAK_API")
        self.init_access_token(api_key)

    def init_access_token(self, api_key):
        self.access_token = requests.post("https://grpc.qwak.ai/api/v1/authentication/qwak-api-key", json={"qwakApiKey": api_key}).json()["accessToken"]

    def get_feature(self, entity_value):
        url = "https://api."+str(self.environment)+".qwak.ai/api/v1/features"
        body = {
            "features":[{"batchFeature":{"name": self.column}}],
            "entity": {"name": self.entity_name, "value": entity_value}
        }
        res = requests.post(url, headers={"Authorization": "Bearer "+self.access_token}, json=body).json()
        # Assuming one returned feature
        res = list(res["featureValues"][0]["featureValue"].values())[0]
        # TODO: Do qwak support vector features ?
        return res



    def __len__(self):
        return self.length

    def json_encode(self, value):
        val = value.translate({ord('('):'[',ord(')'):']'})
        val = json.loads(val)
        if type(val)==dict: # sparse vector
            vec = np.zeros(len(self))
            for k,v in val.items():
                vec[int(k)]=v
        elif type(val)==list:
            vec = np.array(val)
        else:
            raise TypeError(str(type(val))+ " is not supported")
        return vec

    def encode(self, value):
        val = self.get_feature(value)
        return self.json_encode(val)

