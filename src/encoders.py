import json, re, itertools
from copy import deepcopy as clone
from operator import itemgetter as at
import numpy as np
from tree_helpers import lowest_depth, get_values_nested


def parse_schema(schema):
    if hasattr(schema, 'read'):
        schema = schema.read()
    if type(schema) == str:
        schema = json.loads(schema)
    assert type(schema) == dict, "Schema type should be a dict"
    assert "filters" in schema, "filters not in schema"
    assert "encoders" in schema, "encoders not in schema"
    ret = {"metric": schema["metric"]}
    partitions = list(itertools.product(*[f["values"] for f in schema["filters"]]))
    ret["partitions"] = partitions
    tup = lambda t: t if type(t) == tuple else (t,)
    ret["index_num"] = lambda x: partitions.index(tup(at(*[f["field"] for f in schema["filters"]])(x)))
    encoder = dict()
    for enc in schema["encoders"]:
        if enc["type"] in ["onehot", "one_hot", "one hot", "oh"]:
            encoder[enc["field"]] = OneHotEncoder(column=enc["field"], column_weight=enc["weight"],
                                                  values=enc["values"])
        elif enc["type"] in ["ordinal", "ordered"]:
            encoder[enc["field"]] = OrdinalEncoder(column=enc["field"], column_weight=enc["weight"],
                                                   values=enc["values"], window=enc["window"])
        elif enc["type"] in ["bin", "binning"]:
            encoder[enc["field"]] = BinEncoder(column=enc["field"], column_weight=enc["weight"], values=enc["values"],
                                               boundaries=enc["boundaries"])
        elif enc["type"] in ["hier", "hierarchy", "nested"]:
            encoder[enc["field"]] = HierarchyEncoder(column=enc["field"], column_weight=enc["weight"],
                                                     values=enc["values"],
                                                     similarity_by_depth=enc["similarity_by_depth"])
        else:
            raise TypeError("Unknown type {t} in field {f}".format(f=enc["field"], t=enc["type"]))
    ret["encoders"] = encoder
    ret["encode_fn"] = lambda d: np.concatenate([e.encode(d[f]) for f, e in encoder.items()])
    ret["dim"] = sum(map(len, encoder.values()))
    return ret


class ColumnEncoder:
    column = ''
    column_weight = 1

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __len__(self):
        raise NotImplementedError("len is not implemented")

    def encode(self, value):
        return np.array([])


class OneHotEncoder(ColumnEncoder):
    values = ['a', 'b', 'c']

    def __len__(self):
        return len(self.values) + 1

    def encode(self, value):
        vec = np.zeros(1 + len(self.values))
        try:
            vec[1 + self.values.index(value)] = 1
        except ValueError:  # Unknown
            vec[0] = 1
        return vec


class OrdinalEncoder(ColumnEncoder):
    values = ['a', 'b', 'c']

    def __len__(self):
        return len(self.values) + 1

    window = [0.5, 1, 0.5]

    def encode(self, value):
        assert len(window) % 1 == 1, "Window size should be odd"
        vec = np.zeros(1 + len(self.values))
        try:
            ind = self.values.index(value)
        except ValueError:  # Unknown
            vec[0] = 1
            return vec
        vec[1 + ind] = window[len(self.window) // 2 + 1]
        for offset in range(len(self.window) // 2):
            if ind - offset >= 0:
                vec[1 + ind - offset] = self.window[len(self.window) // 2 - offset]
            if ind + offset < len(self.values):
                vec[1 + ind + offset] = self.window[len(self.window) // 2 + offset]

        return vec


class BinEncoder(ColumnEncoder):
    boundaries = [1, 2, 3]

    def __len__(self):
        return len(self.boundaries) + 1

    def encode(self, value):
        vec = np.zeros(2 + len(self.boundaries))
        i = 0
        while i < len(self.boundaries) and value > self.boundaries[i]:
            i += 1
        vec[i] = 1


class HierarchyEncoder(ColumnEncoder):
    values = {'a': ['a1', 'a2'], 'b': ['b1', 'b2'], 'c': {'c1': ['c11', 'c12']}}
    similarity_by_depth = [1, 0.5, 0]

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
