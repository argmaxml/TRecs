import json, re
import numpy as np
from tree_helpers import lowest_depth, get_values_nested

def parse_schema():
    ret = []
    return ret

class ColumnEncoder:
    column=''
    column_weight=1
    def encode(self, value):
        return np.array([])

class OneHotEncoder(ColumnEncoder):
    values=['a','b','c']
    def encode(self, value):
        vec = np.zeros(1+len(self.values))
        try:
            vec[1+self.values.index(value)] = 1
        except ValueError: # Unknown
            vec[0] = 1
        return vec

class OrdinalEncoder(ColumnEncoder):
    values=['a','b','c']
    window=[0.5,1,0.5]
    def encode(self, value):
        assert len(window)%1==1, "Window size should be odd"
        vec = np.zeros(1+len(self.values))
        try:
            ind = self.values.index(value)
        except ValueError: # Unknown
            vec[0] = 1
            return vec
        vec[1+ind] = window[len(self.window)//2+1]
        for offset in range(len(self.window)//2):
            if ind-offset>=0:
                vec[1+ind-offset] = self.window[len(self.window)//2-offset]
            if ind+offset<len(self.values):
                vec[1+ind+offset] = self.window[len(self.window)//2+offset]

        return vec

class BinEncoder(ColumnEncoder):
    boundaries = [1,2,3]
    def encode(self, value):
        vec = np.zeros(2+len(self.boundaries))
        i = 0
        while i<len(self.boundaries) and value>self.boundaries[i]:
            i+=1
        vec[i]=1

class HierarchyEncoder(ColumnEncoder):
    values={'a':['a1','a2'],'b':['b1', 'b2'],'c':{'c1':['c11','c12']}}
    similarity_by_depth = [1,0.5,0]

    def encode(self, value):
        #TODO: very inefficient: move to constructor
        inner_values = get_values_nested(self.values)
        vec = np.zeros(1+len(inner_values))
        try:
            for other_value in inner_values:
                depth = lowest_depth(self.values,value,other_value)
                if depth>=len(self.similarity_by_depth):
                    # defaults to zero
                    continue
                vec[1+inner_values.index(other_value)] = self.similarity_by_depth[depth]
        except ValueError: # Unknown
            vec[0] = 1
        return vec

