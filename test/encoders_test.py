import unittest
import requests
import numpy as np
class StrictOneHot(unittest.TestCase):

    def setUp(self):
        schema = {
        "encoders":[
                    {"field": "enc1",  "values": ["a", "b", "c"],           "type": "soh", "weight": 3},
                    {"field": "enc2",  "values": ["w","x","y","z"],         "type": "soh", "weight": 4},
                    {"field": "enc3",  "values": ["1", "2", "3", "4", "5"], "type": "soh", "weight": 5},
                    {"field": "enc4",  "values": ["j", "i", "k"],           "type": "soh", "weight": 0}
                    ],
        "filters": [{"field": "country", "values": ["US", "EU"]}],
        "metric": "cosine"
        }
        requests.post("http://127.0.0.1:5000/init_schema", json=schema)

        index = [
                {"id": "us_a", "country": "US", "enc1": "a", "enc2": "w", "enc3": "1", "enc4":"k"},
                {"id": "us_b", "country": "US", "enc1": "b", "enc2": "x", "enc3": "2","enc4":"k"},
                {"id": "us_c", "country": "US", "enc1": "c", "enc2": "y", "enc3": "3","enc4":"k"},
                {"id": "us_d", "country": "US", "enc1": "a", "enc2": "z", "enc3": "4","enc4":"k"},
                {"id": "us_e", "country": "US", "enc1": "b", "enc2": "w", "enc3": "5","enc4":"k"},
                {"id": "us_f", "country": "US", "enc1": "c", "enc2": "x", "enc3": "1","enc4":"k"},
                {"id": "us_g", "country": "US", "enc1": "a", "enc2": "y", "enc3": "2","enc4":"k"},
                {"id": "us_h", "country": "US", "enc1": "b", "enc2": "z", "enc3": "3","enc4":"k"},
                {"id": "us_i", "country": "US", "enc1": "c", "enc2": "w", "enc3": "4", "enc4":"k"},
                {"id": "us_j", "country": "US", "enc1": "a", "enc2": "x", "enc3": "5","enc4":"k" },
                {"id": "us_k", "country": "US", "enc1": "b", "enc2": "y", "enc3": "1","enc4":"k"},
                {"id": "us_l", "country": "US", "enc1": "c", "enc2": "z", "enc3": "2","enc4":"k"},
                {"id": "us_m", "country": "US", "enc1": "a", "enc2": "w", "enc3": "3","enc4":"k"},
                {"id": "us_n", "country": "US", "enc1": "b", "enc2": "x", "enc3": "4","enc4":"k"},
                {"id": "us_o", "country": "US", "enc1": "c", "enc2": "y", "enc3": "5","enc4":"k"},
                {"id": "us_p", "country": "US", "enc1": "a", "enc2": "z", "enc3": "1","enc4":"k"},
                {"id": "us_q", "country": "US", "enc1": "b", "enc2": "w", "enc3": "2","enc4":"k"},
                {"id": "us_r", "country": "US", "enc1": "c", "enc2": "x", "enc3": "3","enc4":"k"},
                {"id": "us_s", "country": "US", "enc1": "a", "enc2": "y", "enc3": "4","enc4":"k"},
                {"id": "us_t", "country": "US", "enc1": "b", "enc2": "z", "enc3": "5","enc4":"k"},
                {"id": "us_u", "country": "US", "enc1": "c", "enc2": "w", "enc3": "1","enc4":"k"},
                {"id": "us_v", "country": "US", "enc1": "a", "enc2": "x", "enc3": "2","enc4":"k"},
                {"id": "us_w", "country": "US", "enc1": "b", "enc2": "y", "enc3": "3","enc4":"k"},
                {"id": "us_x", "country": "US", "enc1": "c", "enc2": "z", "enc3": "4","enc4":"k"},
                {"id": "us_y", "country": "US", "enc1": "a", "enc2": "w", "enc3": "5","enc4":"k"},
                {"id": "us_z", "country": "US", "enc1": "b", "enc2": "x", "enc3": "1","enc4":"k"},

                {"id": "eu_a", "country": "EU", "enc1": "a", "enc2": "w", "enc3": "1","enc4":"k"},
                {"id": "eu_b", "country": "EU", "enc1": "b", "enc2": "x", "enc3": "2","enc4":"k"},
                {"id": "eu_c", "country": "EU", "enc1": "c", "enc2": "y", "enc3": "3","enc4":"k"},
                {"id": "eu_d", "country": "EU", "enc1": "a", "enc2": "z", "enc3": "4","enc4":"k"},
                {"id": "eu_e", "country": "EU", "enc1": "b", "enc2": "w", "enc3": "5","enc4":"k"},
                {"id": "eu_f", "country": "EU", "enc1": "c", "enc2": "x", "enc3": "1","enc4":"k"},
                {"id": "eu_g", "country": "EU", "enc1": "a", "enc2": "y", "enc3": "2","enc4":"k"},
                {"id": "eu_h", "country": "EU", "enc1": "b", "enc2": "z", "enc3": "3","enc4":"k"},
                {"id": "eu_i", "country": "EU", "enc1": "c", "enc2": "w", "enc3": "4","enc4":"k"},
                {"id": "eu_j", "country": "EU", "enc1": "a", "enc2": "x", "enc3": "5","enc4":"k"},
                {"id": "eu_k", "country": "EU", "enc1": "b", "enc2": "y", "enc3": "1","enc4":"k"},
                {"id": "eu_l", "country": "EU", "enc1": "c", "enc2": "z", "enc3": "2","enc4":"k"},
                {"id": "eu_m", "country": "EU", "enc1": "a", "enc2": "w", "enc3": "3","enc4":"k"},
                {"id": "eu_n", "country": "EU", "enc1": "b", "enc2": "x", "enc3": "4","enc4":"k"},
                {"id": "eu_o", "country": "EU", "enc1": "c", "enc2": "y", "enc3": "5","enc4":"k"},
                {"id": "eu_p", "country": "EU", "enc1": "a", "enc2": "z", "enc3": "1","enc4":"k"},
                {"id": "eu_q", "country": "EU", "enc1": "b", "enc2": "w", "enc3": "2","enc4":"k"},
                {"id": "eu_r", "country": "EU", "enc1": "c", "enc2": "x", "enc3": "3","enc4":"k"},
                {"id": "eu_s", "country": "EU", "enc1": "a", "enc2": "y", "enc3": "4","enc4":"k"},
                {"id": "eu_t", "country": "EU", "enc1": "b", "enc2": "z", "enc3": "5","enc4":"k"},
                {"id": "eu_u", "country": "EU", "enc1": "c", "enc2": "w", "enc3": "1","enc4":"k"},
                {"id": "eu_v", "country": "EU", "enc1": "a", "enc2": "x", "enc3": "2","enc4":"k"},
                {"id": "eu_w", "country": "EU", "enc1": "b", "enc2": "y", "enc3": "3","enc4":"k"},
                {"id": "eu_x", "country": "EU", "enc1": "c", "enc2": "z", "enc3": "4","enc4":"k"},
                {"id": "eu_y", "country": "EU", "enc1": "a", "enc2": "w", "enc3": "5","enc4":"k"},
                {"id": "eu_z", "country": "EU", "enc1": "b", "enc2": "x", "enc3": "1","enc4":"k"}]

        requests.post("http://127.0.0.1:5000/index", json=index).json()

    def test_encode_vec(self):
        data = {"id": "Tom","country": "US", "enc1": "a", "enc2": "w", "enc3": "1"}

        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()['vec']
        vec = [round(v, 5) for v in vec]
        self.assertEqual(vec, [round(3/np.sqrt(3),5), 0, 0, round(4/np.sqrt(4),5), 0, 0, 0, round(5/np.sqrt(5),5), 0, 0, 0, 0])

    def  test_query_labels(self):
        query = {
                "data":{"id": "Tom","country": "US", "enc1": "a", "enc2": "w", "enc3": "1","enc4":"i"},
                "k":3,
                "explain":True
                }
        ret = requests.post("http://127.0.0.1:5000/query", json=query).json()
        self.assertEqual(ret['ids'], ['us_a', 'us_u', 'us_p'])

    def  test_explainability_labels(self):
        query = {
            "data": {"id": "Tom", "country": "US", "enc1": "a", "enc2": "w", "enc3": "1", "enc4": "i"},
            "k": 3,
            "explain": True
        }
        ret = requests.post("http://127.0.0.1:5000/query", json=query).json()
        weights_0  = [exp['enc4'] for exp in ret['explanation']]
        self.assertEqual(weights_0, [0.0,0.0,0.0])
        full_identity = ret['explanation'][0]
        weights_1 = [full_identity[key] for key in full_identity.keys()]
        self.assertEqual(weights_1, [3.0,4.0,5.0,0.0])


if __name__ == '__main__':
    unittest.main()
