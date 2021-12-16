import unittest
import requests

class StrictOneHot(unittest.TestCase):

    def setUp(self):
        schema = {
        "encoders":[{"field": "state",  "values": ["a", "b", "c", "d", "e"], "type":"np", "weight":1, "url":"/home/ugoren/tabsim/test/test_np_encoder.npz"}],
        "filters": [{"field": "country", "values": ["US", "EU"]}],
        "metric": "cosine"
        }
        requests.post("http://127.0.0.1:5000/init_schema", json=schema)

    def test_read(self):
        data = {"id":1, "country":"US", "state": "b"}
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec, [0,1,0,0,0])

    def test_missing(self):
        data = {"id":1, "country":"US", "state": "i"}
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec, [0,0,0,0,0])

if __name__ == '__main__':
    unittest.main()
