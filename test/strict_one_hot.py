import unittest
import requests

class StrictOneHot(unittest.TestCase):

    def setUp(self):
        schema = {
        "encoders":[{"field": "state",  "values": ["CA", "NY"], "type":"soh", "weight":1}],
        "filters": [{"field": "country", "values": ["US", "EU"]}],
        "metric": "cosine"
        }
        requests.post("http://127.0.0.1:5000/init_schema", json=schema)
    def test_order(self):
        data = {"id":1, "country":"US", "state": "NY"}
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec, [0,1])
        data = {"id":1, "country":"US", "state": "CA"}
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec, [1,0])
    def test_zero_vec(self):
        data = {"id":1, "country":"US", "state": "WA"}
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec, [0,0])

if __name__ == '__main__':
    unittest.main()
