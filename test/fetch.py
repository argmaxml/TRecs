import unittest
import numpy as np
import requests

class Fetch(unittest.TestCase):

    def setUp(self):
        schema = {
        "encoders":[{"field": "state",  "values": ["CA", "NY"], "type":"soh", "weight":1}],
        "filters": [{"field": "country", "values": ["US", "EU"]}],
        "metric": "cosine"
        }
        requests.post("http://127.0.0.1:5000/init_schema", json=schema)
        data = [{"id":"caca", "country":"US", "state": "CA"},{"id":"nyny", "country":"US", "state": "NY"}]
        requests.post("http://127.0.0.1:5000/index", json=data)

    def test_not_found(self):
        vec = requests.post("http://127.0.0.1:5000/fetch", json=["oops"]).json().get("US")
        self.assertEqual(vec, None)

    def test_fetch(self):
        vec = requests.post("http://127.0.0.1:5000/fetch", json=["caca"]).json().get("US")
        self.assertTrue(np.allclose(vec, [1,0]))

    def test_fetch_2(self):
        vecs = requests.post("http://127.0.0.1:5000/fetch", json=["caca", "nyny"]).json().get("US")
        self.assertEqual(len(vecs),2)

if __name__ == '__main__':
    unittest.main()
