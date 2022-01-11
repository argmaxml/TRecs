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
        self.data = [{"id":"caca", "country":"US", "state": "CA"},{"id":"nyny", "country":"US", "state": "NY"}]
        requests.post("http://127.0.0.1:5000/index", json=self.data)

    def test_not_found(self):
        vec = requests.post("http://127.0.0.1:5000/fetch", json=["oops"]).json().get("US")
        self.assertEqual(vec, None)

    def test_fetch(self):
        indexed = requests.post("http://127.0.0.1:5000/encode", json=self.data[0]).json()["vec"]
        fetched = requests.post("http://127.0.0.1:5000/fetch", json=["caca"]).json().get("US")
        print(indexed,fetched)
        self.assertTrue(np.allclose(fetched, indexed))

    def test_fetch_2(self):
        fetched = requests.post("http://127.0.0.1:5000/fetch", json=["caca", "nyny"]).json().get("US")
        indexed0 = requests.post("http://127.0.0.1:5000/encode", json=self.data[0]).json()["vec"]
        indexed1 = requests.post("http://127.0.0.1:5000/encode", json=self.data[1]).json()["vec"]
        # TODO: fix, order is reveresed
        indexed = np.concatenate([indexed1, indexed0])
        self.assertEqual(len(fetched),2)
        fetched = np.array(fetched).reshape(-1)
        self.assertTrue(np.allclose(fetched, indexed))

if __name__ == '__main__':
    unittest.main()
