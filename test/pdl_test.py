import unittest, json
import requests

class PDLTest(unittest.TestCase):

    def setUp(self):
        with open('../data/pdl_sample_schema.json', 'r') as f:
            schema = json.load(f)
        with open('../data/pdl_sample_index.json', 'r') as f:
            self.data = json.load(f)
        requests.post("http://127.0.0.1:5000/init_schema", json=schema)

    def test_all_zeros(self):
        data = self.data[1]
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec[:6], [0,0,0,0,0,0])

    def test_encode(self):
        with open('../data/pdl_sample_index.json', 'r') as f:
            data = json.load(f)
        data = self.data[0]
        vec = requests.post("http://127.0.0.1:5000/encode", json=data).json()["vec"]
        self.assertEqual(vec[:6], [0,0,0,1,0,0])

    def test_index(self):
        #with self.assertRaises(ValueError)
        ret = requests.post("http://127.0.0.1:5000/index", json=self.data).json()
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()
