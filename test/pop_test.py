import sys, unittest
from pathlib import Path
import numpy as np
# print(str(Path(__file__).absolute().parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent))
from TRecSys import AvgUserStrategy

class PopularityTest(unittest.TestCase):
    def setUp(self):
        self.strategy = AvgUserStrategy()


        self.strategy.init_schema(
                filters =  [
                    {"field": "country", "values": ["US", "EU"]}
                ],
                encoders =  [
                    {"field": "popularity", "values": ["1"], "type": "numeric", "weight": 1, "default": 1},
                    {"field": "price", "values": ["low", "mid", "high"], "type": "onehot", "weight": 8},
                    {"field": "category", "values": ["dairy", "meat"], "type": "onehot", "weight": 16}
                ],
                user_encoders = [
                    {"field": "popularity", "values": ["1"], "type": "numeric", "weight": 1, "default": 1},
                ],
                metric = "l2"
            )
        self.strategy.index([
            {
                "id": "1",
                "popularity": 0.1,
                "price": "low",
                "category": "meat",
                "country": "US"
            },
            {
                "id": "2",
                "popularity": 0.1,
                "price": "mid",
                "category": "meat",
                "country": "US"
            },
            {
                "id": "3",
                "popularity": 0.2,
                "price": "low",
                "category": "dairy",
                "country": "US"
            },
            {
                "id": "4",
                "popularity": 0.1,
                "price": "high",
                "category": "meat",
                "country": "EU"
            }
        ])
        self.strategy.save_model("t1")

    def test_encode(self):
        expected = np.array([ 1., 0.,  8.,  0.,  0.,  0.,  0., 16.])
        actual = self.strategy.encode({
                            "price": "low",
                            "category": "meat",
                            "country":"US"
                             })
        np.testing.assert_array_equal(expected, actual)

    def test_query(self):
        query_example = {
                    "k": 2,
                  "data": {
                    "price": "low",
                    "category": "meat",
                    "country":"US"}
                        }
        expected = [['1', '2'], [0.809999942779541, 128.80999755859375], []]
        actual = self.strategy.query(**query_example)
        np.testing.assert_array_equal(expected, actual)

    def test_user_query(self):
        user_query_example  = {
                              "k": 2,
                              "item_history":["1","1"],
                              "user_data": {
                                "country":"US"
                                            }
                             }
        expected = [['1', '2'], [0.0, 128.0]]
        actual = self.strategy.user_query(**user_query_example)
        np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
