import sys, unittest
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from TRecSys import AvgUserStrategy

class IndexParallel(unittest.TestCase):
    def setUp(self):
        self.strategy = AvgUserStrategy()


        self.strategy.init_schema(
        filters= [
            {"field": "country", "values": ["US", "EU"]}
        ],
        encoders= [
            {"field": "price", "values":["low", "mid", "high"], "type": "onehot", "weight":1},
            {"field": "category", "values":["dairy","meat"], "type": "onehot", "weight":2}
        ],
        metric= "l2"
        )
        self.strategy.index([
        {
            "id": "1",
            "price": "low",
            "category": "meat",
            "country":"US"
        },
        {
            "id": "2",
            "price": "mid",
            "category": "meat",
            "country":"US"
        },
        {
            "id": "3",
            "price": "low",
            "category": "dairy",
            "country":"US"
        },
        {
            "id": "4",
            "price": "high",
            "category": "meat",
            "country":"EU"
        }
        ])
        self.strategy.save_model("t1")

    def test_pandas(self):
        df = pd.DataFrame([
            ['id1', "low", "meat", "US"],
            ['id2', "mid", "meat", "EU"],
            ['id3', "low", "dairy", "US"],
            ['id4', "high", "meat", "EU"],
            ['id5', "low", "meat", "US"],
            ['id6', "mid", "meat", "EU"],
            
        ], columns=["id", "price", "category", "country"])
        self.strategy.index_dataframe(df)
        actual = self.strategy.index_labels
        expected = ['1', '2', '3', '4', 'id2', 'id4', 'id6', 'id1', 'id3', 'id5']
        self.assertEqual(expected, actual)




if __name__ == '__main__':
    unittest.main()