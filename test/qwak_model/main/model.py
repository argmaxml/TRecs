import json, collections, itertools, subprocess, sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
import qwak
from qwak.feature_store.entities import ValueType
from qwak.model.base import QwakModelInterface
from qwak.feature_store.offline import OfflineFeatureStore
from qwak.feature_store.online import OnlineFeatureStore
from qwak.model.schema import ModelSchema, BatchFeature, Entity, Prediction

__dir__ = Path(__file__).absolute().parent

def bgprocess(p:Path, *args):
    python = sys.executable
    p = p.absolute()
    return subprocess.Popen([python, p.name]+list(args), cwd = str(p.parent))

def config_file(name):
    with (__dir__.parent/"config"/(name+".json")).open('r') as f:
        return json.load(f)

class CompundVectorSearch(QwakModelInterface):
    """ The Model class inherit QwakModelInterface base class
    """

    def __init__(self, train_size=1e6, feature_set="csv_bq"):
        self.train_size=train_size
        self.feature_set=feature_set

    def __del__(self):
        try:
            self.tabsim.kill()
        except:
            pass

    def build(self):
        """ Responsible for loading the model. This method is invoked during build time (qwak build command)

           Example:
           >>> def build(self):
           >>>     ...
           >>>     train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
           >>>     validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
           >>>     self.catboost.fit(train_pool, eval_set=validate_pool)
           """
        #TODO: tabsim path
        tabsim_path = __dir__.parent.parent / "src" / "endpoint.py"
        self.tabsim = bgprocess(tabsim_path)
        offline_feature_store = OfflineFeatureStore()

        train_df = offline_feature_store.get_sample_data(self.feature_set,number_of_rows=self.train_size)
        self.k = 2
        self.tabsim_host = "http://127.0.0.1:5000"
        schema = config_file("schema")
        #TODO: tabsim init_schema
        requests.post(self.tabsim_host + "/init_schema", json=schema)
        #TODO: tabsim index
        records = train_df.to_dict(orient="records")
        requests.post(self.tabsim_host + "/index", json=records)

    def schema(self):
        """ Specification of the model inputs and outputs. Optional method

        Example:
        >>> from qwak.model.schema import ModelSchema, Prediction, ExplicitFeature
        >>>
        >>> def schema(self) -> ModelSchema:
        >>>     model_schema = ModelSchema(
        >>>     features=[
        >>>         ExplicitFeature(name="State", type=str),
        >>>     ],
        >>>     predictions=[
        >>>         Prediction(name="score", type=float)
        >>>     ])
        >>>     return model_schema

       Returns: a model schema specification
       """
        user = Entity(name='user_id', type=ValueType.STRING.value)

        model_schema = ModelSchema(
            entities=[
                user
            ],
            features = [
                BatchFeature(name="csv_bq.vec", entity=user),
            ],
            predictions=[
                Prediction(name="id", type=str),
                Prediction(name="recommendation", type=str),
                Prediction(name="distance", type=float),
            ]
        )
        return model_schema

    @qwak.analytics()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        online_feature_store = OnlineFeatureStore()
        extracted_df = online_feature_store.get_feature_values(self.schema(), df)
        records = extracted_df.to_dict(orient="records")
        #TODO: not query once per row
        similarity_results = []
        for record in records:
            treq = requests.post(self.tabsim_host + "/query", json={"data": record, "k": self.k})
            tres = similarity_results.append((treq).json())
            similarity_results.extend([(i,r,d) for i,r,d in zip(itertools.repeat(record["id"]), tres["ids"], tres["distances"])])
        return pd.DataFrame(similarity_results, columns=["id", "recommendation", "distance"])
