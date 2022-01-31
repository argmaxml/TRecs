import json, collections
import requests
import pandas as pd
from datetime import datetime
import qwak
from qwak.feature_store.entities import ValueType
from qwak.model.base import QwakModelInterface
from qwak.feature_store.offline import OfflineFeatureStore
from qwak.feature_store.online import OnlineFeatureStore
from qwak.model.schema import ModelSchema, BatchFeature, Entity, Prediction


class CompundVectorSearch(QwakModelInterface):
    """ The Model class inherit QwakModelInterface base class
    """

    def __init__(self, filters, encoders, metric="l2", train_size=1e6, feature_set="csv_bq"):
        self.filters = filters
        self.encoders = encoders
        self.metric = metric
        self.train_size=train_size
        self.feature_set=feature_set

    def build(self):
        """ Responsible for loading the model. This method is invoked during build time (qwak build command)

           Example:
           >>> def build(self):
           >>>     ...
           >>>     train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
           >>>     validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
           >>>     self.catboost.fit(train_pool, eval_set=validate_pool)
           """
        offline_feature_store = OfflineFeatureStore()

        train_df = offline_feature_store.get_sample_data(self.feature_set,number_of_rows=self.train_size)
        tabsim_schema = {
            "filters":[],
            "encoders":[],
            "metric": self.metric,
        }
        #TODO: tabim init_schema
        #TODO: tabsim index

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
                Prediction(name="distance", type=float),
            ]
        )
        return model_schema

    @qwak.analytics()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Invoked on every API inference request.
        Args:
            pd (DataFrame): the inference vector, as a pandas dataframe

        Returns: model output (inference results), as a pandas dataframe
        """
        online_feature_store = OnlineFeatureStore()
        extracted_df = online_feature_store.get_feature_values(self.schema(), df)
        #TODO: tabsim query
        res = tabsim()
        return pd.DataFrame({"id": res["ids"], "distance": res["distances"]})
