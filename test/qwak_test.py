from collections import Counter
import requests


def predict_test(accessToken):
    url = "https://models.lightricks.qwak.ai/v1/split_test/predict"
    predict_res = requests.post(url, headers={"Authorization": "Bearer "+accessToken}, json={"columns": ["boom"], "index": [0], "data": [["boom"]]}).json()
    try:
        return int(predict_res[0]['res'])
    except:
        return -1

def get_access_token(api_key):
    accessToken = requests.post("https://grpc.qwak.ai/api/v1/authentication/qwak-api-key", json={"qwakApiKey": api_key}).json()["accessToken"]
    return accessToken


def get_feature(accessToken, feature_name = "csv_bq.vec",entity_name = "user_id",entity_value = "1"):
    url = "https://api.lightricks.qwak.ai/api/v1/features"
    body = {
        "features":[{"batchFeature":{"name": feature_name}}],
        "entity": {"name": entity_name, "value": entity_value}
    }
    res = requests.post(url, headers={"Authorization": "Bearer "+accessToken}, json=body).json()
    # Assuming one returned feature
    res = list(res["featureValues"][0]["featureValue"].values())[0]
    # TODO: Do qwak support vector features ?
    return res



if __name__ == "__main__":
    import os
    accessToken = get_access_token(os.environ["QWAK_API"])
    print(get_feature(accessToken, "csv_bq.vec","user_id","2"))
