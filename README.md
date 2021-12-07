# tabsim - Tabular Similarity Search Server
## API

   1. `/index` - gets a list of dicts
   1. `/query` - gets a single item and returns nearest neighbors
   1. `/save`  - saves model to disk
   1. `/load`  - loads model from disk

# Example data
## init_schema
```
{
    "filters": [
        {"field": "country", "values": ["US", "EU"]}
    ],
    "encoders": [
        {"field": "age", "values":["1","2"], "type": "onehot", "weight":1},
        {"field": "sex", "values":["m","f"], "type": "onehot", "weight":1}
    ],
    "metric": "cosine"
}
```

## index

```
[
  {
    "id": "1",
    "age": "1",
    "sex": "f",
    "country":"US"
  },
  {
    "id": "2",
    "age": "2",
    "sex": "f",
    "country":"US"
  },
  {
    "id": "3",
    "age": "1",
    "sex": "m",
    "country":"US"
  },
  {
    "id": "1",
    "age": "1",
    "sex": "f",
    "country":"EU"
  }
]
```
## Query
```
{
  "k": 2,
  "data": {
    "id": "2",
    "age": "2",
    "sex": "f",
    "country":"US"
  }
}
```
