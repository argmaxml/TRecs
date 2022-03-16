# TRecs - Tabular Similarity Search Server

## Setting up a server
```
pip install TRecSys
python -m TRecSys
```

## API

  1. `/init_schema` - Specify filters, encoders and user encoders for the similarity engines.
   1. `/index` - Index a list of items, each item should be a `dict` mapping an item feature to its value.
   1. `/query` - Gets a single item and returns its k nearest neighbors.
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
        {"field": "price", "values":["low", "mid", "high"], "type": "onehot", "weight":1},
        {"field": "category", "values":["dairy","meat"], "type": "onehot", "weight":2}
    ],
    "metric": "l2"
}
```

## index

```
[
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
]
```
## Item Query
```
{
  "k": 2,
  "data": {
    "price": "low",
    "category": "meat",
    "country":"US"
  }
}
```

## User Query
```
{
  "k": 2,
  "item_history":["1","3","3"],
  "data": {
    "country":"US"
  }
}
```
