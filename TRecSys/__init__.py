from .strategies import BaseStrategy, AvgUserStrategy
from .encoders import PartitionSchema
from .endpoint import run_server

if __name__ == "__main__":
    run_server()