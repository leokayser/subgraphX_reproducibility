import pickle
from typing import Dict


def save_data(path: str, data: Dict):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data(path: str):
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict
