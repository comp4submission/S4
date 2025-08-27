import json
import pickle

from typing import Any, NoReturn, Optional


# wrapper function for pickle
def pickle_dump(obj: Any, filepath: str) -> NoReturn:
    with open(filepath, 'wb') as o:
        pickle.dump(obj, o, protocol=2)


def pickle_load(filepath: str) -> Any:
    with open(filepath, 'rb') as i:
        obj = pickle.load(i)
    return obj


# wrapper function for json
def json_dump(obj: dict, filepath: str) -> NoReturn:
    with open(filepath, 'w') as o:
        json.dump(obj, o, indent=4)


def json_load(filepath: str) -> Optional[list or dict]:
    try:
        with open(filepath, 'r') as i:
            obj = json.load(i)
    except:
        obj = None

    return obj
