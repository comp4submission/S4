from typing import Union, Any


def add_item_to_list_of_dict(d: dict[str, list[Any]], k: Union[str, tuple], i: Any):
    if k in d.keys():
        d[k].append(i)
    else:
        d[k] = [i]


def extend_items_to_list_of_dict(d: dict[str, list[Any]], k: Union[str, tuple], items: Union[list, set]):
    if k in d.keys():
        d[k].extend(items)
    else:
        d[k] = list(items)


def add_item_to_set_of_dict(d: dict[str, set[Any]], k: Union[str, tuple], i: Any):
    if k in d.keys():
        d[k].add(i)
    else:
        d[k] = {i}


def extend_items_to_set_of_dict(d: dict[str, set[Any]], k: Union[str, tuple], items: Union[list, set]):
    if k in d.keys():
        d[k].update(items)
    else:
        d[k] = set(items)
