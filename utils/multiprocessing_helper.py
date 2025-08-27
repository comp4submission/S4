import math
from typing import Any, Iterator


def generator(full_list: list[Any], sublist_length: int) -> Iterator[list[Any]]:
    group_count = math.ceil(len(full_list) / sublist_length)

    for idx in range(group_count):
        start = sublist_length * idx
        end = min(start + sublist_length, len(full_list))

        yield full_list[start:end]
