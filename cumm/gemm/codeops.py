from collections import OrderedDict
from typing import (Any, Callable, Dict, Generator, Generic, Hashable,
                    Iterable, List, Tuple, TypeVar, Union)

import pccm

_T = TypeVar("_T")
_T2 = TypeVar("_T2")


def group_by(key_func: Callable[[_T], _T2],
             data_iter: Iterable[_T]) -> Dict[_T2, List[_T]]:
    res = OrderedDict()  # type: OrderedDict[_T2, List[_T]]
    for d in data_iter:
        key = key_func(d)  # type: _T2
        if key not in res:
            res[key] = []
        res[key].append(d)
    return res


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def unpack(arr_name: str,
           indices: Iterable[int],
           op: str = ", ",
           left: str = "[",
           right: str = "]"):
    return op.join(f"{arr_name}{left}{i}{right}" for i in indices)


def unpack_str(arr_name: str,
               indices: Iterable[int],
               op: str = ", ",
               left: str = "_"):
    return unpack(arr_name, indices, op, left, "")


class Condition:
    def __init__(self, flag: bool):
        self.flag = flag

    def __call__(self, true_cond: str, false_cond: str = ""):
        if self.flag:
            return true_cond
        else:
            return false_cond
