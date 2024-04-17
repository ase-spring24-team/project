'''
Container to store 'the' values, so that all modules can access it
'''

from typing import Any


class SLOTS(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

_the = SLOTS({"file":"../data/auto93.csv", "__help": "", "m":2}) #Will be overridden when the is loaded by gate.py


class THE:
    _slot = None
    def __init__(self) -> None:
        pass
    def _set(self, slot):
        self._slot = slot
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "_slot":
            super().__setattr__(__name, __value)
        else:
            self._slot.__setattr__(__name, __value)
    
    def __getattr__(self, __name: str) -> Any:
        if __name != "_slot":
            return self._slot.__getattr__(__name)
        else:
            return None

the = THE()