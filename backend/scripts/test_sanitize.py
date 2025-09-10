import numpy as np
from importlib import import_module
m = import_module('src.api.routes.analyze')
fn = getattr(m, '_sanitize_for_json')

class X:
    def __init__(self):
        self.a = np.int64(10)

vals = [np.bool_(True), np.int64(5), np.float32(3.14), np.array([1,2,3]), np.array([[1,2],[3,4]]), {'a': np.int64(2)}, X()]
for v in vals:
    s = fn(v)
    print(type(v), '->', type(s), s)
