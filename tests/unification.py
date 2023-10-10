import sys
sys.path.append("fast-openISP")

import modules as gt_modules
from utils.yacs import Config as gt_Config

from jaxisp.nodes.awb import AWB

def dict_to_config(dot_dict):
    result = {}
    for key, value in dot_dict.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return gt_Config(result)

def wrap_gt_module(module):
    class WrappedModule:
        def __init__(self, cfg):
            config = dict_to_config(cfg)
            self.module = module(config)
            
        def __call__(self, **data):
            self.module.execute(data)
            return data

    return WrappedModule


gt_awb = wrap_gt_module(gt_modules.AWB)({
    "hardware.bayer_pattern": "rggb", "saturation_values.hdr": 1000, "awb.r_gain": 2415, "awb.gr_gain": 1024, "awb.gb_gain": 1024, "awb.b_gain": 1168
})
aaf = AWB({"bayer_pattern": "rggb", "saturation_values_hdr": 1000, "r_gain": 2415, "gr_gain": 1024, "gb_gain": 1024, "b_gain": 1168})

import numpy as np
from time import perf_counter

a = np.random.randint(0, 2**12 - 1, (4000, 4000), dtype=np.int16)

t_0 = perf_counter()
gt_res = gt_awb(bayer=a)["bayer"]
t_1 = perf_counter()

res = aaf(a)

t_2 = perf_counter()
res = aaf(a)
t_3 = perf_counter()

print(gt_res)
print(res)


print("gt: ", t_1 - t_0)
print("ours: ", t_3 - t_2)
print(np.all(gt_res == res))

