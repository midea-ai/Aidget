# Copyright (c) Midea Group
# Licensed under the Apache License 2.0

from functools import reduce

def get_module_by_name(model, access_string):
    module_name = access_string.split(sep='.')
    return reduce(getattr, module_name, model)
