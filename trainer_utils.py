import yaml
import re
import sys
import os.path
from importlib import import_module
import sys


def get_class(class_name, file_path: str = None, module_path: str = None):
    if file_path:
        try:
            module = import_module(os.path.basename(file_path.replace('.py', '')))
        except ImportError:
            sys.path.append(os.path.dirname(file_path))
            module = import_module(os.path.basename(file_path.replace('.py', '')))
    elif module_path:
        try:
            module = import_module(module_path)
        except ImportError:
            sys.path.append(os.path.dirname(file_path))
            module = import_module(os.path.basename(file_path.replace('.py', '')))
    else:
        raise Exception('module path or file path are required')
    return getattr(module, class_name)


def read_yaml(file_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(file_path) as f:
        data = yaml.load(f, Loader=loader)
    return data
