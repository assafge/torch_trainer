import yaml
import re
import sys
import os
from importlib import import_module


def get_class(class_name, file_path=None, module_path=None):
    if file_path:
        class_dir = os.path.dirname(file_path)
        sys.path.append(class_dir)
        module = import_module(os.path.basename(class_dir))
    elif module_path:
        module = import_module(module_path)
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
    return yaml.load(file_path, Loader=loader)
