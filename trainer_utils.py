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


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
