from os.path import exists
from time import perf_counter
from pickle import dump, load
from pandas import Series, read_csv


def memoization(key_parameter={}):
    memory = {}
    def memoize(function):
        def use_memory(*args, **kwargs):
            kwargs.update(zip(function.__code__.co_varnames, args))
            key = []
            for p in key_parameter:
                parameter = args[p] if isinstance(p, int) else kwargs[p]
                if isinstance(parameter, tuple):
                    key.append(tuple(sorted(parameter)))
                elif isinstance(parameter, list):
                    key.append(tuple(sorted(parameter)))
                elif isinstance(parameter, set):
                    key.append(tuple(sorted(parameter)))
                elif isinstance(parameter, dict):
                    key.append(tuple(sorted(parameter.items())))
                elif isinstance(parameter, Series):
                    key.append(tuple(sorted(parameter.to_dict().items())))
                else:
                    key.append(parameter)
            hashable = tuple(key)
            if hashable not in memory:
                memory[hashable] = function(**kwargs)
            return memory[hashable]

        return use_memory
    return memoize


def timer(function):
	def time_function(*args, **kwargs):
		start = perf_counter()
		result = function(*args, **kwargs)
		end = perf_counter()
		print(f'{function.__name__}() executed in {(end-start):.8f}s\n')
		return result
	return time_function


def cache_result(function):
    def get(*args, **kwargs):
        if 'cache' in kwargs:
            file, ext = kwargs['cache']
            del kwargs['cache']

            if not exists(file) or ('force' in kwargs and kwargs['force']):
                del kwargs['force']
                save_file(file, function(*args, **kwargs), ext)
            return load_file(file, ext)
        else:
            return function(*args, **kwargs)
    return get


def save_file(file, data, ext='pickle'):
    if ext == 'pickle':
        with open(file, 'wb') as f:
            dump(data, f)
    elif ext == 'csv':
        data.to_csv(file, index = False)


def load_file(file, ext='pickle'):
    if ext == 'pickle':
        with open(file, 'rb') as f:
            return load(f)
    elif ext == 'csv':
        return read_csv(file)