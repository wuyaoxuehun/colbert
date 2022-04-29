from functools import wraps

from tqdm import tqdm


def name_print_decorater(fun):
    @wraps(fun)
    def inner(*args, **kwargs):
        print(f"starting {fun.__name__}")
        res = fun(*args, **kwargs)
        print(f"end {fun.__name__}")
        return res

    return inner


@name_print_decorater
def index_by_key(key, d, enable_tqdm=False):
    if type(key) == str:
        key_ = lambda _: _[key]
    else:
        key_ = key
    res = {key_(_): _ for _ in (tqdm(d) if enable_tqdm else d)}
    if len(res) != len(d):
        print(f"key not distinct - {len(res) - len(d)}")
    return res

def read_lines_file(file):
    with open(file, 'r', encoding='utf8') as f:
        return [_.strip() for _ in f.readlines()]

def testdecorator():
    d = [{"a": 1}, {"a": 2}]
    print(index_by_key('a', d))


if __name__ == '__main__':
    testdecorator()
