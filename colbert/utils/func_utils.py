def cache_decorator(cache_fun):
    cache = {}

    def decorator(funct):
        def fun(*args, **kwargs):
            key = cache_fun(*args, **kwargs)
            if key in cache:
                print('hit cache')
                pass
            else:
                cache[key] = funct(*args, **kwargs)
            return cache[key]

        return fun

    return decorator


def test_cache_decorator():
    @cache_decorator(cache_fun=lambda x: x['key'])
    def func(inst):
        return inst['val'] ** 2

    t1 = {'key': 1, 'val': 1}
    t2 = {'key': 2, 'val': 2}

    print(func(t1))
    print(func(t2))
    print(func(t1))
    print(func(t2))

if __name__ == '__main__':
    test_cache_decorator()
