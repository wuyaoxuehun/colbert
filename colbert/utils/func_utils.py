def cache_decorator(cache_fun):
    cache = {}

    def decorator(funct):
        def fun(*args, **kwargs):
            key = cache_fun(*args, **kwargs)
            if key not in cache:
                # print('not hit' + key)
                cache[key] = funct(*args, **kwargs)
                return cache[key]
            else:
                return cache[key]
            # else:
            #     print(key, len(cache.keys()))
            #     input('hit!')

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
