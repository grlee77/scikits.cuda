import functools
def defineIf(condition):
    if condition:
        def decorator(func):
            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                return func
            return func_wrapper
        return decorator
    else:
        return lambda: None