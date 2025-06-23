import functools
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Eval import default_guarded_getiter
from setuptools.errors import CompileError
from concurrent.futures import ThreadPoolExecutor


def timeout(seconds: float):
    """
    不能真正打断子线程
    :param seconds:
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    raise TimeoutError(f"Function '{func.__name__}' exceeded the time limit of {seconds} seconds.")
        return wrapper
    return decorator


class LocalSandbox(object):
    def __init__(self, allowed_modules=None):
        # 设置允许导入的模块白名单
        self.allowed_modules = {'math'} if allowed_modules is None else allowed_modules

        # 自定义导入函数以限制模块访问
        def restricted_import(name, *args, **kwargs):
            if name not in self.allowed_modules:
                raise ImportError(f"Module '{name}' is not allowed.")
            return __import__(name, *args, **kwargs)

        def inplacevar(op, var, expr):
            if op == "+=":
                return var + expr
            elif op == "-=":
                return var - expr
            elif op == "*=":
                return var * expr
            elif op == "/=":
                return var / expr
            elif op == "%=":
                return var % expr
            elif op == "**=":
                return var ** expr
            elif op == "<<=":
                return var << expr
            elif op == ">>=":
                return var >> expr
            elif op == "|=":
                return var | expr
            elif op == "^=":
                return var ^ expr
            elif op == "&=":
                return var & expr
            elif op == "//=":
                return var // expr
            elif op == "@=":
                return var // expr

        # 构建受限的全局命名空间
        self.restricted_globals = {
            '__builtins__': {
                **safe_builtins,
                '__import__': restricted_import,
            },
            '__metaclass__': type,
            "__name__": "__main__",
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_unpack_sequence,
            "_getattr_": safer_getattr,
            "_inplacevar_": inplacevar,
        }

    def _inner_run(self, code_str) -> str:
        # 编译代码
        try:
            byte_code = compile_restricted(code_str, filename='<inline>', mode='exec')
        except Exception as e:
            raise CompileError(f"Failed to compile code: {e}")

        # 执行代码
        ses_globals = self.restricted_globals
        ses_locals = {
            "_print_": PrintCollector,
        }
        try:
            exec(byte_code, ses_globals, ses_locals)
        except Exception as e:
            raise RuntimeError(f"Failed to execute code: {e}")
        if "_print" not in ses_locals:
            return ""
        return str(ses_locals["_print"]())

    def run(self, code_str, timeout_seconds=5) -> str:
        return timeout(timeout_seconds)(self._inner_run)(code_str)
