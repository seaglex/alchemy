import functools
import re
import subprocess
import warnings
from RestrictedPython import compile_restricted
from RestrictedPython import Guards, Eval
from RestrictedPython.PrintCollector import PrintCollector
from setuptools.errors import CompileError
from concurrent.futures import ThreadPoolExecutor


warnings.filterwarnings("ignore", module="RestrictedPython")


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
    """
    容易出现无限循环，直接卡死所有进程，最多调试用，不要实际用
    """
    def __init__(self, allowed_modules=None):
        # 设置允许导入的模块白名单
        self.allowed_modules = {'math', 'sympy', 'scipy', 'numpy', 'pandas'} if allowed_modules is None else allowed_modules

        # 自定义导入函数以限制模块访问
        def restricted_import(name, *args, **kwargs):
            if name not in self.allowed_modules:
                raise ImportError(f"Module '{name}' is not allowed.")
            return __import__(name, *args, **kwargs)

        def my_set_item(var, index, value):
            var[index] = value

        def my_del_item(var, index):
            del var[index]

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
                **Guards.safe_builtins,
                '__import__': restricted_import,
            },
            '__metaclass__': type,
            "__name__": "__main__",
            "_getiter_": Eval.default_guarded_getiter,
            "_iter_unpack_sequence_": Guards.guarded_iter_unpack_sequence,
            "_unpack_sequence_": Guards.guarded_unpack_sequence,
            "_getattr_": Guards.safer_getattr,
            "_delattr_": Guards.guarded_delattr,
            "_setattr_": Guards.guarded_setattr,
            "_getitem_": Eval.default_guarded_getitem,
            "_write_": Guards.full_write_guard,  # for set item and del item
            "_inplacevar_": inplacevar,
            "sum": sum,
            "max": max,
            "min": min,
        }

    def _inner_run(self, code_str) -> str:
        if not code_str:
            raise CompileError("Empty code string.")
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


class SubprocessSandbox:
    def __init__(self, allowed_modules=None):
        # 设置允许导入的模块白名单
        self.allowed_modules = {'math', 'sympy', 'scipy', 'numpy', 'pandas'} if allowed_modules is None else allowed_modules

    def _analyze_imports(self, code_str: str):
        """
        分析代码中是否只导入了允许的模块。
        支持 import 和 from ... import 语法。
        """
        import_pattern = re.compile(r'^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', re.MULTILINE)
        imports = import_pattern.findall(code_str)

        for module in imports:
            base_module = module.split('.')[0]  # 只检查顶层模块
            if base_module not in self.allowed_modules:
                raise ImportError(f"Module '{base_module}' is not allowed.")
        match = re.search(r"_import_", code_str)
        if match:
            raise ImportError("_import_ is not allowed.")

    def _try_compile(self, code_str):
        # 编译代码
        try:
            _ = compile_restricted(code_str, filename='<inline>', mode='exec')
        except Exception as e:
            raise CompileError(f"Failed to compile code: {e}")

    def _inner_run(self, code_str: str, timeout_seconds) -> str:
        """
        在子进程中执行代码，并返回 stdout 的内容。
        """
        # 构造完整的命令：python -c "code"
        cmd = ['python', '-c', code_str]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_seconds
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute code: {e.stderr or str(e)}")

    def run(self, code_str: str, timeout_seconds=5) -> str:
        self._analyze_imports(code_str)
        self._try_compile(code_str)  ## keep similar behavior as LocalSandbox, for RFT
        return self._inner_run(code_str, timeout_seconds)