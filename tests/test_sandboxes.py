import unittest

from setuptools.errors import CompileError

from alchemy.tools.sandboxes import LocalSandbox, SubprocessSandbox
import math


class LocalSandboxTest(unittest.TestCase):
    def test_compile_error(self):
        sandbox = LocalSandbox()
        with self.assertRaises(CompileError):
            sandbox.run("result = ")

    def test_print(self):
        sandbox = LocalSandbox()
        self.assertEqual(sandbox.run("print(1+1)").strip(), "2")

    def test_import(self):
        sandbox = LocalSandbox()
        result = sandbox.run("import math\nprint(math.sqrt(2))").strip()
        self.assertAlmostEqual(float(result), math.sqrt(2))

    def test_timeout(self):
        sandbox = LocalSandbox({'time'})
        with self.assertRaises(TimeoutError):
            sandbox.run('import time\ntime.sleep(2)\nprint(7)', 1)

    def test_runtime_error(self):
        sandbox = LocalSandbox()
        with self.assertRaises(RuntimeError):
            sandbox.run("1/0")

class SubprocessSandboxTest(unittest.TestCase):
    def test_print(self):
        sandbox = SubprocessSandbox()
        self.assertEqual(sandbox.run("print(1+1)").strip(), "2")

    def test_import(self):
        sandbox = SubprocessSandbox({"subprocess"})
        code_str = 'import subprocess\nprint(subprocess.run(["ls", "/"]))'
        print(sandbox.run(code_str).strip())
        sandbox = SubprocessSandbox()
        with self.assertRaises(ImportError):
            sandbox.run(code_str).strip()
