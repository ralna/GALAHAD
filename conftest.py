import pytest
import subprocess
import tempfile
import sys

def pytest_collect_file(file_path, parent):
    """Tells pytest to discover any script starting with test_ and ending in .py"""
    if file_path.name.startswith("test_") and file_path.suffix == ".py":
        return ScriptFile.from_parent(parent, path=file_path)

class ScriptFile(pytest.File):
    def collect(self):
        yield ScriptItem.from_parent(self, name=self.path.name)

class ScriptItem(pytest.Item):
    def runtest(self):
        res = subprocess.run(
            [sys.executable, "-s", str(self.path.resolve())],
            cwd=tempfile.gettempdir()
        )
        if res.returncode != 0:
            raise RuntimeError(f"Script failed with exit code {res.returncode}")
