"""Run CTest against its build's bindings and explicitly selected plugins."""
import argparse
import ctypes
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--python-dir', action='append', default=[])
    parser.add_argument('--script')
    args = parser.parse_args()
    sys.path[:0] = args.python_dir
    # Import OpenMM first so its Python package selects the matching core ABI.
    import openmm
    ctypes.CDLL(args.core, mode=ctypes.RTLD_GLOBAL)
    for plugin in args.plugin:
        openmm.Platform.loadPluginLibrary(plugin)
    if args.script:
        sys.argv = [args.script]
        runpy.run_path(args.script, run_name='__main__')
        return 0
    import pytest
    return pytest.main([str(Path(__file__).parent), '-q', '--tb=short'])


if __name__ == '__main__':
    raise SystemExit(main())
