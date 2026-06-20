"""Tests for cta_common.subprocess_env."""

import os

from cta_common.subprocess_env import DEFAULT_STRIP_VARS, make_subprocess_env


def test_strips_host_python_vars():
    base = {"PYTHONHOME": "/x", "PYTHONPATH": "/y", "KEEP": "1", "PATH": "/usr/bin"}
    env = make_subprocess_env("/opt/py/bin/python3", base_env=base)
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert env["KEEP"] == "1"


def test_prepends_interpreter_bin_dir_to_path():
    base = {"PATH": "/usr/bin"}
    env = make_subprocess_env("/opt/py/bin/python3", base_env=base)
    assert env["PATH"].startswith("/opt/py/bin" + os.pathsep)
    assert env["PATH"].endswith("/usr/bin")


def test_extra_overrides_applied_and_stringified():
    env = make_subprocess_env("/opt/py/bin/python3", base_env={}, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=4)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "4"  # coerced to str


def test_custom_strip_vars():
    base = {"FOO": "1", "BAR": "2", "PATH": ""}
    env = make_subprocess_env("/p/bin/python", base_env=base, strip_vars=("FOO",))
    assert "FOO" not in env and env["BAR"] == "2"


def test_default_strip_vars_includes_dyld():
    assert "DYLD_LIBRARY_PATH" in DEFAULT_STRIP_VARS
    assert "PYTHONHOME" in DEFAULT_STRIP_VARS
