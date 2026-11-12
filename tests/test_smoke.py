import importlib


def test_src_package_importable():
    assert importlib.import_module("src") is not None
