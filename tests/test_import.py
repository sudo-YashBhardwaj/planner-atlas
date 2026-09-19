from importlib import import_module


def test_package_import() -> None:
    import_module("planner_atlas")
