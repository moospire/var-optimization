import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--remote_sphere", action="store_true", default=False, help="run tests that require remote server process"
    )
    parser.addoption(
        "--remote_shadow", action="store_true", default=False, help="run tests that require remote server process"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--remote_sphere"):
        # --runslow given in cli: do not skip slow tests
        skip_remote = pytest.mark.skip(reason="need --remote_sphere option to run")
        for item in items:
            if "remote_sphere" in item.keywords:
                item.add_marker(skip_remote)

    if not config.getoption("--remote_shadow"):
        # --runslow given in cli: do not skip slow tests
        skip_remote = pytest.mark.skip(reason="need --remote_shadow option to run")
        for item in items:
            if "remote_shadow" in item.keywords:
                item.add_marker(skip_remote)
