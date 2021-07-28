import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--remote", action="store_true", default=False, help="run tests that require remote server process"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--remote"):
        # --runslow given in cli: do not skip slow tests
        skip_remote = pytest.mark.skip(reason="need --remote option to run")
        for item in items:
            if "remote" in item.keywords:
                item.add_marker(skip_remote)
