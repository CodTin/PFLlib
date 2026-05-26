import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests (slower, may need data)")
