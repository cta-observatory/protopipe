import pytest


@pytest.fixture(scope="session")
def pipeline_testdir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_pipeline")
