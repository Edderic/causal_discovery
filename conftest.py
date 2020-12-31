import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def multinomial_RV():
    def _setup(size=10000):
        multinomial = np.random.multinomial(
                n=1,
                pvals=[0.25,
                    0.25,
                    0.25,
                    0.25],
                size=size)

        return multinomial[:, 0] \
                + multinomial[:, 1] * 2 \
                + multinomial[:, 2] * 3 \
                + multinomial[:, 3] * 4

    yield _setup

@pytest.fixture
def df_2_multinomial_indep_RVs(multinomial_RV):
    def _setup(size=10000):
        x = multinomial_RV(size=size)
        y = multinomial_RV(size=size)

        return pd.DataFrame({'x': x, 'y': y})

    yield _setup

@pytest.fixture
def df_X_and_Y_are_deterministic(multinomial_RV):
    def _setup(size=10000):
        x = multinomial_RV(size=size)
        return pd.DataFrame({'x': x, 'y': x})
    yield _setup

@pytest.fixture
def df_X_Y_Z_are_deterministic(multinomial_RV):
    def _setup(size=10000):
        x = multinomial_RV(size=size)
        return pd.DataFrame({'x': x, 'y': x, 'z': x})
    yield _setup

@pytest.fixture
def df_X_causes_Y(multinomial_RV):
    def _setup(size=10000):
        x = multinomial_RV(size=size)
        y = (x == 2)

        return pd.DataFrame({'x': x, 'y': y})
    yield _setup

@pytest.fixture
def df_Z_causes_X_and_Y(multinomial_RV):
    def _setup(size=10000):
        z = multinomial_RV(size=size)

        y = (z == 2)
        x = (z == 1)

        return pd.DataFrame({'x': x, 'y': y, 'z': z})
    yield _setup

@pytest.fixture
def df_X_and_Y_cause_Z(multinomial_RV):
    def _setup(size=10000, proba_noise=0.1):
        x = multinomial_RV(size=size)
        y = multinomial_RV(size=size)
        noise = np.random.binomial(n=1, p=1-proba_noise, size=size)

        z = (x == 1) & (y == 2) & noise

        return pd.DataFrame({'x': x, 'y': y, 'z': z})
    yield _setup

