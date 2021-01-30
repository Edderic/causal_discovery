import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def multinomial_RV():
    def _setup(
        pvals=[0.25,
            0.25,
            0.25,
            0.25
        ],
        size=10000
    ):
        multinomial = np.random.multinomial(
            n=1,
            pvals=pvals,
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
    def _setup(size=10000, proba_noise=0.2):
        z = multinomial_RV(size=size)

        y = (z == 2) & np.random.binomial(n=1, p=1-proba_noise, size=size)
        x = ((z == 1) | (z == 3)) & np.random.binomial(n=1, p=1-proba_noise, size=size)
        return pd.DataFrame({'x': x, 'y': y, 'z': z})
    yield _setup

@pytest.fixture
def df_X_and_Y_cause_Z(multinomial_RV):
    def _setup(size=10000, proba_noise=0.1):
        x = multinomial_RV(size=size, pvals=[0.2,0.3,0.4,0.1])
        y = multinomial_RV(size=size, pvals=[0.4,0.3,0.2,0.1])
        noise = np.random.binomial(n=1, p=1-proba_noise, size=size)

        z = (x == 1) & (y == 2) & noise

        return pd.DataFrame({'x': x, 'y': y, 'z': z})
    yield _setup

@pytest.fixture
def df_X_and_Y_cause_Z_and_Z_cause_MI_X(df_X_and_Y_cause_Z):
    def _setup(size=10000, proba_noise=0.1):
        df = df_X_and_Y_cause_Z(size=size, proba_noise=proba_noise)

        missingness_of_x = \
            np.random.binomial(n=1, p=proba_noise, size=size) \
            & df['z'].isin([1])

        missingness_indices = np.where(missingness_of_x == 1)

        df.at[missingness_indices[0], 'x'] = np.nan

        return df

    yield _setup

@pytest.fixture
def df_X_Y_indep_Y_causes_MI_X(df_2_multinomial_indep_RVs):
    #  X  Y -*> MI_x
    def _setup(size=10000, proba_noise=0.1):

        df = df_2_multinomial_indep_RVs(size=size)

        # Y causes missingness of X
        missingness_of_x = \
            np.random.binomial(n=1, p=1-proba_noise, size=size) \
            & (df['y'].isin([1,2]))

        missingness_indices = np.where(missingness_of_x == 1)

        df.at[missingness_indices[0], 'x'] = np.nan

        return df

    yield _setup

@pytest.fixture
def df_Z_causes_X_Y_and_X_Z_causes_MI_Y(df_Z_causes_X_and_Y):
    #          Z
    #         /|\
    #        / | \
    #       v  |  v
    #       X  |  Y
    #       \  |
    #        v v
    #        MI_Y
    def _setup(size=10000, proba_noise=0.1):
        df = df_Z_causes_X_and_Y(size=size, proba_noise=proba_noise)

        missingness_of_y = \
            np.random.binomial(n=1, p=1-proba_noise, size=size) \
            & df['x'].isin([1]) \
            & df['z'].isin([3])

        missingness_indices = np.where(missingness_of_y == 1)

        df.at[missingness_indices[0], 'y'] = np.nan

        return df

    yield _setup

@pytest.fixture
def df_long_chains_and_collider_without_MI(multinomial_RV):
    #
    #   A         E
    #    \       /
    #     v     v
    #     B     D
    #     \     /
    #      \   /
    #       v v
    #        C
    #
    def _setup(size=10000, proba_noise=0.5):
        a = multinomial_RV(size=size)
        e = multinomial_RV(size=size)

        b = (pd.Series(a).isin([1,2])) & np.random.binomial(n=1, p=proba_noise - 0.15, size=size)
        d = (pd.Series(e).isin([2,3])) & np.random.binomial(n=1, p=proba_noise + 0.15, size=size)

        # collider
        c = b & d & np.random.binomial(n=1, p=0.8, size=size)

        df = pd.DataFrame(
            {
                'a': a,
                'b': b,
                'c': c,
                'd': d,
                'e': e,
            }
        )

        return df

    yield _setup

@pytest.fixture
def df_long_chains_and_collider_with_MI(df_long_chains_and_collider_without_MI):
    #
    #   A         E
    #    \       /
    #     v     v
    #     B     D
    #     \     /
    #      \   /
    #       v v
    #        C
    #        |
    #        v
    #       MI_b
    #
    def _setup(size=10000, proba_noise=0.5):
        df = df_long_chains_and_collider_without_MI(size=size, proba_noise=proba_noise)
        MI_b = df['c'] * np.random.binomial(n=1, p=proba_noise, size=size)

        missingness_indices = np.where(MI_b == 1)

        df.at[missingness_indices[0], 'b'] = np.nan

        return df

    yield _setup

@pytest.fixture
def df_chain_and_collider_without_MI(multinomial_RV):
    #
    #    A -> B -> C
    #     \       /
    #      \     /
    #       \   /
    #        v v
    #         D
    #
    def _setup(size=10000, proba_noise=0.8):
        a = np.random.binomial(n=1, p=0.8, size=size)
        b_val = np.random.binomial(n=1, p=0.7, size=size)

        b = np.random.binomial(n=1, p=0.5, size=size)
        copy_bval_loc = np.where(b_val == 1)[0]
        b[copy_bval_loc] = a[copy_bval_loc]

        c_val = np.random.binomial(n=1, p=0.8, size=size)
        c = np.random.binomial(n=1, p=0.8, size=size)
        copy_c_val_loc = np.where(c_val == 1)[0]
        c[copy_c_val_loc] = b[copy_c_val_loc]

        d = (c==0) & (a==1)

        df = pd.DataFrame({
            'a': a,
            'b': b,
            'c': c,
            'd': d,
        })

        return df

    yield _setup
@pytest.fixture
def df_chain_and_collider_with_MI(df_chain_and_collider_without_MI):
    #
    #    A -> B -> C
    #     \       /
    #      \     /
    #       \   /
    #        v v
    #         D
    #         |
    #         v
    #        M_a
    #
    def _setup(size=10000, proba_noise=0.8):
        df = df_chain_and_collider_without_MI(
            size=size, proba_noise=proba_noise
        )

        MI_a = df['d'] * np.random.binomial(n=1, p=0.3, size=size)

        missingness_indices = np.where(MI_a == 1)

        df.at[missingness_indices[0], 'a'] = np.nan

        return df

    yield _setup

@pytest.fixture
def df_2_deterministic_and_3rd_var_caused_by_one_of_them():
    #   X -> Y -> Z
    #   X and Y are deterministic
    #
    def _setup(size=10000, proba_noise=0.8):

        x = np.random.binomial(n=1, p=1-proba_noise, size=size)
        y = np.copy(x) # deterk
        z = y * np.random.binomial(n=1, p=1-proba_noise, size=size)

        return pd.DataFrame({'x': x, 'y': y, 'z': z})

    yield _setup
