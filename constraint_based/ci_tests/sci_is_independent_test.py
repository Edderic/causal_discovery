import pandas as pd
import pytest
from causal_discovery.constraint_based.ci_tests.sci_is_independent import sci_is_independent

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    df = df_long_chains_and_collider_without_MI(size=10000)

    indep = sci_is_independent(
        data=df,
        vars_1=['d'],
        vars_2=['b']
    )

    assert indep == True

    indep = sci_is_independent(
        data=df,
        vars_1=['b'],
        vars_2=['d']
    )

    assert indep == True

def test_uniform_multinomial_with_4_possible_values_size_10000(
    df_2_multinomial_indep_RVs
):
    params = {
       "data": df_2_multinomial_indep_RVs(size=10000),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    params["data"] = df_2_multinomial_indep_RVs(size=10000)
    assert sci_is_independent(**params) == True

    params["data"] = df_2_multinomial_indep_RVs(size=1000)
    assert sci_is_independent(**params) == True

    params["data"] = df_2_multinomial_indep_RVs(size=100)
    assert sci_is_independent(**params) == True

def test_X_Y_Z_are_deterministic(
    df_X_Y_Z_are_deterministic
):
    params = {
       "data": df_X_Y_Z_are_deterministic(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params) == True

def test_X_and_Y_are_deterministic(
    df_X_and_Y_are_deterministic
):
    params = {
       "data": df_X_and_Y_are_deterministic(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params) == False

def test_X_causes_Y(
    df_X_causes_Y
):
    params = {
       "data": df_X_causes_Y(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params) == False

def test_X_Y_Z_indep(
    multinomial_RV
):
    size=1000
    df = pd.DataFrame(
        {
            'x': multinomial_RV(size=size),
            'y': multinomial_RV(size=size),
            'z': multinomial_RV(size=size),
        }
    )

    params = {
       "data": df,
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params) == True

def test_X_and_Y_cause_Z(
    df_X_and_Y_cause_Z
):
    params_1 = {
       "data": df_X_and_Y_cause_Z(size=1000),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params_1) == True

    params_2 = {
       "data": df_X_and_Y_cause_Z(size=1000),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params_2) == False

def test_Z_causes_X_and_Y(
    df_Z_causes_X_and_Y
):
    params_1 = {
       "data": df_Z_causes_X_and_Y(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params_1) == True

    params_2 = {
       "data": df_Z_causes_X_and_Y(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params_2) == False

def test_spurious_edge(
    df_long_chains_and_collider_with_MI
):
    size = 10000

    df = df_long_chains_and_collider_with_MI(size=size)

    params = {
       "data": df,
       "vars_1": ['b'],
       "vars_2": ['d'],
       "conditioning_set": []
    }

    # we expect b-d to be a spurious edge because we're implicitly conditioning
    # on the collider MI_b

    assert sci_is_independent(**params) == False

def test_3_multinom_RVs_MAR(
    df_Z_causes_X_Y_and_X_Z_causes_MI_Y
):
    size = 10000

    df = df_Z_causes_X_Y_and_X_Z_causes_MI_Y(size=size)

    params = {
       "data": df,
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params) == True
