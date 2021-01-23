import numpy as np
import pandas as pd
import pytest
from .bmd_is_independent import bmd_is_independent, posterior

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
    assert bmd_is_independent(**params) == True

    params["data"] = df_2_multinomial_indep_RVs(size=1000)
    assert bmd_is_independent(**params) == True

def test_3_indep_small_100():
    x = np.random.binomial(n=1, p=0.5, size=100)
    y = np.random.binomial(n=1, p=0.5, size=100)
    z = np.random.binomial(n=1, p=0.5, size=100)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    is_independent = bmd_is_independent(
        data=df,
        vars_1=['x'],
        vars_2=['y'],
        conditioning_set=['z']
    )

    assert is_independent == True

def test_long_chains_collider_bias_without_MI(
    df_long_chains_and_collider_without_MI
):
    df = df_long_chains_and_collider_without_MI(size=10000)

    indep = bmd_is_independent(
        data=df,
        vars_1=['d'],
        vars_2=['b']
    )

    assert indep == True

    indep = bmd_is_independent(
        data=df,
        vars_1=['b'],
        vars_2=['d']
    )

    assert indep == True

def test_X_and_Y_are_deterministic(
    df_X_and_Y_are_deterministic
):
    params = {
       "data": df_X_and_Y_are_deterministic(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert bmd_is_independent(**params) == False

def test_X_causes_Y(
    df_X_causes_Y
):
    params = {
       "data": df_X_causes_Y(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert bmd_is_independent(**params) == False

def test_X_and_Y_cause_Z(
    df_X_and_Y_cause_Z
):
    params_1 = {
       "data": df_X_and_Y_cause_Z(size=10000),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert bmd_is_independent(**params_1) == True

    params_2 = {
       "data": df_X_and_Y_cause_Z(size=10000),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert bmd_is_independent(**params_2) == False

def test_Z_causes_X_and_Y(
    df_Z_causes_X_and_Y
):
    params_1 = {
       "data": df_Z_causes_X_and_Y(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert bmd_is_independent(**params_1) == True

    params_2 = {
       "data": df_Z_causes_X_and_Y(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert bmd_is_independent(**params_2) == False

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

    # We still get dependence because when doing this test, e.g. computing
    # P(X|Z, MI_Y=0, Y*) and comparing that to P(X|Z). By testing for the
    # observed Y only, we are implicitly conditioning on MI_Y, which is a
    # collider between # X and Z, so we get a biased estimate and still see an
    # association between X and Y, even after having adjusted for Z.
    assert bmd_is_independent(**params) == False

def test_spurious_edge(
    df_long_chains_and_collider_with_MI
):
    size = 100000

    df = df_long_chains_and_collider_with_MI(size=size)

    params = {
       "data": df,
       "vars_1": ['b'],
       "vars_2": ['d'],
       "conditioning_set": []
    }

    # we expect b-d to be a spurious edge because we're implicitly conditioning
    # on the collider MI_b

    assert bmd_is_independent(**params) == False

def test_spurious_edge_2(
    df_chain_and_collider_with_MI
):
    size = 100000

    df = df_chain_and_collider_with_MI(size=size)

    params = {
       "data": df,
       "vars_1": ['a'],
       "vars_2": ['c'],
       "conditioning_set": []
    }

    # we expect a-c to be a  spurious edge because we're implicitly conditioning
    # on the collider MI_b

    assert bmd_is_independent(**params) == False

