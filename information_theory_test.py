import pandas as pd
import numpy as np
import pytest
from pytest import approx
from information_theory import entropy, conditional_entropy, multinomial_normalizing_sum, conditional_mutual_information, sci_is_independent
def test_entropy_uniform_multinomial_with_4_possible_values_size_10000():
    size = 10000
    multinomials = np.random.multinomial(
            n=1,
            pvals=[0.25,
                0.25,
                0.25,
                0.25],
            size=size)

    x = multinomials[:, 0] \
            + multinomials[:, 1] * 2 \
            + multinomials[:, 2] * 3 \
            + multinomials[:, 3] * 4

    df = pd.DataFrame({'x': x})

    assert entropy(data=df, variables=['x']) == approx(2, abs=0.01)

def test_cond_ent_uniform_multinomial_with_4_possible_values(
    df_2_multinomial_indep_RVs
):
    assert conditional_entropy(
               data=df_2_multinomial_indep_RVs(size=10000),
               variables=['x', 'y'],
               conditioning_set=['y']
           ) == approx(2, abs=0.01)

def test_cond_mut_inf_uniform_multinomial_with_4_possible_values_size_10000(
    df_2_multinomial_indep_RVs
):
    assert conditional_mutual_information(
               data=df_2_multinomial_indep_RVs(),
               vars_1=['x'],
               vars_2=['y'],
               conditioning_set=[]
           ) == approx(0, abs=0.01)

def test_sci_indep_uniform_multinomial_with_4_possible_values_size_10000(
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

def test_sci_indep_X_Y_Z_are_deterministic(
    df_X_Y_Z_are_deterministic
):
    params = {
       "data": df_X_Y_Z_are_deterministic(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params) == True

def test_sci_indep_X_and_Y_are_deterministic(
    df_X_and_Y_are_deterministic
):
    params = {
       "data": df_X_and_Y_are_deterministic(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params) == False

def test_sci_indep_X_causes_Y(
    df_X_causes_Y
):
    params = {
       "data": df_X_causes_Y(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params) == False

def test_cmi_indep_X_causes_Y(
    df_X_causes_Y
):
    params = {
       "data": df_X_causes_Y(size=100),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert conditional_mutual_information(**params) \
        != approx(0, abs=0.01)

def test_cmi_indep_Z_causes_X_and_Y(
    df_Z_causes_X_and_Y
):
    params = {
       "data": df_Z_causes_X_and_Y(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert conditional_mutual_information(**params) \
        == approx(0, abs=0.01)

def test_sci_indep_Z_causes_X_and_Y(
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

def test_sci_indep_X_and_Y_cause_Z(
    df_X_and_Y_cause_Z
):
    params_1 = {
       "data": df_X_and_Y_cause_Z(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": []
    }

    assert sci_is_independent(**params_1) == True

    params_2 = {
       "data": df_X_and_Y_cause_Z(size=500),
       "vars_1": ['x'],
       "vars_2": ['y'],
       "conditioning_set": ['z']
    }

    assert sci_is_independent(**params_2) == False

def test_sci_indep_X_Y_Z_indep(
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

def test_multinomial_normalizing_sum():
    mns = multinomial_normalizing_sum(num_classes=4, sample_size=5)

    assert mns == approx(7.51, abs=0.01)


