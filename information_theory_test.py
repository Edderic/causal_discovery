import pandas as pd
import numpy as np
import pytest
from pytest import approx
from information_theory import entropy, conditional_entropy, multinomial_normalizing_sum, conditional_mutual_information
from constraint_based.ci_tests.sci_is_independent import sci_is_independent

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

    assert entropy(data=df, variables=['x'], base_2=True) == approx(2, abs=0.01)

def test_cond_ent_uniform_multinomial_with_4_possible_values(
    df_2_multinomial_indep_RVs
):
    assert conditional_entropy(
               data=df_2_multinomial_indep_RVs(size=10000),
               variables=['x', 'y'],
               conditioning_set=['y'],
               base_2=True
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

# Page 10 of
# https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/23025/2/hdl_23025.pdf
# says the normalizing constant should be 1.53 for num_classes 2 and
# sample_size 10
def test_binomial_example_with_10_data_points():
    mns = multinomial_normalizing_sum(num_classes=2, sample_size=10)

    assert np.log(mns) == approx(1.53, abs=0.01)

def test_binomial_example_with_10_data_points_mns0():
    mns = multinomial_normalizing_sum(num_classes=2, sample_size=2)

    assert mns == approx(2.5, abs=0.01)
