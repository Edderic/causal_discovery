import numpy as np
import pandas as pd

DAYS_OF_WEEK = np.array(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

def collapse_multinomial(multinomial, labels):
    num_rows, num_cols = multinomial.shape

    series = pd.Series(np.repeat([labels[0]], num_rows))

    for col_index in range(num_cols):
        series[np.where(multinomial.T[col_index])[0]] = labels[col_index]

    return series

def next_day(day):
    day_index = np.where(DAYS_OF_WEEK == day)[0][0]

    if day_index == len(DAYS_OF_WEEK) - 1:
        return DAYS_OF_WEEK[0]

    return DAYS_OF_WEEK[int(day_index) + 1]

def dog_example(size=10000):
    weekend = np.random.binomial(n=1, p=0.28, size=size)
    rain = np.random.binomial(n=1, p=0.2, size=size)
    best_friends_visit = \
        weekend * rain * np.random.binomial(n=1, p=0.15, size=size) \
        + (weekend == 0) * rain * np.random.binomial(n=1, p=0.01, size=size) \
        + weekend * (rain == 0) * np.random.binomial(n=1, p=0.35, size=size) \
        + (weekend == 0) * (rain == 0) * np.random.binomial(n=1, p=0.13, size=size)

    _activity_rain = np.random.multinomial(n=1, pvals=[0.9, 0.05, 0.05], size=size)
    _activity_not_rain = np.random.multinomial(n=1, pvals=[0.1, 0.4, 0.5], size=size)

    exercise_levels_act_go_to_park_best_friends_visit = np.random.multinomial(n=1, pvals=[0.9, 0.05, 0.05], size=size)
    exercise_levels_act_go_to_dog_park_best_friends_visit = np.random.multinomial(n=1, pvals=[0.9, 0.05, 0.05], size=size)
    exercise_levels_act_stay_inside_best_friends_visit = np.random.multinomial(n=1, pvals=[0.8, 0.1, 0.1], size=size)

    exercise_levels_act_go_to_park_best_friends_dont_visit = np.random.multinomial(n=1, pvals=[0.6, 0.30, 0.10], size=size)
    exercise_levels_act_go_to_dog_park_best_friends_dont_visit = np.random.multinomial(n=1, pvals=[0.6, 0.35, 0.05], size=size)
    exercise_levels_act_stay_inside_best_friends_dont_visit = np.random.multinomial(n=1, pvals=[0.1, 0.4, 0.5], size=size)

    activity = collapse_multinomial(
        _activity_rain * np.repeat(rain.reshape((size, 1)), 3, axis=1) \
        + _activity_not_rain * np.repeat((rain == 0).reshape((size, 1)), 3, axis=1),
        ['stay inside', 'go to dog park', 'go to park']
    )

    _exercise_levels = \
        exercise_levels_act_go_to_park_best_friends_visit \
        * np.repeat((best_friends_visit * (activity == 'go to park').values).reshape((size, 1)), 3, axis=1)\
        + exercise_levels_act_go_to_dog_park_best_friends_visit \
        * np.repeat((best_friends_visit * (activity == 'go to dog park').values).reshape((size, 1)), 3, axis=1)\
        + exercise_levels_act_stay_inside_best_friends_visit \
        * np.repeat((best_friends_visit * (activity == 'stay inside').values).reshape((size, 1)), 3, axis=1)\
        + exercise_levels_act_go_to_park_best_friends_dont_visit \
        * np.repeat(((best_friends_visit == 0) * (activity == 'go to park').values).reshape((size, 1)), 3, axis=1)\
        + exercise_levels_act_go_to_dog_park_best_friends_dont_visit \
        * np.repeat(((best_friends_visit == 0) * (activity == 'go to dog park').values).reshape((size, 1)), 3, axis=1)\
        + exercise_levels_act_stay_inside_best_friends_dont_visit \
        * np.repeat(((best_friends_visit == 0) * (activity == 'stay inside').values).reshape((size, 1)), 3, axis=1)\

    exercise_levels = collapse_multinomial(_exercise_levels, ['high', 'med', 'low'])

    dog_tired = (exercise_levels == 'high') * np.random.binomial(n=1, p=0.97, size=size) \
        + (exercise_levels == 'med') * np.random.binomial(n=1, p=0.8, size=size) \
        + (exercise_levels == 'low') * np.random.binomial(n=1, p=0.4, size=size)

    mentally_exhausted_before_bed = \
        np.random.binomial(n=1, p=0.8, size=size) \
        * best_friends_visit \
        * (activity == 'stay inside') \
        + np.random.binomial(n=1, p=0.2, size=size) \
        * best_friends_visit \
        * (activity == 'go to dog park') \
        + np.random.binomial(n=1, p=0.3, size=size) \
        * best_friends_visit \
        * (activity == 'go to park') \
        + np.random.binomial(n=1, p=0.9, size=size) \
        * (best_friends_visit == 0) \
        * (activity == 'stay inside') \
        + np.random.binomial(n=1, p=0.3, size=size) \
        * (best_friends_visit == 0) \
        * (activity == 'go to dog park') \
        + np.random.binomial(n=1, p=0.4, size=size) \
        * (best_friends_visit == 0) \
        * (activity == 'go to park')

    dog_teeth_brushed = \
        np.random.binomial(n=1, p=0.90, size=size) \
        * dog_tired \
        * (mentally_exhausted_before_bed == 0) \
        + np.random.binomial(n=1, p=0.4, size=size) \
        * (dog_tired) \
        * (mentally_exhausted_before_bed == 1) \
        + np.random.binomial(n=1, p=0.1, size=size) \
        * (dog_tired == 0)

    df = pd.DataFrame({
        'weekend': weekend,
        'best_friends_visit': best_friends_visit,
        'rain': rain,
        'exercise_levels': exercise_levels,
        'dog_tired': dog_tired,
        'mentally_exhausted_before_bed': mentally_exhausted_before_bed,
        'dog_teeth_brushed': dog_teeth_brushed,
        'activity': activity
    })

    return df

