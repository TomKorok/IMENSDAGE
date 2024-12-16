import numpy as np
from syntheval import SynthEval

import IMENSDAGE

# TODO Statistics:
    # TODO Koglomorov test to test individual columns
    # TODO Mixed correlation matrix difference Pairwise mutual information difference to seee pairwise results
    # TODO Propensity mean squared error (pMSE) to check the distribution of the synth data

# TODO Real use case
    # TODO Classification accuracy test

# TODO Privacy
    # TODO Epsilon identifiability risk

original_indians = IMENSDAGE.read_file('datasets/diabetes.csv')
original_mellitus = IMENSDAGE.read_file('datasets/diabetic_mellitus.arff')

evaluator_indians = SynthEval(original_indians)
evaluator_mellitus = SynthEval(original_mellitus)

models = np.array(["cgan", "gan", "ctgan", "tvae"])

dataframes_indians = {
    "cgan" : IMENSDAGE.read_file('datasets/cgan_data_indians.csv'),
    "gan" : IMENSDAGE.read_file('datasets/gan_data_indians.csv'),
    "ctgan" : IMENSDAGE.read_file('datasets/ctgan_data_indians.csv'),
    "tvae" : IMENSDAGE.read_file('datasets/tvae_data_indians.csv'),
}

dataframes_mellitus = {
    "cgan" : IMENSDAGE.read_file('datasets/cgan_data_mellitus.csv'),
    "gan" : IMENSDAGE.read_file('datasets/gan_data_mellitus.csv'),
    "ctgan" : IMENSDAGE.read_file('datasets/ctgan_data_mellitus.csv'),
    "tvae" : IMENSDAGE.read_file('datasets/tvae_data_mellitus.csv'),
}

used_metrics = {
    "ks_test"   : {"sig_lvl": 0.05, "n_perms": 1000},
    "corr_diff" : {"mixed_corr": True},
    "p_mse"     : {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
    "cls_acc"   : {"F1_type": "micro", "k_folds": 5},
    "eps_risk"  : {},
}

individual_evaluation = False
if individual_evaluation:
    evaluator_indians.evaluate(dataframes_indians["cgan"], 'Outcome', "full_eval").to_csv('resutls/cgan_results_indians.csv', index=False)
    evaluator_indians.evaluate(dataframes_indians["gan"], 'Outcome', "full_eval").to_csv('resutls/gan_results_indians.csv', index=False)
    evaluator_indians.evaluate(dataframes_indians["ctgan"], 'Outcome', "full_eval").to_csv('resutls/ctgan_results_indians.csv', index=False)
    evaluator_indians.evaluate(dataframes_indians["tvae"], 'Outcome', "full_eval")('resutls/tvae_results_indians.csv', index=False)

    evaluator_mellitus.evaluate(dataframes_mellitus["cgan"], 'Outcome', "full_eval").to_csv('resutls/cgan_results_mellitus.csv', index=False)
    evaluator_mellitus.evaluate(dataframes_mellitus["gan"], 'Outcome', "full_eval").to_csv('resutls/gan_results_mellitus.csv', index=False)
    evaluator_mellitus.evaluate(dataframes_mellitus["ctgan"], 'Outcome', "full_eval").to_csv('resutls/ctgan_results_mellitus.csv', index=False)
    evaluator_mellitus.evaluate(dataframes_mellitus["tvae"], 'Outcome', "full_eval").to_csv('resutls/tvae_results_mellitus.csv', index=False)

# running a benchmark across all datasets
df_vals, df_rank = evaluator_indians.benchmark(dataframes_indians,'Outcome',rank_strategy='linear', **used_metrics)
# df_vals, df_rank = evaluator_indians.benchmark(dataframes_indians,'Outcome',rank_strategy='summitation', **used_metrics)

# df_vals, df_rank = evaluator_mellitus.benchmark(dataframes_mellitus,'Outcome',rank_strategy='linear', **used_metrics)
# df_vals, df_rank = evaluator_mellitus.benchmark(dataframes_mellitus,'Outcome',rank_strategy='summitation', **used_metrics)