from syntheval import SynthEval

def evaluate(original_location, classification, dataframes):
    used_metrics = {
        "ks_test": {"sig_lvl": 0.05, "n_perms": 1000},
        "corr_diff": {"mixed_corr": True},
        "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
        "cls_acc": {"F1_type": "micro", "k_folds": 5},
        "eps_risk": {},
    }
    evaluator_indians = SynthEval(original_location)

    if classification:
        df_vals, df_rank = evaluator_indians.benchmark(dataframes, 'Target', rank_strategy='linear',
                                                   **used_metrics)
    else:
        df_vals, df_rank = evaluator_indians.benchmark(dataframes, rank_strategy='linear', **used_metrics)

    return