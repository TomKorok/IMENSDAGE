from syntheval import SynthEval
import pandas as pd

def evaluate(original_df, dataframes, target=None):
    used_metrics = {
        "ks_test": {"sig_lvl": 0.05, "n_perms": 1000},
        "corr_diff": {"mixed_corr": True},
        "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
        "cls_acc": {"F1_type": "micro", "k_folds": 5},
        "eps_risk": {},
    }

    evaluator = SynthEval(original_df)

    if target:
        df_vals, df_rank = evaluator.benchmark(dataframes, target, rank_strategy='linear',
                                                   **used_metrics)
    else:
        df_vals, df_rank = evaluator.benchmark(dataframes, rank_strategy='linear', **used_metrics)

    return


synth_dataframes = { "normal_pima" : pd.read_csv(f'results/synth_datasets/pima_n_dc_LN.csv'),
                     "cond_pima" : pd.read_csv(f'results/synth_datasets/pima_c_dcc_LTC.csv'),}

evaluate(pd.read_csv("source_datasets/pima.csv"), synth_dataframes, 'Outcome')