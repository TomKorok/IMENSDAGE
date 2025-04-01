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

dataset = 'pima'
target = 'Outcome'

synth_dataframes = { "tvae" : pd.read_csv(f'results/synth_datasets/{dataset}_tvae.csv'),
                     "ctgan" : pd.read_csv(f'results/synth_datasets/{dataset}_ctgan.csv'),
                     "ddpm" : pd.read_csv(f'results/synth_datasets/{dataset}_ddpm.csv'),
                     "rtvae" : pd.read_csv(f'results/synth_datasets/{dataset}_rtvae.csv'),
                     "decaf" : pd.read_csv(f'results/synth_datasets/{dataset}_decaf.csv'),
                     "n_igtd" : pd.read_csv(f'results/synth_datasets/{dataset}_n_igtd_dc_igtd.csv'),
                     "igtd_low" : pd.read_csv(f'results/synth_datasets/{dataset}_igtd_lowparam.csv'),
                     "co16" : pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16.csv'),
                     }

evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes, target)