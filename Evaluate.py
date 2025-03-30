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
                     "c_o_3_16" : pd.read_csv(f'results/synth_datasets/{dataset}_c_o_3_16_dcc_3_16.csv'),
                     "n_o_3_16" : pd.read_csv(f'results/synth_datasets/{dataset}_n_o_3_16_dc_3_16.csv'),
                     "c_o_16" : pd.read_csv(f'results/synth_datasets/{dataset}_c_o16_dcc16.csv'),
                     "n_o_16" : pd.read_csv(f'results/synth_datasets/{dataset}_n_o16_dc16.csv'),
                     }

evaluate(pd.read_csv(f"source_datasets/{dataset}.csv"), synth_dataframes, target)