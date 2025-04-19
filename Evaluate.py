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

    #try submason ranking when comparing the same model in different runs
    if target:
        df_vals, df_rank = evaluator.benchmark(dataframes, target, rank_strategy='linear',
                                                   **used_metrics)
    else:
        df_vals, df_rank = evaluator.benchmark(dataframes, rank_strategy='linear', **used_metrics)

    return


def min_max_eval(dataset, model):
    # Step 1: Read the CSVs
    real_df = pd.read_csv(f"source/datasets/{dataset}.csv")
    fake_df = pd.read_csv(f"results/synth_datasets/{dataset}_{model}.csv")

    # Step 2: Keep only numeric columns
    real_df = real_df.select_dtypes(include='number')
    fake_df = fake_df.select_dtypes(include='number')

    # Step 3: Compute min-max ranges
    real_ranges = real_df.max() - real_df.min()
    fake_ranges = fake_df.max() - fake_df.min()

    # Step 4: Compute metrics
    denominator = pd.concat([real_ranges, fake_ranges], axis=1).max(axis=1) + 1e-8
    percent_change = (abs(fake_ranges - real_ranges) / denominator + 1e-8) * 100

    # Step 5: Select top 5 features based on percent change
    top_5_features = percent_change.sort_values(ascending=False).head(5)

    # Step 6: Gather final results
    result_df = pd.DataFrame({
        'Feature': top_5_features.index,
        'Real Range': real_ranges[top_5_features.index].values,
        'Fake Range': fake_ranges[top_5_features.index].values,
        'Change (%)': top_5_features.values
    })

    result_df.to_csv(f"results/metrics/{dataset}_{model}_min_max.csv", index=False,  sep=';', decimal=',')

dataset = "pima"
target = "Outcome"

synth_dataframes = { "tvae" : pd.read_csv(f'results/synth_datasets/{dataset}_tvae.csv'),
                     "ctgan" : pd.read_csv(f'results/synth_datasets/{dataset}_ctgan.csv'),
                     "ddpm" : pd.read_csv(f'results/synth_datasets/{dataset}_ddpm.csv'),
                     "rtvae" : pd.read_csv(f'results/synth_datasets/{dataset}_rtvae.csv'),
                     "decaf" : pd.read_csv(f'results/synth_datasets/{dataset}_decaf.csv'),
                     "co16" : pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16.csv'),
                     "co64" : pd.read_csv(f'results/synth_datasets/{dataset}_co64_dcc64.csv'),
                     "c" : pd.read_csv(f'results/synth_datasets/{dataset}_c_dcc64.csv'),
                     "c_igtd" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd.csv'),
                     "no16" : pd.read_csv(f'results/synth_datasets/{dataset}_no16_dc16.csv'),
                     "no64" : pd.read_csv(f'results/synth_datasets/{dataset}_no64_dc64.csv'),
                     "n" : pd.read_csv(f'results/synth_datasets/{dataset}_n_dc64.csv'),
                     }


evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes, target)

min_max_eval(dataset, "c_igtd_dcc_igtd")
min_max_eval(dataset, "co16_dcc16")

print("Analysis complete. Results saved.")