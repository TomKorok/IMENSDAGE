from syntheval import SynthEval
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def stat_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot original data
    df.drop(['Mean', 'Std Error']).plot(kind='bar', ax=ax, yerr=standard_errors, capsize=5)

    # Add mean as a line
    ax.axhline(means['A'], color='blue', linestyle='--', label=f"Mean A: {means['A']}")
    ax.axhline(means['B'], color='orange', linestyle='--', label=f"Mean B: {means['B']}")
    ax.axhline(means['C'], color='green', linestyle='--', label=f"Mean C: {means['C']}")

    # Customizing the plot
    ax.set_title('Data with Mean and Standard Error')
    ax.set_ylabel('Values')
    ax.set_xlabel('Categories')
    ax.legend()

    plt.show()

def evaluate(original_df, dataframes, target=None):
    used_metrics = {
        "ks_test": {"sig_lvl": 0.05, "n_perms": 1000},
        "corr_diff": {"mixed_corr": True},
        "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
        "cls_acc": {"F1_type": "micro", "k_folds": 5},
        "eps_risk": {},
    }

    evaluator = SynthEval(original_df)

    # try submason ranking when comparing the same model in different runs
    if target:
        df_vals, df_rank = evaluator.benchmark(dataframes, target, rank_strategy='linear',
                                               **used_metrics)
    else:
        df_vals, df_rank = evaluator.benchmark(dataframes, rank_strategy='linear', **used_metrics)

    return

def cleanup_after_syntheval(save, dataset=None, title=None):
    matched_files = glob.glob('*SE_benchmark_ranking_*.csv')
    original_file = matched_files[0]
    df = pd.read_csv(original_file)
    os.remove(original_file)
    df = df.drop(['frac_ks_sigs_error', 'corr_mat_diff_error', 'eps_identif_risk_error'], axis=1)
    if save:
        save_df(df, f'{dataset}_{title}')
    matched_files = glob.glob('*SE_benchmark_results_*.csv')
    original_file = matched_files[0]
    os.remove(original_file)
    return df

def expand_metrics(df, dataset):
    df["fid"] = None
    df = df.drop(['rank_', 'u_rank_', 'p_rank_', 'f_rank_'], axis=1)
    for index, row in df.iterrows():
        min_max_result = min_max_eval(dataset, row["dataset"])
        for _, row_mm in min_max_result.iterrows():
            df.at[index, f"min_max_{row_mm['Feature']}"] = round(row_mm["Change (%)"] / 100, 4)
        fid_score = \
        pd.read_csv(f"results/metrics/fid_{dataset}_{row['dataset']}.csv", sep=";", decimal=",")["FID Score"].iloc[0]
        df.at[index, "fid"] = fid_score

    return df

def stat_analysis(df):
    numeric_df = df.iloc[:, 1:]
    means = numeric_df.mean()
    standard_errors = numeric_df.std(ddof=1) / np.sqrt(numeric_df.count())

    # Append to df as new rows
    df.loc['Mean'] = ['Mean'] + means.tolist()
    df.loc['Std Error'] = ['Std Error'] + standard_errors.tolist()
    return df

def save_df(df, title):
    df.to_csv(f'{title}.csv', index=False, sep=';', decimal=',', float_format='%.4f')

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

    result_df.to_csv(f"results/metrics/{dataset}_{model}_min_max.csv", index=False, sep=';', decimal=',')
    return result_df

datasets = ["pima", "diabetic", "heart"]
targets = ["Outcome", "TYPE", "DEATH_EVENT"]

for dataset, target in zip(datasets, targets):
    # running the full syntheval framework to compare my models with the state-of-the-art models
    synth_dataframes_sota = {   "tvae" : pd.read_csv(f'results/synth_datasets/{dataset}_tvae.csv'),
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
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_sota, target)
    cleanup_after_syntheval(True, dataset, "sota_comparison")

    # running the full syntheval framework to compare diff training for the igtd model
    synth_dataframes_reps_igtd = { "c_igtd_dcc_igtd_rep_0" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_0.csv'),
                                   "c_igtd_dcc_igtd_rep_1" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_1.csv'),
                                   "c_igtd_dcc_igtd_rep_2" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2.csv'),
                                   "c_igtd_dcc_igtd_rep_3" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_3.csv'),
                                   "c_igtd_dcc_igtd_rep_4" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_4.csv'),
                             }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_reps_igtd, target)
    df = cleanup_after_syntheval(False)
    df = expand_metrics(df, dataset)
    df = stat_analysis(df)
    save_df(df, f"{dataset}_reps_igtd.csv" )


    # running the full syntheval framework to compare diff samplings for the igtd model
    synth_dataframes_samples_igtd = { "c_igtd_dcc_igtd_rep_2_sample_0" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_0.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_1" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_1.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_2" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_2.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_3" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_3.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_4" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_4.csv'),
                                  }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_samples_igtd, target)
    df = cleanup_after_syntheval(False)
    df = cleanup_after_syntheval(False)
    df = expand_metrics(df, dataset)
    df = stat_analysis(df)
    save_df(df, f"{dataset}_samples_igtd.csv")

    # running the full syntheval framework to compare diff training for the co16 model
    synth_dataframes_reps_co16 = {
        "co16_dcc16_rep_0": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_0.csv'),
        "co16_dcc16_rep_1": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_1.csv'),
        "co16_dcc16_rep_2": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2.csv'),
        "co16_dcc16_rep_3": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_3.csv'),
        "co16_dcc16_rep_4": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_4.csv'),
        }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_reps_co16, target)
    cleanup_after_syntheval(False)

    # running the full syntheval framework to compare diff samplings for the co16 model
    synth_dataframes_samples_co16 = {
        "co16_dcc16_rep_2_sample_0": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_0.csv'),
        "co16_dcc16_rep_2_sample_1": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_1.csv'),
        "co16_dcc16_rep_2_sample_2": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_2.csv'),
        "co16_dcc16_rep_2_sample_3": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_3.csv'),
        "co16_dcc16_rep_2_sample_4": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_4.csv'),
        }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_samples_co16, target)
    cleanup_after_syntheval(False)

    #TODO mean and error across min max

    #TODO mean and error across syntheval metrics

    #TODO mean and error across FID

print("Analysis complete. Results saved.")