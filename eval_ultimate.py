from syntheval import SynthEval
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def align_columns(df, all_columns):
    missing_columns = all_columns - set(df.columns)
    for col in missing_columns:
        df[col] = 0
    return pd.concat([df.iloc[:, 0], df.iloc[:, 1:].sort_index(axis=1)], axis=1)

def stat_plot(df_array, legend, title):
    all_columns = set(df_array[0].columns) | set(df_array[1].columns) | set(df_array[2].columns) | set(df_array[3].columns)
    df_array[0] = align_columns(df_array[0], all_columns)
    df_array[1] = align_columns(df_array[1], all_columns)
    df_array[2] = align_columns(df_array[2], all_columns)
    df_array[3] = align_columns(df_array[3], all_columns)

    colors = ['darkblue', 'deepskyblue', 'darkorange', 'gold']
    numeric_df = pd.DataFrame()
    x = []

    for i, df in enumerate(df_array):
        # Drop the first column (assumed to be non-numeric / labels)
        numeric_df = df.drop(df.columns[0], axis=1)

        # Get mean and SE from the last two rows
        mean = numeric_df.iloc[-2].astype(float).values
        se = numeric_df.iloc[-1].astype(float).values
        x = np.arange(len(mean))

        plt.errorbar(x, mean, yerr=se, label=legend[i], fmt='o', capsize=5, color=colors[i])

    # X-axis labels based on column names (excluding the first column)
    plt.xticks(x, numeric_df.columns, rotation=25, fontsize=7)
    plt.ylabel("Value")
    plt.title(f"Mean ± SE for {title}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"results/stats/stat_{title}.png")
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
    df = df.drop(['frac_ks_sigs_error', 'corr_mat_diff_error', 'eps_identif_risk_error', 'ks_tvd_stat_error', 'avg_pMSE_error', 'cls_F1_diff_error', 'f_rank_'], axis=1)
    df.columns = df.columns.str.replace(r'_value$', '', regex=True)
    if save:
        save_df(df, f'{dataset}_{title}')
    matched_files = glob.glob('*SE_benchmark_results_*.csv')
    original_file = matched_files[0]
    os.remove(original_file)
    return df

def fid_scaling(score):
    scaler = 1
    if score > 100:
        scaler = 0.001
        fid_col_name = "fid/1000"
    elif score > 10:
        scaler = 0.01
        fid_col_name = "fid/100"
    elif score > 1:
        scaler = 0.1
        fid_col_name = "fid/10"
    else:
        fid_col_name = "fid"

    return scaler, fid_col_name

def expand_metrics(df, dataset):
    df = df.drop(['rank_', 'u_rank_', 'p_rank_'], axis=1)
    fid_scaled = False
    fid_scaler = 1
    fid_col_name = "fid"
    for index, row in df.iterrows():
        min_max_result = min_max_eval(dataset, row["dataset"])
        for _, row_mm in min_max_result.iterrows():
            df.at[index, f"mm{row_mm['Feature']}"[:10]] = round(row_mm["Change (%)"] / 100, 4)
        fid_score = pd.read_csv(f"results/metrics/fid_{dataset}_{row['dataset']}.csv", sep=";", decimal=",")["FID Score"].iloc[0]
        if not fid_scaled:
            fid_scaler, fid_col_name = fid_scaling(fid_score)
            fid_scaled = True
        scaled_fid = fid_score * fid_scaler
        df.at[index, fid_col_name] = scaled_fid

    return pd.concat([df.iloc[:, 0], df.iloc[:, 1:].sort_index(axis=1)], axis=1)

def stat_analysis(df):
    numeric_df = df.iloc[:, 1:]
    means = numeric_df.mean()
    standard_errors = numeric_df.std(ddof=1) / np.sqrt(numeric_df.count())

    # Append to df as new rows
    df.loc['Mean'] = ['Mean'] + means.tolist()
    df.loc['Std Error'] = ['Std Error'] + standard_errors.tolist()
    return df

def save_df(df, title):
    df.to_csv(f'results/stats/{title}.csv', index=False, sep=';', decimal=',', float_format='%.4f')

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
    reps_igtd_df = stat_analysis(df)
    save_df(reps_igtd_df, f"{dataset}_reps_igtd.csv" )


    # running the full syntheval framework to compare diff samplings for the igtd model
    synth_dataframes_samples_igtd = { "c_igtd_dcc_igtd_rep_2_sample_0" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_0.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_1" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_1.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_2" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_2.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_3" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_3.csv'),
                                      "c_igtd_dcc_igtd_rep_2_sample_4" : pd.read_csv(f'results/synth_datasets/{dataset}_c_igtd_dcc_igtd_rep_2_sample_4.csv'),
                                  }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_samples_igtd, target)
    df = cleanup_after_syntheval(False)
    df = expand_metrics(df, dataset)
    samples_igtd_df = stat_analysis(df)
    save_df(samples_igtd_df, f"{dataset}_samples_igtd.csv")

    # running the full syntheval framework to compare diff training for the co16 model
    synth_dataframes_reps_co16 = {
        "co16_dcc16_rep_0": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_0.csv'),
        "co16_dcc16_rep_1": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_1.csv'),
        "co16_dcc16_rep_2": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2.csv'),
        "co16_dcc16_rep_3": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_3.csv'),
        "co16_dcc16_rep_4": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_4.csv'),
        }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_reps_co16, target)
    df = cleanup_after_syntheval(False)
    df = expand_metrics(df, dataset)
    reps_co16_df = stat_analysis(df)
    save_df(reps_co16_df, f"{dataset}_reps_co16.csv")

    # running the full syntheval framework to compare diff samplings for the co16 model
    synth_dataframes_samples_co16 = {
        "co16_dcc16_rep_2_sample_0": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_0.csv'),
        "co16_dcc16_rep_2_sample_1": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_1.csv'),
        "co16_dcc16_rep_2_sample_2": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_2.csv'),
        "co16_dcc16_rep_2_sample_3": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_3.csv'),
        "co16_dcc16_rep_2_sample_4": pd.read_csv(f'results/synth_datasets/{dataset}_co16_dcc16_rep_2_sample_4.csv'),
        }
    evaluate(pd.read_csv(f"source/datasets/{dataset}.csv"), synth_dataframes_samples_co16, target)
    df = cleanup_after_syntheval(False)
    df = expand_metrics(df, dataset)
    samples_co16_df = stat_analysis(df)
    save_df(samples_co16_df, f"{dataset}_samples_co16.csv")

    stat_plot([reps_igtd_df, samples_igtd_df, reps_co16_df, samples_co16_df], ["reps_igtd", "samples_igtd", "reps_co16", "samples_co16"], f"{dataset}")


print("Analysis complete. Results saved.")