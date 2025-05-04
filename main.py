import IMENSDAGE

# 16 sized models old architecture
# {"model": "no16"}; {"model": "dc16"}
# {"model": "co16"}; {"model": "dcc16"}

#64 sized models old architecture
# {"model": "no64"}; {"model": "dc64"}
# {"model": "co64"}; {"model": "dcc64"}

#64 sized AE models new architecture
# {"model": "n"}; {"model": "dc64"}
# {"model": "c"}; {"model": "dcc64"}

#IGTD models
# {"model": "n_igtd"}; {"model": "dc_igtd"}
# {"model": "c_igtd"}; {"model": "dcc_igtd"}

datasets = ["heart", "pima", "diabetic", "breast_cancer"]
targets = ["DEATH_EVENT", "Outcome", "TYPE", "diagnosis"]
datasets = ["breast_cancer"]
targets = ["diagnosis"]

for dataset, target in zip(datasets, targets):
    """
    models = [[target, "c_igtd", "dcc_igtd"],
                [target, "co16", "dcc16"],]

    for i in range(5):
        for model in models:
            imensdage = IMENSDAGE.IMENSDAGE()
            imensdage.fit(f"source/datasets/{dataset}.csv", f"{model[1]}_{model[2]}_rep_{i}", target = model[0], ae_model = model[1], gen_model=model[2])
            if i == 2:
                imensdage.multiple_sampling()
    """
    models = [[target, "n", "dc64"],
        [target, "c", "dcc64"]]

    for model in models:
        imensdage = IMENSDAGE.IMENSDAGE()
        imensdage.fit(f"source/datasets/{dataset}.csv", f"{model[1]}_{model[2]}", target = model[0], ae_model = model[1], gen_model=model[2])
