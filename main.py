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

dataset = "playground"
target = "Kfold"

models = [
            [target, "c_igtd", "dcc_igtd"],
            [target, "co16", "dcc16"],
          ]

#Step 1 -- Create the class, read the dataset, train the selected autoencoder model

for i in range(5):
    for model in models:
        imensdage = IMENSDAGE.IMENSDAGE()
        imensdage.fit(f"source/datasets/{dataset}.csv", f"{model[1]}_{model[2]}_rep_{i}", target = model[0], ae_model = model[1], gen_model=model[2])
        if i == 3:
            imensdage.multiple_sampling()
