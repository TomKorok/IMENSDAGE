import IMENSDAGE

no16 = { "model": "no16"}
co16 = { "model": "co16"}
no64 = { "model": "no64"}
co64 = { "model": "co64"}
ae = {  "model": "n"}
cae = { "model": "c"}
dc64 = {  "model": "dc64"}
dcc64 = { "model": "dcc64"}
dcc16 = {"model": "dcc16"}
dc16 = {"model": "dc16"}
n_igtd = { "model": "n_igtd"}
c_igtd = { "model": "c_igtd"}

models = [
            #[None, "n_igtd", "dc_igtd"],
            ["Kfold", "c_igtd", "dcc_igtd"],
            ["Kfold", "co16", "dcc16"],
          ]

#Step 1 -- Create the class, read the dataset, train the selected autoencoder model

for model in models:
        imensdage = IMENSDAGE.IMENSDAGE()
        imensdage.fit("source/datasets/playground.csv", f"{model[1]}_{model[2]}", target = model[0], ae_model = model[1], gen_model=model[2])

'''
imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
imensdage.fit("source/heart.csv", ae_model=ae, gen_model=dc)
'''