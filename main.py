import IMENSDAGE

no16 = { "model": "n_o16"}
co16 = { "model": "c_o16"}
no64 = { "model": "n_o64"}
co64 = { "model": "c_o64"}
ae = {  "model": "n"}
cae = { "model": "c"}
dc64 = {  "model": "dc64"}
dcc64 = { "model": "dcc64"}
dcc16 = {"model": "dcc16"}
dc16 = {"model": "dc16"}

models = [
            [None, no16, dc16],
            ['Outcome', co16, dcc16 ]
          ]

#Step 1 -- Create the class, read the dataset, train the selected autoencoder model

for model in models:
        imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
        imensdage.fit("source_datasets/pima.csv", target = model[0], ae_model = model[1], gen_model=model[2])

'''
imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
imensdage.fit("source_datasets/heart.csv", ae_model=ae, gen_model=dc)
'''