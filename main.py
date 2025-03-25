import IMENSDAGE

ae =  {
    "model": "n",
    "latent_size": 64
}
dc = {
    "model": "dc",
    "g_dim": 64,
    "d_dim": 64
}

cae = {
    "model": "c",
    "latent_size": 64
}
dcc = {
    "model": "dcc",
    "g_dim": 64,
    "d_dim": 64
}


models = [None, 'Outcome']

#Step 1 -- Create the class, read the dataset, train the selected autoencoder model

for model in models:
    imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
    imensdage.fit("source_datasets/pima.csv", target = model)

'''
imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
imensdage.fit("source_datasets/pima.csv", ae_model=ae, gen_model=dc)
'''