import IMENSDAGE
import pandas as pd
###################################################################################################################
# AE handles all type of datasets as non-classification # if ae_model, gan_model = 'n' --> classification = False #
# CAE handles classification datasets                   # if ae_model, gan_model = 'c' --> classification = True  #
# IGTD_AE                                               #                                                         #
###################################################################################################################

#Step 1 -- Create the class, read the dataset, train the selected autoencoder model
imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)
# do not include categorical data in the round exceptions
imensdage.read_data("source_datasets/diabetes.csv", ['BMI', 'DiabetesPedigreeFunction'], title="Pima Indians", target='Outcome')

selected_ae_normal = {
    "model": "n",
    "latent_size": 64
}
selected_ae_cond = {
    "model": "c",
    "latent_size": 64
}

en_images, en_labels = imensdage.train_ae(ae_model=selected_ae_cond)

selected_gan_dc = {
    "model": "dc",
    "g_dim": 64,
    "d_dim": 64
}
selected_gan_dcc = {
    "model": "dcc",
    "g_dim": 64,
    "d_dim": 64
}

#Step 2 -- Train any generative model and retain the generated synth images
gen_images, gen_labels = imensdage.train_gen_model(en_images, en_labels, gan_model=selected_gan_dcc) #make sure the output img size is compatible with the decoder's input size

#(Optional) Step 3 -- Display the results
imensdage.show_results(en_images, gen_images, en_labels=en_labels, gen_labels=gen_labels)

#Step 4 -- Save the generated synthetic tabular dataset
synth_dataset = imensdage.save_full_synth_set(gen_images, gen_labels)

'''
#(Optional) Step 5 -- Evaluate the synth dataset and compare with others
synth_dataframes = { "cgan" : pd.read_csv('your_file.csv')}
imensdage.evaluate(synth_dataframes)
'''