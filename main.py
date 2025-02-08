import IMENSDAGE


imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)

imensdage.read_data("source_datasets/diabetes.csv", title="Pima Indians", target='Outcome', classification=False)

########################################################################################################
# AE handles all type of datasets as non-classification # if ae_model = 'n' --> classification = False #
# CAE handles classification datasets                   # if ae_model = 'c' --> classification = True  #
# IGTD_AE                                               #                                              #
########################################################################################################

encoded_images = imensdage.train_ae(ae_model='c')
print("encoded_images")

# gen_data = imensdage.train_gen_model(encoded_images) #make sure the output img size is compatible with the decoder's input size

# imensdage.show_results(encoded_images, gen_data, ['BMI', 'DiabetesPedigreeFunction'])

# imensdage.evaluate()