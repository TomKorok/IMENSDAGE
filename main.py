import IMENSDAGE


imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32)

imensdage.read_data("source_datasets/diabetes.csv", ['BMI', 'DiabetesPedigreeFunction'], title="Pima Indians", target='Outcome', classification=True)

########################################################################################################
# AE handles all type of datasets as non-classification # if ae_model = 'n' --> classification = False #
# CAE handles classification datasets                   # if ae_model = 'c' --> classification = True  #
# IGTD_AE                                               #                                              #
########################################################################################################

ae_data = imensdage.train_ae(ae_model='c')
print("encoded_images")

gen_data = imensdage.train_gen_model(ae_data["encoded_images"], ae_data["real_labels"]) #make sure the output img size is compatible with the decoder's input size

imensdage.show_results(ae_data["encoded_images"], gen_data)

fake_records = imensdage.get_full_fake_set(gen_data)

# imensdage.evaluate()