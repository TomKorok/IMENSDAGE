import IMENSDAGE

imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32, n_classes = 2)

imensdage.read_data("datasets/diabetes.csv", title="Pima Indians", target='Outcome')
encoded_images = imensdage.train_ae(ae_model='c')
gen_data = imensdage.train_gen_model(encoded_images) #make sure the output img size is compatible with the decoder's input size
imensdage.show_results(encoded_images, gen_data, ['BMI', 'DiabetesPedigreeFunction'])
imensdage.evaluate()