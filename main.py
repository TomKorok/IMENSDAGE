import IMENSDAGE

imensdage = IMENSDAGE.IMENSDAGE(batch_size = 32, n_classes = 2)

fake_samples = imensdage.fit("datasets/diabetes.csv", ['BMI', 'DiabetesPedigreeFunction'], "Pima Indians", conditional='c')
fake_samples.to_csv(f'datasets/cgan_data_indians.csv', index=False)

fake_samples = imensdage.fit("datasets/diabetic_mellitus.arff", ['BMI'], "Mellitus", conditional='c')
fake_samples.to_csv(f'datasets/cgan_data_mellitus.csv', index=False)