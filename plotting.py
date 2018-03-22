import numpy as np
import matplotlib.pyplot as plt


train_acc = np.loadtxt("C:\\Users\\Larry\\NilearnStuff\\reg_train_accuracy.txt", dtype=float)
val_acc = np.loadtxt("C:\\Users\\Larry\\NilearnStuff\\reg_validation_accuracy.txt", dtype=float)

train_acc = [0.46] + list(train_acc)
val_acc = [0.46] + list(val_acc)

plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.savefig("C:\\Users\\Larry\\NilearnStuff\\FinalDataset\\reg_accuracies.svg")