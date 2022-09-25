
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import skimage
from skimage.transform import rescale , resize
import numpy as np


gamma_list = [0.01,0.005,0.001,0.0005,0.0001]
c_list = [0.1,0.2,0.5,0.7,1,2,5,10]

h_param_comb = [{'gamma':g,'C': c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)


#GAMMA = 0.001
#C = 0.5

train_frac = 0.8
test_frac = 0.1
dev_frac  = 0.1

print(train_frac)
digits = datasets.load_digits()



_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)



n_samples = len(digits.images)

print()
print(" size of imaes in old dataset is  ")
print(digits.images[1].shape)
print()


digits_modified = []

for i in range(len(digits.images)):
    digits_modified.append(resize(digits.images[i],(5,5),anti_aliasing= True))



print("size of images in modified dataset ")
print(digits_modified[1].shape)
print()

digits_modified_new = np.array(digits_modified)

#data = digits.images.reshape((n_samples, -1))

data = digits_modified_new.reshape((n_samples, -1))



X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size= 1 - train_frac, shuffle= True
)

X_test, X_dev , y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size= (dev_frac)/(test_frac + dev_frac), 
    shuffle= True
)


best_acc = -1.0
best_model = None
best_h_params = None



for cur_h_params in h_param_comb :

    clf = svm.SVC()

    #hyper_params = {'gamma':GAMMA , 'C' : C}
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)


    clf.fit(X_train, y_train)
    #print(cur_h_params)
    predicted_dev = clf.predict(X_dev)


    cur_acc = metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_model =clf
        best_h_params = cur_h_params
        print("found new best acc with : " + str(cur_h_params))
        print("new best val accuracy :" + str(cur_acc))

    
predicted = clf.predict(X_test)



_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(5, 5)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


print("Best hyperparameters were ")
print(best_h_params)

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()
