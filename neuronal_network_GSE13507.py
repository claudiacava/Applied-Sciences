import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler


myfile= pd.read_csv('GSE13507_label.csv')
print(myfile['target'].value_counts())
print(myfile.shape)


#label_train=myfile[:16]
#label_test=myfile[16:]






myfile_samplet= pd.read_csv('GSE13507_series_matrix_trasp_mod.csv',header=None)

print(myfile_samplet.shape)
print(len(myfile_samplet))
print(len(myfile_samplet.columns))

# split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test= train_test_split(myfile_samplet,
                                                               myfile,
                                                              test_size=0.30,
                                                               random_state=2)
print(len(X_train))
print(len(X_train.columns))
print(y_train)

print(len(X_test))
print(len(X_test.columns))
print(y_test)




ros = RandomOverSampler(random_state=2)

X_train, y_train = ros.fit_resample(X_train, y_train)
X_test, y_test = ros.fit_resample(X_test, y_test)


print(len(X_train))
print(len(X_train.columns))
print(y_train)

print(len(X_test))
print(len(X_test.columns))
print(y_test)



scaler=StandardScaler()
sample_train_scaled=scaler.fit_transform(X_train)

sample_test_scaled=scaler.fit_transform(X_test)


pca=PCA(n_components=0.95)
sample_train_pca=pca.fit_transform(sample_train_scaled)
sample_test_pca=pca.transform(sample_test_scaled)


print(len(sample_train_pca))
print(sample_train_pca.shape)
print(sample_test_pca.shape)



NN_model=Sequential([
    Dense(17,activation='relu',input_shape=sample_train_pca[1].shape),
    Dense(8,activation='relu'),
    Dense(1,activation='sigmoid'),
])




NN_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)



#NN_model.compile(
 #   loss='mean_squared_logarithmic_error',
  #  optimizer='adam',
   # metrics=['accuracy']
#)




early_stopping=EarlyStopping(
    patience=5,
    min_delta=0.005,
    restore_best_weights=True,
)



train_history=NN_model.fit(
 sample_train_pca,y_train,
 validation_data=(sample_test_pca,y_test),
 batch_size=8,
 epochs=200,
 callbacks=[early_stopping]
 )









###############################################
# EVALUATING THE MODEL  ######################
score = NN_model.evaluate(sample_test_pca, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


predict_x=NN_model.predict(sample_test_pca)

auc = metrics.roc_auc_score(y_test, predict_x)

#classes_x=np.argmax(predict_x,axis=1)

##############################################
## # extract the predicted class labels ######

pr=np.where(predict_x > 0.5, 1,0)

print(pr)
#print(classes_x)
#print(classes_x)
#print(label_test)
cm=confusion_matrix(y_test,pr)
print(cm)




##############################################

#####from confusion matrix calculate accuracy

TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]




print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)








#total1=sum(sum(cm))

# calculate accuracy
conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
        
# calculate the sensitivity
conf_sensitivity = (TP / float(TP + FN))

# calculate the specificity
conf_specificity = (TN / float(TN + FP))
    

print('accuracy:', conf_accuracy)
print('sensitivity:', conf_sensitivity)
print('specificity:', conf_specificity)

print('auc:', auc)

#ax=plt.subplot()

#sns.heatmap(cm,annot=True,ax=ax,fmt='g',cmap='Greens')


#print(ax)

#ax.set_xlabel('Predicted labels')
#ax.set_ylabel('True labels')

#ax.set_title('Neuronal Network Confusion Matrix')
#ax.xaxis.set_ticklabels(labels)
#plt.show()



fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_x)

#plt.figure(0).clf()


#create ROC curve
plt.plot(fpr,tpr)


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print('fpr:', fpr)
print('tpr:', tpr)



    


