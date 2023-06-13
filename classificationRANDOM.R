
training<-read.delim("C:\\Users\\inlab\\Desktop\\pan-cancer neuronal net\\Bladder Urothelial Carcinoma\\random forest\\training10.txt")
target_training<-read.delim("C:\\Users\\inlab\\Desktop\\pan-cancer neuronal net\\Bladder Urothelial Carcinoma\\random forest\\label_training10.txt")
set.seed(1)
x <- training
y <- as.factor(target_training$target)

training2<-cbind(training,y)


testing<-read.delim("C:\\Users\\inlab\\Desktop\\pan-cancer neuronal net\\Bladder Urothelial Carcinoma\\random forest\\testing10.txt")

target_testing<-read.delim("C:\\Users\\inlab\\Desktop\\pan-cancer neuronal net\\Bladder Urothelial Carcinoma\\random forest\\label_testing10.txt")




z<-testing

zi<-as.factor(target_testing$target)#factor for classification

#library(e1071)
#svm_tune <- tune(svm, train.x=x, train.y=y, 
                 #kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

colnames(training2)[ncol(training2)]<-c("target")
#svm_model_after_tune<- svm(target ~ ., data=training2, kernel="radial", cost=svm_tune$best.parameters$cost, gamma=svm_tune$best.parameters$gamma,probability = TRUE)
require(randomForest)
#svm_model_after_tune<- svm(target ~ ., data=training2, kernel="linear", cost = 10, scale = FALSE)
#plot(svm_model_after_tune, training2)
ezz<-randomForest(target ~ ., data=training2, ntree = 500)
#pred3 <- predict(ezz, newdata = z,decision.values = TRUE,probability = FALSE)
pred3 <- predict(ezz,newdata = z,type='class')

a<-table(zi,pred3)
library(MLmetrics)

acc<-Accuracy(pred3, zi)
sens<-Sensitivity(zi, pred3)
spec<-Specificity(zi, pred3)

library(randomForest)
library(pROC)
library(ROCR)
library(caret)

pred_auc <- predict(ezz,newdata = z,type='prob')[,2]
rf_pr_test <- prediction(pred_auc, zi)

r_auc_test <- performance(rf_pr_test, measure = "auc")@y.values[[1]] 

perf <- performance(rf_pr_test,"tpr","fpr")
plot(perf,col='red',lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
print(r_auc_test)