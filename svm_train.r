library(tm)
library(e1071)
library(RTextTools)

folderdir="C:/Users/anjit/Documents/20" #Give your training data location
alldir=DirSource(folderdir, encoding = "UTF-8", recursive=TRUE) #read in the recursive way

news <- Corpus(alldir, readerControl=list(reader=readPlain,language="en"))

news.p <- tm_map(news, function(x) iconv(enc2utf8(x), sub = "byte")) 

#generate DocumentTermMatrix, note its difference from TermDocumentMatrix. Term Document matrix and Document Term Matrix are Transpose of each other.
#the operations are listed in the control list in order: remove punctuations, to lowercases, remove stopwords, strip white spaces, keep words whose lengths are between 3-15, keep word global frequency (number of documents a word occurs in) larger or equal to 5, use SMART weight with ntc options
dtm <- DocumentTermMatrix(news.p, control=list(removePunctuation=TRUE, removeNumbers=TRUE,stemming = TRUE ,tolower=TRUE, stopwords=c("english"), stripWhitespace=TRUE, wordLengths=c(3,15), bounds=list(global=c(5,Inf)), weighting=function(x) weightSMART(x,spec="ntc")))

#construct an empty vector, to be used for holding class labels of the documents
classvec <- vector()

#loop all the files: for each document, make its parent folder name as the class label of this document, and put the class label value in classvec vector
alldir$filelist  #display all the files
for (filedir in alldir$filelist) {

classlabel=basename(dirname(filedir))
classvec=c(classvec,classlabel)

}
summary(classvec)

#factor classvec to let R know it's a categorical variable
classvec <- factor(classvec)

# convert dtm to dtm_matrix using sparse storage
dtm_matrix=as.matrix.csr(as.matrix(dtm))

#specify the features, vector to be predicted, and kernel method in the svm model
svm_model <- svm(dtm_matrix, classvec, kernel="linear")


summary(svm_model)

#Give the location to save Model and its dictionary since it is useful for testing section.
saveRDS(svm_model, "D:/classification/result/svm_model.rds") 
saveRDS(dtm, "D:/classification/result/dtm.rds")
#Svm model and its document term matrix has been saved as svm_model.rds and dtm.rds inside result folder

#To check the accuracy of model.
pred <- predict(svm_model, dtm)

#Confusion matrix of model.
confMatrix = table(classvec, pred)
cm = confMatrix

n = sum(cm) # number of instances
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
accuracy = sum(diag) / n 
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall)
OverallPrecision = mean(precision)
OverallRecall = mean(recall)
OverallF1 = mean(f1)
summary_prf = data.frame(accuracy,OverallPrecision, OverallRecall, OverallF1) 
write.table(summary_prf, file = "D:/classification/result/summary_aprf.csv", sep = ",", row.names = FALSE)
#Accuracy, Precision, Recall and F1 ratio has been saved as summary_aprf.csv inside result folder.

write.table(confMatrix, file = 'D:/classification/result/confusion_matrix.csv',sep = ",")
#Confusion matrix of model has been saved as confusion_matrix.csv inside result folder.
