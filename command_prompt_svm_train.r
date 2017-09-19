library(tm)
library(e1071)
library(RTextTools)

cat(" Enter the location of directory for training model: ")
location<- readLines("stdin", 1)

  start.time <- Sys.time()
alldir=DirSource(location, encoding = "UTF-8", recursive=TRUE)
news <- Corpus(alldir, readerControl=list(reader=readPlain,language="en"))
news.p <- tm_map(news, function(x) iconv(enc2utf8(x), sub = "byte")) 

#generate DocumentTermMatrix, note its difference from TermDocumentMatrix
dtm <- DocumentTermMatrix(news.p, control=list(removePunctuation=TRUE, removeNumbers=TRUE,stemming = TRUE ,tolower=TRUE, stopwords=c("english"), stripWhitespace=TRUE, wordLengths=c(3,15), bounds=list(global=c(5,Inf)), weighting=function(x) weightSMART(x,spec="ntc")))
#the operations are listed in the control list in order: remove punctuations, to lowercases, remove stopwords, strip white spaces, keep words whose lengths are between 3-15, keep word global frequency (number of documents a word occurs in) larger or equal to 5, use SMART weight with ntc options

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

end.time <- Sys.time()
	time.taken <- round(end.time - start.time,2)
	cat(" \n\n\nTime taken to train the model:\n ")
	time.taken


summary(svm_model)
saveRDS(svm_model, "D:/classification/result/svm_model.rds")
saveRDS(dtm, "D:/classification/result/dtm.rds")
cat("\n\nSvm model and its document term matrix has been saved as svm_model.rds and dtm.rds inside result folder.\n\n ")

pred <- predict(svm_model, dtm)
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
cat("\n\nAccuracy, Precision, Recall and F1 ratio has been saved as summary_aprf.csv inside result folder.\n\n ")
write.table(confMatrix, file = 'D:/classification/result/confusion_matrix.csv',sep = ",")
cat("\n\nConfusion matrix of model has been saved as confusion_matrix.csv inside result folder.\n\n ")
