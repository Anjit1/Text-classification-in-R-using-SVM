library(tm)
library(RTextTools)
library(caret)
library(e1071)

#Reading previously saved training's dictionary and model.
dtm<- readRDS("D:/classification/result/dtm.rds")
svm_model <-readRDS("D:/classification/result/svm_model.rds")

location_test<- "Your test docs full location"

#Reads in recursive way.
filenames <- list.files(location_test, pattern="*.*", full.names=F,include.dirs = F , recursive=TRUE)

#Creating Corpus.
docs=VCorpus(DirSource(location_test, pattern = "*.*", recursive = TRUE, mode = "text")) 
 corpus <- VCorpus(VectorSource(docs))
   
#Here dictionary of Train model is used as dtm inside control function.   
  tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(dtm), 
                                                     removePunctuation = TRUE, 
                                                     stopwords = TRUE, 
                                                    stemming = TRUE, 
                                                     removeNumbers = TRUE))
    test <- as.matrix(tdm)
    
    # Check accuracy on test.
    a= predict(svm_model, newdata = test)
	
#Extra cloumn named text_topic is added of which each rows is original filenames with its succesive predicted topic.    
    n <- cbind(filenames)
    colnames(n)[ncol(n)] <- 'text_topic'
    n <- as.data.frame(n)
    abcd <- data.frame(a,n)
#Inside file='', you can give your desire path to save your final result.	
write.table(abcd, file = 'final_result.csv',sep = ",", row.names=FALSE,col.names=FALSE)
cat(" \n\n\nPredicted result has been saved as .CSV files inside result folder as final_result.csv .")
