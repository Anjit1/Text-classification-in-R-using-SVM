library(tm)
library(RTextTools)
library(caret)
library(e1071)

dtm<- readRDS("D:/classification/result/dtm.rds")
svm_model <-readRDS("D:/classification/result/svm_model.rds")

cat(" \n Enter the directory for your test set for model:\n")
location_test<- readLines("stdin", 1)

filenames <- list.files(location_test, pattern="*.*", full.names=F,include.dirs = F , recursive=TRUE)
docs=VCorpus(DirSource(location_test, pattern = "*.*", recursive = TRUE, mode = "text")) 

    data2 <- docs
  start.time <- Sys.time()
 
  corpus <- VCorpus(VectorSource(data2))
    
  tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(dtm),    
                                                     removePunctuation = TRUE, 
                                                     stopwords = TRUE, 
                                                    stemming = TRUE, 
                                                     removeNumbers = TRUE))
    test <- as.matrix(tdm)
    
    # Check accuracy on test.
    a= predict(svm_model, newdata = test)
	
	
	
	end.time <- Sys.time()
	time.taken <- round(end.time - start.time,2)
	cat(" \n\n\nTime taken to predict the topic:\n ")
	time.taken
	cat(" \n\n\n ")
	
    n <- cbind(filenames)
    colnames(n)[ncol(n)] <- 'text_topic'
    n <- as.data.frame(n)
    #b= write.table(n, file = "a.csv", sep= ",", row.names = FALSE)
    abcd <- data.frame(a,n)
	
cat(" Enter the location to save your final result:\n ")
location_save_result<- readLines("stdin", 1)
setwd(location_save_result)
	
write.table(abcd, file = 'final_result.csv',sep = ",", row.names=FALSE,col.names=FALSE)
cat(" \n\n\nPredicted result has been saved as .CSV files inside result folder as final_result.csv .")
