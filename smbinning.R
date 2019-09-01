
# Used R for coarse classing smbinning() by Decision Tree
# Optimal Binning categorizes a numeric characteristic into bins for ulterior usage in scoring modeling. 
# This process, also known as supervised discretization, utilizes Recursive Partitioning to 
# categorize the numeric characteristic.
# The specific algorithm is Conditional Inference Trees which initially excludes missing values (NA) 
# to compute the cutpoints, adding them back later in the process for the calculation of the Information Value.

setwd("PATH...")
getwd()
# install.packages("smbinning")
library(smbinning)

train<-read.csv(file="train.csv")
train$def_woe <- (1- train$def)

columns <- colnames(train)[-which(names(train) %in% c("def","def_woe"))]
nums <- sapply(train[,columns], is.numeric) 
columns.n <- columns[nums==T]

var <- columns.n[1]
cut_offs<-data.frame(VAR=character(), cuts=integer())

for (var in columns.n){
  coarse<-smbinning(train[,c("def_woe", var)], y="def_woe", x=var, p=0.05)
  if(length(coarse)>1){
    points<-list(coarse$cuts)
    cut_offs<-rbind(cut_offs,as.data.frame(cbind("VAR"=var,"cuts"=points)))
  }
}


cut_offs <- apply(cut_offs,2,as.character)

write.csv(cut_offs, "coarse_r.csv", row.names=FALSE)
