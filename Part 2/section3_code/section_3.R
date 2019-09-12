################ Start of Section 3 #################
# =================================================


#librarys 
library(tree)
library(rpart)
library(ipred)
library(mlbench)
library(ggplot2)
library(caret)
library(data.table)
library(class)
library(dplyr)
library(data.table)
library(randomForest)


set.seed(3060)

#*Code below same section 2 to retrieve test data
#=================================================

## Temporarily set working directory to features files directory 
setwd("../training_data/")

## Get list of file names ending in features.csv
files = list.files(pattern="*features.csv")

## Load all the -features.csv files into a single list using fread()
allSymbols = rbindlist(lapply(files, fread))

# Name columns of dataset 
cols = c('label', 'index', 'nr_pix', 'height', 'width', 'tallness',
         'rows_with_1', 'cols_with_1', 'rows_with_5', 'cols_with_5',
         'one_neigh', 'three_neigh', 'none_below', 'none_above',
         'none_before', 'none_after', 'nr_regions', 'nr_eyes',
         'r5_c5', 'bd_decimal', 'r_max_pixel', 'c_max_pixel')
colnames(allSymbols)= cols
#============================================

## Return working directory to source file location 
setwd("../section3_code/")

#*============================================


################ 3.1 #################
# ====================================

## Bagging of decision trees with different number of bags 
## using 5 fold CV to check accuracy of each model

## First set cross-validataion train control
control.bag<- trainControl(method = "cv", number=5)
## number of bags used each time 
nbags= c(25, 50,200,400)
accuracies = c()
for(nbag in nbags){
  cv.bagged<- train(factor(label)~ .-index, data=allSymbols, 
                    method="treebag", trControl = control.bag, nbagg= nbag, 
                    oob_score=TRUE)
  accuracies = cbind(accuracies, cv.bagged$results$Accuracy)
  print(nbag)
}

## Create a dataframe of the CV accuracy for each nbags
bagged<-data.frame(accuracy= t(accuracies), nbags)

## 
0.8277778-0.8266440

## Plot bagged
plot.bagged<- ggplot(bagged, aes(x=nbags, y =accuracy))+
  geom_line()
plot.bagged

## Set a path to 'plots' folder to save all plots to 
path.plot <- "../plots/"
ggsave('bagged.png',scale=0.7,dpi=400, path= path.plot)
################ 3.2 #################
# ====================================

## Create CV control for model 
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"
## Tune grid to try each number of features used  
tunegrid <- expand.grid(.mtry=c(2,4,6,8))

## Create empty dataframe
rf_df =data.frame()
## Create model for each of the number of trees/ number of features
## Will return a dataframe with cv accuracy for each ntrees&nfeatures 
#*loop takes a long time

for (i in seq(from=25, to= 400, by=5)){
  rf <- train(factor(label)~ .-index, data=allSymbols, method="rf", 
              metric=metric, tuneGrid=tunegrid, ntree= i,trControl=control)
  ## Create dataframe of accuracy, ntrees, and nfeat
  df<-data.frame(Accuracy = rf$results['Accuracy'],
                 ntrees = c(i,i,i,i),
                 nfeatures = c(2,4,6,8))
  
  rf_df= rbind(rf_df, df)
  print(i)
}
# Nfeatures as factor 
rf_df$nfeatures <- as.factor(rf_df$nfeatures)


## Plot showing cv accuracy at each value of ntree for each nfeatures
rf<-ggplot(rf_df, aes(x= ntrees, y= Accuracy, group= nfeatures, color=nfeatures))+
  geom_point()+
  geom_smooth(se= FALSE)
rf
ggsave('rf.png',scale=0.7,dpi=400, path= path.plot)

## Highest accuracy 
rf_df[which.max(rf_df$Accuracy),]
