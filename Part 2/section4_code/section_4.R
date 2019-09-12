################ Start of Section 4 #################
# =================================================

#librarys 
library(ggplot2)
library(caret)
library(randomForest)
library(data.table)

set.seed(3060)
#===============================================================
########### Section 4.1 - Data Retrieval & Cleaning  ############
#===============================================================

#*Code below same as section 2&3 to retrieve test data 
#====================================================

## Temporarily set working directory to features files directory 
setwd("../training_data/")

## Get list of file names ending in features.csv
files = list.files(pattern="*features.csv")

## Load all the -features.csv files into a single DF using fread()
allSymbols = rbindlist(lapply(files, fread))

# Name columns of dataset 
cols = c('label', 'index', 'nr_pix', 'height', 'width', 'tallness',
         'rows_with_1', 'cols_with_1', 'rows_with_5', 'cols_with_5',
         'one_neigh', 'three_neigh', 'none_below', 'none_above',
         'none_before', 'none_after', 'nr_regions', 'nr_eyes',
         'r5_c5', 'bd_decimal', 'r_max_pixel', 'c_max_pixel')
colnames(allSymbols)= cols
#============================================


#========================================================
### Recreate custom features created in Assignment 1
##  and apply to training data
#========================================================

# Create a list of all test pixel info csv files
pixel.files<-list.files(pattern="*pixels.csv")
pixel.files

# Create a loop to read in all test pixel info into dataframe in separate lists
pixel.data<- list()
for (i in 1:length(pixel.files)){
  pixel.data[[i]]<-read.csv(pixel.files[i], header= FALSE, ",")
  
  ## Check progress of loop 
  if(i%%1000==0){
    print(i)
  }
}

## Return wd to src file location 
setwd("../section4_code/")

## First Custom Feature = bd_decimal
#===================================

## This would return the decimal value of the 
## number of pixels in first half/ total number of pixels
bd_decimal<- c()
for (n in 1:length(pixel.data)){
  bd_decimal= append(bd_decimal,sum(pixel.data[[n]][1:200]) /sum(pixel.data[[n]]))
}

## Second and Third Custom Feature- 
## Max pixels in a row & Max pixels in a column 
#================================================


## Create a list of 20*20 matricies for each image
## Find max nrpix per row/col
r_max_pixel<- c()
c_max_pixel<-c()
pixel.data.matrix<-list()
r_max_pixel_current= 0 
c_max_pixel_current= 0

for (n in 1:length(pixel.data)){
  pixel.data.matrix[[n]]<-matrix(unlist(t(pixel.data[[n]])), byrow=T, 20)
  
  ## Loop 1:20 to find current max for each 
  for (i in 1:20){
    if(sum(pixel.data.matrix[[n]][,i])>c_max_pixel_current)
      c_max_pixel_current = sum(pixel.data.matrix[[n]][,i])
    if(sum(pixel.data.matrix[[n]][i,])>r_max_pixel_current)
      r_max_pixel_current= sum(pixel.data.matrix[[n]][i,])
  }
  c_max_pixel = c(append(c_max_pixel, c_max_pixel_current))
  r_max_pixel = c(append(r_max_pixel, r_max_pixel_current))
  c_max_pixel_current=0
  r_max_pixel_current=0
}

## Add 3 custom features to current allSymbols training data

allSymbols$bd_decimal= bd_decimal
allSymbols$r_max_pixel = r_max_pixel
allSymbols$c_max_pixel = c_max_pixel

## Change list of symbols into a datframe called of training data
training.data<- data.frame(allSymbols)

## Set first two columns as factors 
training.data[,1:2]<-sapply(training.data[,1:2], as.factor)


#===============================================================
####################### Section 4.2 - PCA  ######################
#===============================================================

## First want to visualise PCA to choose k/ num components ##

## Standardize all feature variables 

scaled.training.data = scale(training.data[,3:22], 
                          center=TRUE,scale=TRUE)
summary(scaled.training.data)

## Perform PCA on scaled data
pca = prcomp(scaled.training.data)

# Calculate the variance explained by each component
# and express that as a proportion:

prop_var_explained = (pca$sdev)^2/sum((pca$sdev)^2)
prop_var_explained

##Create a dataframe with the cumulative sum of prop var explained
var_explained<-data.frame(prop_var_explained = cumsum(prop_var_explained),
                          num_comp=1:20)

## Plot cumulative some for each num companents 
## Assume cut off of 80% explained variance
plot.var<-ggplot(var_explained, aes(y=prop_var_explained, x=num_comp))+
  geom_point()+
  geom_line(alpha= 0.5)+
  geom_hline(yintercept=0.8, linetype="dashed", color = "red")+
  xlab("Number of Components")+
  ylab("Cumulative Proportion of Variance Explained")
plot.var
## Set a path to 'plots' folder to save all plots to 
path.plot <- "../plots/"
ggsave('var_explained.png',scale=0.7,dpi=400, path= path.plot)

#========================================================================
###### Section 4.3 - Random Forests tuning with Pre-processing  #########
#=======================================================================

## Begin by calculating the preprocessing parameters from the dataset 

prp<- preProcess(training.data, method=c("center", 'scale', 'pca'),
                         thresh = 0.8 ) 
## Transform training dataset using 
processed.training.data<-predict(prp, training.data)

## Create 5 fold cross-validation control for model 
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

## Tune grid to try each number of features used  
tunegrid <- expand.grid(.mtry=c(2,3,4,5))

## Create empty dataframe
final.rf_df =data.frame()

## Create model for each of the number of trees/ number of features
## Will return a dataframe with cv accuracy for each ntrees&nfeatures 


for (i in seq(from=100, to= 150, by=5)){
  rf <- train(label~ ., data=processed.training.data, 
              method="rf", metric=metric, tuneGrid=tunegrid, 
              ntree= i,trControl=control)
  
  ## Create dataframe of accuracy, ntrees, and nfeat
  df<-data.frame(Accuracy = rf$results['Accuracy'],
                 ntrees = c(i,i,i,i),
                 nfeatures = c(2,3,4,5))
  
  final.rf_df= rbind(final.rf_df, df)
  print(i)
}
final.rf_df

## Create a plot showing 
rf.plot<-ggplot(final.rf_df, aes(x= ntrees, y= Accuracy, group= nfeatures, 
                                 color=factor(nfeatures)))+
  geom_point()+
  geom_smooth(se= FALSE)
rf.plot
ggsave('rf.png',scale=0.7,dpi=400, path= path.plot)

## Highest accuracy !
final.rf_df[which.max(final.rf_df$Accuracy),]

#========================================================================
###### Section 4.4 - Use model to classify test data  #########
#=======================================================================

#==================================================
###########  Code copied from above   #############
## read in test data and apply custom features ##
#===================================================

## Temporarily set working directory to features files directory 
setwd("../test_data/")

## Get list of file names ending in features.csv
files = list.files(pattern="*features.csv")

## Load all the -features.csv files into a single list using fread()
test.allSymbols = rbindlist(lapply(files, fread))

# Name columns of dataset 
cols = c('label', 'index', 'nr_pix', 'height', 'width', 'tallness',
         'rows_with_1', 'cols_with_1', 'rows_with_5', 'cols_with_5',
         'one_neigh', 'three_neigh', 'none_below', 'none_above',
         'none_before', 'none_after', 'nr_regions', 'nr_eyes',
         'r5_c5', 'bd_decimal', 'r_max_pixel', 'c_max_pixel')
colnames(test.allSymbols)= cols
#============================================


#========================================================
### Recreate custom features created in Assignment 1
##  and apply to test data
#========================================================

# Create a list of all test pixel info csv files
pixel.files<-list.files(pattern="*pixels.csv")
pixel.files

# Create a loop to read in all test pixel info into dataframe in separate lists
pixel.data<- list()
for (i in 1:length(pixel.files)){
  pixel.data[[i]]<-read.csv(pixel.files[i], header= FALSE, ",")
  
  ## Check progress of loop 
  if(i%%1000==0){
    print(i)
  }
}

## Return wd to src file location 
setwd("../section4_code/")

## First Custom Feature = bd_decimal
#===================================

## This would return the decimal value of the 
## number of pixels in first half/ total number of pixels
bd_decimal<- c()
for (n in 1:length(pixel.data)){
  bd_decimal= append(bd_decimal,sum(pixel.data[[n]][1:200]) /sum(pixel.data[[n]]))
}

## Second and Third Custom Feature- 
## Max pixels in a row & Max pixels in a column 
#================================================


## Create a list of 20*20 matricies for each image
## Find max nrpix per row/col
r_max_pixel<- c()
c_max_pixel<-c()
pixel.data.matrix<-list()
r_max_pixel_current= 0 
c_max_pixel_current= 0

for (n in 1:length(pixel.data)){
  pixel.data.matrix[[n]]<-matrix(unlist(t(pixel.data[[n]])), byrow=T, 20)
  
  ## Loop 1:20 to find current max for each 
  for (i in 1:20){
    if(sum(pixel.data.matrix[[n]][,i])>c_max_pixel_current)
      c_max_pixel_current = sum(pixel.data.matrix[[n]][,i])
    if(sum(pixel.data.matrix[[n]][i,])>r_max_pixel_current)
      r_max_pixel_current= sum(pixel.data.matrix[[n]][i,])
  }
  c_max_pixel = c(append(c_max_pixel, c_max_pixel_current))
  r_max_pixel = c(append(r_max_pixel, r_max_pixel_current))
  c_max_pixel_current=0
  r_max_pixel_current=0
}

## Add 3 custom features to current allSymbols test data

test.allSymbols$bd_decimal= bd_decimal
test.allSymbols$r_max_pixel = r_max_pixel
test.allSymbols$c_max_pixel = c_max_pixel

## Change list of symbols into a datframe called of test data
test.data<- data.frame(test.allSymbols)


#===================================
### Test data retrieval Complete  #####
#===================================

## Preproccess data using parameters as before
processed.test.data<-predict(prp, test.data)

#===================================
########### Final Model  ###########
#===================================

## Highest cv accuracy on training data ntrees=145 & nfeatures= 4

finalModel<- randomForest(factor(label) ~ ., data =processed.training.data, 
                          ntree = 145, mtry = 4)

## Use model to predict label of test data 

test.data$prediction = predict(finalModel, processed.test.data , type ="class")

index_prediction<- data.frame(index=test.data$index, 
                              prediction = test.data$prediction )
## Pad index with 0s
index_prediction$index<-sprintf("%03d", as.numeric(index_prediction$index))

## Get data in correct format for csv
final.predictions<-paste(index_prediction$index,index_prediction$prediction, sep= "," )

## Save index and prediction to csv
setwd("..")
dir()
write.table(final.predictions, file = "40154387_section4_predictions.csv",
          row.names = FALSE, col.names = FALSE, sep=",", quote = TRUE)
setwd("./section4_code/")
?write.table
