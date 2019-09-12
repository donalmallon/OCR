################ Start of Section 2 #################
# =================================================

#librarys 
library(ggplot2)
library(caret)
library(data.table)
library(class)
library(dplyr)

set.seed(3060)
################ 2.1 #################
# ====================================



## Retrive features data from downloaded file
## Bind all feature data into single dataframe 
#============================================

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

setwd("../section2_code/")

## Perform k-nearest-neighbour classification on training data
# =========================================================

## The training data is the first 10 features in allSymbols
## this is columns 3-12 

## *Each feature is scalled for KNN calulation 
train.tenFeatures <- data.frame(scale(allSymbols$nr_pix), 
                      scale(allSymbols$height),
                      scale(allSymbols$width),
                      scale(allSymbols$tallness),
                      scale(allSymbols$rows_with_1),
                      scale(allSymbols$cols_with_1),
                      scale(allSymbols$rows_with_5),
                      scale(allSymbols$cols_with_5),
                      scale(allSymbols$one_neigh),
                      scale(allSymbols$three_neigh))
                      
cols2 = c('nr_pix', 'height', 'width', 'tallness', 'rows_with_1', 'cols_with_1', 
          'rows_with_5', 'cols_with_5', 'one_neigh', 'three_neigh')
colnames(train.tenFeatures)= cols2

train.Symbol = allSymbols$label

## K values to be used 
k_values = c(1,3,5,7,8,11,15,19,23,31)
## Empty vecotr accuracies that will store accuracy for each k
accuracies = c()

## Perform k-nearest-neighbour classification on training data
for (k in k_values){
  knn.pred=knn(train.tenFeatures,train.tenFeatures,train.Symbol ,k=k)
  table(knn.pred, train.Symbol)
  accuracies = cbind(accuracies, mean(knn.pred==train.Symbol))
  print(k)
}

##create dataframe of accuracies to plot using ggplot2
accuracies = as.vector(accuracies)
accuracies_df = data.frame(k = k_values,
                           Accuracy = accuracies)

## Plot accuracies for each k value
plt.accuracies<- ggplot(accuracies_df, aes(x=k, y = Accuracy), color=("blue"))+
  geom_line()+theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
plt.accuracies

## Set a path to 'plots' folder to save all plots to 
path.plot <- "../plots/"
ggsave('plt_accuracy.png',scale=0.7,dpi=400, path= path.plot) 



################ 2.2 #################
# ====================================

## Define cv control
train_control <- trainControl(method  = "cv", number  = 5)

## Add symbol label to train.tenFeatures
train.tenFeatures$Symbol = train.Symbol

## K-nearest-neighbour classification using 5 fold crossvalidation
cv_10_feat <- train(factor(Symbol) ~ .,
                    method     = "knn",
                    tuneGrid   = expand.grid(k = k_values),
                    trControl  = train_control,
                    metric     = "Accuracy",
                    data       = train.tenFeatures)
cv_10_feat

## Create dataframe of CV accuracies to plot using ggplot2
cv_10_feat.accuracies<- cv_10_feat$results["Accuracy"]
cv_accuracies_df = data.frame(k = k_values,
                           Accuracy = cv_10_feat.accuracies)

## Plot accuracies for each k value
plt.accuracies<- ggplot(cv_accuracies_df, aes(x=k, y = Accuracy))+
  geom_line()+ theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
plt.accuracies
ggsave('plt_accuracy_cv.png',scale=0.7,dpi=400, path= path.plot) 

## Find error rates as 1-accuracies
crossValidatedError<-1- cv_10_feat.accuracies
names(crossValidatedError)<-"crossValidatedError"
trainingError = 1- accuracies

## Dataframe with the  two different accuaries and 1/k
inverse.k<- 1/k_values
error_df = data.frame(inverse_k =inverse.k,
                       trainingError =trainingError,
                       crossValidatedError = crossValidatedError)
## Plot error rate over training set against CV error rate
ErrorRatePlot<-ggplot(error_df, aes(inverse.k)) +
  geom_line(aes(y= trainingError),linetype = "dashed", color = "sky blue")+
  geom_point(aes(y= trainingError), color = "sky blue")+
  geom_line(aes(y= crossValidatedError), linetype = "dashed", color= "orange")+
  geom_point(aes(y= crossValidatedError), color= "orange")+
  xlab("1/K") + ylab("Error Rate")
ErrorRatePlot
ggsave('error_rate_plot.png',scale=0.7,dpi=400,path= path.plot)


################ 2.3 #################
# ====================================

##*Use k value of 15 as it produced the optimal model (largest Accuracy in CV)
train.tenFeatures<- subset(train.tenFeatures, select=-c(Symbol))

## Perform knn on training data with a k of 15 using first 10 features
knn.pred=knn(train.tenFeatures,train.tenFeatures,train.Symbol ,k=15)
knn_15_table = table(knn.pred, train.Symbol)
  
knn_15_df= as.data.frame(knn_15_table, stringsAsFactors = FALSE)

## Create separate data frames for symbol type to create three plots 

## Force symbol first to a numeric value 
knn_15_df$train.symbolLabel<- as.numeric(as.character(knn_15_df$train.Symbol))

## Replace the labels with the actual Symbols 
symbols = c(1,2,3,4,5,6,7,'a', 'b', 'c', 'd',
            'e', 'f', 'g', '<','>', '=', '≤',
            '≥', '≠', '≈')

input_symbols = rep(symbols,each = 21)
pred_symbols = rep(symbols, 21)
knn_15_df$train.Symbol = input_symbols
knn_15_df$knn.pred = pred_symbols

digit.knn_15_df <- knn_15_df[ which(knn_15_df$train.symbolLabel<19), ]
letter.knn_15_df <- knn_15_df[ which(knn_15_df$train.symbolLabel>19
                                & knn_15_df$train.symbolLabel<29), ]
math.knn_15_df <- knn_15_df[ which(knn_15_df$train.symbolLabel>29), ]


## Digit plot KNN k =15, remove lower 5% values ie. freq<21
plt.digxit.knn_15<- ggplot(digit.knn_15_df, aes(x = knn.pred, 
                                         y = Freq, colour = train.Symbol)) +
  geom_text(data = subset(digit.knn_15_df, Freq >=21),size= 3,
            aes( factor(knn.pred),y= Freq, label=factor(knn.pred)))+
  theme(legend.position="none",
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  facet_wrap( ~ train.Symbol)
ggsave('digit_knn15.png',scale=0.7,dpi=400,path= path.plot) 

## Letter plot KNN k =15, remove lower 5% values ie. freq<21
plt.letter.knn_15<- ggplot(letter.knn_15_df, aes(x = knn.pred, 
                                               y = Freq, colour = train.Symbol)) +
  geom_text(data = subset(letter.knn_15_df, Freq >=21),size= 3,
            aes( factor(knn.pred),y= Freq, label=factor(knn.pred)))+
  theme(legend.position="none",
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  facet_wrap( ~ train.Symbol)
ggsave('letter_knn15.png',scale=0.7,dpi=400, path= path.plot) 

## Math plot KNN k =15, remove lower 5% values ie. freq<21
plt.math.knn_15<- ggplot(math.knn_15_df, aes(x = knn.pred, size= 3,
                                               y = Freq, colour = train.Symbol)) +
  geom_text(data = subset(math.knn_15_df, Freq>=21),
            aes( factor(knn.pred),y= Freq, label=factor(knn.pred)))+
  theme(legend.position="none",
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  facet_wrap( ~ train.Symbol)

ggsave('math_knn15.png',scale=0.7,dpi=400,path= path.plot)


## Which digits are most confusable ?

## The pair of digits with the highest combined wrong predictions 

## It is apparent from graph that 3 &5 are most often confused 

################ End of Section 2 #################
# =================================================
  


