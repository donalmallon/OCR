################ Start of Section 1 #################
# =================================================

#librarys 
library(ggplot2)
library(caret)

#set seed
set.seed(3060)

############################### 1.1 ###################################
# =====================================================================

## Change working direction to location of features.csv
setwd("../40154387_features/")
dir()
##read in features csv 
allSymbols <- read.csv(file="40154387_features.csv", header=TRUE, sep=",")
head(allSymbols)

## Return wd to source file location 
setwd("../section1_code/")


## Drop the extra index column
allSymbols = subset(allSymbols, select = -c(X) )

## Subset of dataframe with only digits and letters 
digit_letter_df <- allSymbols[ which(allSymbols$label<29), ]

## Dummy variable called type to discriminate symbol type
# 0 for digit and 1 for letter 

digit_letter_df$type <-0
digit_letter_df$type[digit_letter_df$label>19]<-1

## Rows_with_1 is most significant differentiator between letters and numbers 

## Plot hist showing count for rows with 1 to see how letter and numbers 
plt <- ggplot(digit_letter_df, aes(x=rows_with_1, fill=as.factor(type))) +
  geom_histogram(binwidth=1, alpha=.5, position='identity')+
  guides(fill=guide_legend(title="Digit/Letter"))
plt 

## Set a path to 'plots' folder to save all plots to 
path.plot <- "../plots/"
ggsave('rows_with_1_hist.png',scale=0.7,dpi=400, path= path.plot)   

## Digit/letter logistic regression with rows_with_1 

fit<-glm(type ~ rows_with_1, 
            data = digit_letter_df, 
            family = 'binomial') 
summary(fit)


## Plot fitted curve
#====================

x.range = range(digit_letter_df[["rows_with_1"]])

x.values = seq(x.range[1],x.range[2],length.out=1000)

fitted.curve <- data.frame(rows_with_1 = x.values)
fitted.curve[["type"]] = predict(fit, fitted.curve, type="response")

# Plot the data and the fitted  curve:
plt <-ggplot(digit_letter_df, aes(x=rows_with_1, y=type)) + 
  geom_point(aes(colour = factor(type)), 
             show.legend = T, position="dodge")+
  geom_line(y=0.5)+
  geom_line(data=fitted.curve, colour="orange", size=1)

plt

ggsave('glm_fitted_curve.png',scale=0.7,dpi=400, path= path.plot)   

## Calculating accuracy of the fit (assuming p>0.5 cut off)
# ===========================================

remove(x.values)
x.values = digit_letter_df[["rows_with_1"]]

digit_letter_df[["predicted_val"]] = predict(fit, digit_letter_df, type="response")
digit_letter_df[["predicted_class"]] = 0
digit_letter_df[["predicted_class"]][digit_letter_df[["predicted_val"]] > 0.5] = 1

correct_items = digit_letter_df[["predicted_class"]] == digit_letter_df[["type"]] 
# proportion correct:
rows_with_1_acc = nrow(digit_letter_df[correct_items,])/nrow(digit_letter_df)
rows_with_1_acc
# proportion incorrect:
nrow(digit_letter_df[!correct_items,])/nrow(digit_letter_df)

## Table to show the exact num times each symbols predicted correct/incorrect
table(pred = digit_letter_df$predicted_class, input = digit_letter_df$type)


############################### 1.2 ###################################
# =====================================================================

## Create a logistic regression fit for the first 8 features
colnames(allSymbols)
fit8features<-glm(type ~nr_pix+height+width+tallness+rows_with_1+cols_with_1+rows_with_5plus+cols_with_5plus , 
         data = digit_letter_df, 
         family = 'binomial') 
summary(fit8features)


## Calculating accuracy of the fit8features (assuming p>0.5 cut off)

remove(x.values)
x.values = digit_letter_df[["nr_pix"]]+digit_letter_df[["height"]]+digit_letter_df[["width"]]+digit_letter_df[["tallness"]]+digit_letter_df[["rows_with_1"]]+digit_letter_df[["cols_with_1"]]+digit_letter_df[["rows_with_5plus"]]+digit_letter_df[["cols_with_5plus"]]

digit_letter_df[["predicted_val"]] = predict(fit8features, digit_letter_df, type="response")
digit_letter_df[["predicted_class_fit8features"]] = 0
digit_letter_df[["predicted_class_fit8features"]][digit_letter_df[["predicted_val"]] > 0.5] = 1

correct_items = digit_letter_df[["predicted_class_fit8features"]] == digit_letter_df[["type"]] 

# proportion correct:
fit8features_acc = nrow(digit_letter_df[correct_items,])/nrow(digit_letter_df)
fit8features_acc
# proportion incorrect:
nrow(digit_letter_df[!correct_items,])/nrow(digit_letter_df)

## Table to show the exact num times each symbols predicted correct/incorrect by model
table(pred = digit_letter_df$predicted_class_fit8features, input = digit_letter_df$type)

############################### 1.3 ###################################
# =====================================================================

 
## CV model for fit rows_with_1
# =============================

train_control=trainControl(method = "cv", number = 7)
# train the model
cv_model1_1 <- train(as.factor(type) ~ as.factor(rows_with_1), 
               data = digit_letter_df, method="glm", 
               family= 'binomial',
               trControl = train_control)
cv_model1_1_acc<-as.numeric(cv_model1_1$results['Accuracy'])
# Accuracy of model 0.7857143

## CV model for fit8features
# ===========================

train_control=trainControl(method = "cv", number = 7)
# train the model
cv_model1_2 <- train(as.factor(type) ~nr_pix+height+width+tallness+rows_with_1+cols_with_1+rows_with_5plus+cols_with_5plus, 
               data = digit_letter_df, method="glm", 
               family= 'binomial',
               trControl = train_control)
# results
cv_model1_2_acc<-as.numeric(cv_model1_2$results['Accuracy'])
## Accuracy of model 0.6964286



############################### 1.4 ###################################
# =====================================================================

## A model that predicts digit everytime will have an accuracy of 50%
##a model that predicts 50% digit and 50% letter with have an average accuracy of 50%

##Check to see if accuaracy of four models are 
##significantly greater than 0.5 using binomial distribution

## H0 for each model = Accuracy of model is equal to 0.5
## HA for each model = Accuracy of model is greater than 0.5 
## at a confidence interval of 95%

size = nrow(digit_letter_df)

p.rows_with_1<-1-pbinom(rows_with_1_acc*size, size, 0.5)
p.fit8features<-1-pbinom(fit8features_acc*size, size, 0.5)
p.cv_model1_1<-1-pbinom(cv_model1_1_acc*size, size, 0.5)
p.cv_model1_2<-1-pbinom(cv_model1_2_acc*size, size, 0.5)



############################### 1.5 ###################################
# =====================================================================
digit_letter_df$predicted_class_fit8features

## Split digit_letter_df into digit and letter 
digit_df <- digit_letter_df[ which(digit_letter_df$label<19), ]
letter_df<- digit_letter_df[ which(digit_letter_df$label>19), ]

## Plot a graph of count of incorrect classifications for each DIGIT 

plt.digits <-ggplot(digit_df, aes(x=label-10, y = predicted_class_fit8features,
                                  fill= factor(label))) + 
  geom_bar(stat= "identity")+
  theme_bw()+
  theme(legend.position="none")+
  scale_x_continuous(breaks = seq(0, 8, 1), lim = c(0, 8))+ 
  theme(plot.title = element_text(hjust = 0.5))+
  ggtitle("Incorrect Classification Count \nDigits")+
  labs(x= "Digits", y = "Number of Incorrect Classifications")
plt.digits

ggsave('plt_digits.png',scale=0.7,dpi=400, path = path.plot)


## Plot a graph of count of incorrect classifications for each LETTER 

plt.letters <-ggplot(letter_df, aes(x=label-20, y = 1-predicted_class_fit8features,
                                    fill= factor(label))) + 
  geom_bar(stat= "identity")+
  scale_x_discrete(limits=1:7, labels = letters[1:7])+ theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position="none")+
  ggtitle("Incorrect Classification Count \nLetters")+
  labs(x= "Letters", y = "Number of Incorrect Classifications")
plt.letters

ggsave('plt_letters.png',scale=0.7,dpi=400, path = path.plot)


################ End of Section 1 #################
# =================================================

