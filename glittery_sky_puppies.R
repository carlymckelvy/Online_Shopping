library(pacman)
p_load(tidyverse,janitor, caret, glmnet, magrittr, janitor,rpart.plot,gbm,
       ggplot2,ggpubr, viridis, reshape2,ranger, 
       pROC, plotROC, dplyr, RColorBrewer,formattable,fastDummies,
       psych)

training_sample <-  read.csv("~/Downloads/training_sample.csv")
training_sample %<>% clean_names()

training_sample %>% describe()
## Step 1: Clean the dataset 
summary(training_sample)
anyNA(training_sample)

## Dummy the variables 
total_df <- subset(training_sample %>%select(-ordered))
total_df <- subset(total_df %>%select(-user_id))
clean_total <- total_df %>% select_if(is.numeric)

## Divide the dataset to train and test dataset
clean_total$ordered <- training_sample$ordered
is.numeric(clean_total$ordered)
clean_total$ordered <- factor(clean_total$ordered, labels = c("N", "Y"))
smp_size <- floor(0.5 * nrow(clean_total))
set.seed(123)
train_ind <- sample(seq_len(nrow(clean_total)), size = smp_size)
train <- clean_total[train_ind, ]
test <- clean_total[-train_ind, ]



## Step 2: Choose different models to work with 
## Model 1: Random Forest
seed <- set.seed(1998)
random_forest <- train(
  ordered ~ ., 
  data = train,
  method = "ranger",
  num.trees = 20,
  trControl = trainControl(method = "cv", 
                           number = 5, 
                           savePredictions = TRUE,
                           classProbs = T, 
                           summaryFunction = twoClassSummary,
                           seeds= seed
  ),     ## the method for cross validation: 5-fold cross validation  
  tuneGrid = expand.grid(
    'mtry' = 2:10,                ## the tuned parameter: the number of randomly selected predictors 
    'splitrule' = 'gini',
    'min.node.size' = 1:20        ## the tuned parameter: the minimal node size 
  ),
  importance = "permutation"
)

## Make prediction for Model 1: Random Forest
randomforest <- predict(
  random_forest,
  newdata = test %>% select(-ordered)
)


## Check the error rate for Model 1: Random Forest
rf_result <- data.frame( class = test$ordered, prediction = randomforest)
rf_result$diff <- ifelse(rf_result[,1]==rf_result[,2],0,1)
mean(rf_result$diff)


## Check the ROC and AUC
graph <- ggplot(random_forest$pred, aes(m=Y, d = factor(obs, levels =c("N","Y") )))+
  geom_roc(hjust=-0.6, vjust=2.5, color = 'seagreen4')+coord_equal()+
  geom_abline(slope=1, intercept=0)
graph
graph_new <- graph + annotate('text', x=0.75, y = 0.25, label = paste('AUC =',
                                                                      calc_auc(graph)$AUC))+ 
  ggtitle('ROC and AUC for Random Forest')+ 
  labs (x = "False Positive Rate", y = "True Positive Rate")+
  theme(plot.title = element_text(hjust = 0.5))

graph_new

## Check the importance of variables 
Imp <- varImp(random_forest)
rf_Imp <- data.frame(variable = names(train %>% select (-ordered)), 
                     overall = Imp$importance$Overall)

rf_Imp <- rf_Imp[order(rf_Imp$overall, decreasing = TRUE),]
rf_Imp_head <- head(rf_Imp,5)

rf_Imp_graph <- ggplot(rf_Imp_head, aes(variable, y = overall,fill = overall))+coord_flip()+
  geom_col()+
  theme_light()+
  scale_fill_distiller(palette = 'Paired')+
  ggtitle("Variables'Importance in Random Forest")+
  theme(plot.title = element_text(hjust = 0.5))

rf_Imp_graph

## Check the confusion matrix 
random_forest_confusion <- confusionMatrix(
  data = randomforest %>% as.factor(),
  reference = test$ordered %>% as.factor()
)

random_forest_confusion

## Plot for Confusion Matrix
rf_table <- as.data.frame(random_forest_confusion$table)

rf_plotTable <- rf_table %>%
  mutate(goodbad = ifelse(rf_table$Prediction == rf_table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

## fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups 
rf_confusion_graph <- ggplot(data = rf_plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "darkgreen", bad = "lightcoral")) +
  theme_bw() +
  xlim(rev(levels(rf_table$Reference)))+labs(x= "Truth")+
  ggtitle ( "Random Forest")+
  theme(plot.title = element_text(hjust = 0.5))

rf_confusion_graph

## Model 2: Boosting
boosting <- train(
  ordered ~.,
  data = train,
  method = 'gbm',
  trControl = trainControl(
    method = "cv",
    number = 5,
    savePredictions = TRUE,
    classProbs = T, 
    summaryFunction = twoClassSummary,
    seeds= seed
  ),                                    ## the method for cross validation: 5-fold cross validation 
  tuneGrid = expand.grid(
    "n.trees" = seq(25,200, by = 25),   ## the tuned paramater: the number of trees 
    'interaction.depth'=1:3,            ## the tuned parameter: the number of splits 
    'shrinkage'= c(.1,0.01,0.001),      ## the tuned parameter: the learning rate 
    'n.minobsinnode'=10
  )
)

## Make prediction for Model 2: Boosting
boosting_results <- predict(
  boosting,
  newdata = test %>% select(-ordered)
)


## Check the error rate for Model 2: Boosting
boostingresult <- data.frame(class = test$ordered, prediction = boosting_results)
boostingresult$diff <- ifelse(boostingresult[,1]==boostingresult[,2],0,1)
mean(boostingresult$diff)

## Check the ROC and AUC 
graph_1<- ggplot(boosting$pred, aes(m=Y, d = factor(obs, levels =c("N","Y") )))+
  geom_roc(hjust=-0.6, vjust=2.5, color = 'sandybrown')+coord_equal()+
  geom_abline(slope=1, intercept=0)
graph_1_new <- graph_1+ annotate('text', x=0.75, y = 0.25, label = paste('AUC =',
                                                                         calc_auc(graph_1)$AUC))+ 
  ggtitle('ROC and AUC for Boosting')+ 
  labs (x = "False Positive Rate", y = "True Positive Rate")+
  theme(plot.title = element_text(hjust = 0.5))

graph_1_new


## Check the importance of variables 
Imp_boosting <- varImp(boosting)
boosting_Imp <- data.frame(variable = names(train %>% select (-ordered)), 
                           overall = Imp_boosting$importance$Overall)

boosting_Imp <- boosting_Imp[order(boosting_Imp$overall, decreasing = TRUE),]
boosting_Imp_head <- head(boosting_Imp,5)

boosting_Imp_graph <- ggplot(boosting_Imp_head, aes(variable, y = overall,fill = overall))+coord_flip()+
  geom_col()+
  theme_light()+scale_fill_distiller(palette = 'Set3')+
  ggtitle("Variables'Importance in Boosting")+
  theme(plot.title = element_text(hjust = 0.5))

boosting_Imp_graph 

## Check the confusion matrix 
boosting_confusion <- confusionMatrix(
  data = boosting_results %>% as.factor(),
  reference = test$ordered %>% as.factor()
)

boosting_confusion

## Plot the confusion matrix 
boost_table <- as.data.frame(boosting_confusion$table)

boost_plotTable <- boost_table %>%
  mutate(goodbad = ifelse(boost_table$Prediction == boost_table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups 
boosting_confusion_graph <- ggplot(data = boost_plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "goldenrod1", bad = "midnightblue")) +
  theme_bw() +
  xlim(rev(levels(boost_table$Reference)))+
  labs(x= "Truth")+
  ggtitle ( "Boosting")+
  theme(plot.title = element_text(hjust = 0.5))

boosting_confusion_graph 


## Model 3: Elasticnet and logistic regression
lambdas <- 10^seq(from = 5, to = -2, length = 100)
alphas <- seq(from = 0, to = 1, by = 0.05)
net_cv <-  train(
  ordered ~.,
  data = train,
  method = "glmnet",
  family = "binomial",
  trControl = trainControl("cv", 
                           number=5,
                           savePredictions = TRUE,
                           classProbs = T, 
                           summaryFunction = twoClassSummary,
                           seeds= seed),     ## the method for cross validation: 5-fold cross validation 
  tuneGrid = expand.grid(alpha = alphas, lambda = lambdas)
)

## Make a the prediction for Model 4: Elasticnet and logistic regression
net_results <- predict(
  net_cv,
  newdata = test %>% select(-ordered)
)

## Check the model performance 
net_cv$results


## Make the error rate for Model 4: Elasticnet and logistic regression
net_diff <- data.frame(class = test$ordered, prediction= net_results)
net_diff$diff <- ifelse(net_diff[,1]==net_diff[,2],0,1)
mean(net_diff$diff)


graph_3<- ggplot(net_cv$pred, aes(m=Y, d = factor(obs, levels =c("N","Y") )))+
  geom_roc(hjust=-0.6, vjust=2.5, color = 'turquoise4')+coord_equal()+
  geom_abline(slope=1, intercept=0)

graph_3_new <- graph_3+ annotate('text', x=0.75, y = 0.25, label = paste('AUC =',
                                                                         calc_auc(graph_3)$AUC))+
  ggtitle('ROC and AUC for Elasticnet&Logistic Regression')+ 
  labs (x = "False Positive Rate", y = "True Positive Rate")+
  theme(plot.title = element_text(hjust = 0.5))

graph_3_new 

## Check the importance of variables 
Imp_net <- varImp(net_cv)
net_Imp <- data.frame(variable = names(train %>% select (-ordered)), 
                      overall = Imp_net$importance$Overall)

net_Imp <- net_Imp[order(net_Imp$overall, decreasing = TRUE),]
net_Imp_head <- head(net_Imp,5)

net_Imp_graph <- ggplot(net_Imp_head, aes(variable, y = overall,fill = overall))+coord_flip()+
  geom_col()+
  theme_light()+scale_fill_distiller(palette = 'Set3')+
  ggtitle("Variables' Importance in Elasticnet&Logistic Regression")+
  theme(plot.title = element_text(hjust = 0.5))

net_Imp_graph

## Check the confusion matrix 
net_confusion <- confusionMatrix(
  data = net_results %>% as.factor(),
  reference = test$ordered %>% as.factor()
)

net_confusion

## Plot the confusion matrix 
net_table <- as.data.frame(net_confusion$table)

net_plotTable <- net_table %>%
  mutate(goodbad = ifelse(net_table$Prediction == net_table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups 
net_confusion_graph <- ggplot(data = net_plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "steelblue", bad = "maroon")) +
  theme_bw() +
  xlim(rev(levels(net_table$Reference)))+labs(x= "Truth")+
  ggtitle ( "Elastinet&Logistic Regression")+
  theme(plot.title = element_text(hjust = 0.5))

net_confusion_graph

## Final Step: Report the final results for four models together
## Make the table to report the final results
model <- c('Random Forest', 'Boosting', 'Elasticnet and logistic regression')
Accuracy <- c(0.9929,0.993,0.993)
Sensitivity <- c (0.9935,0.9932,0.9934)
Specificity <- c(0.9796,0.9898,0.9832)
AUC <- c(0.9943,0.9902,0.7037)

Most_Importance_Variables <- c('saw_checkout','checked_delivery_detail',"checked_delivery_detail")

Final_result <- data.frame(model = model, Accuracy=Accuracy, Sensitivity=Sensitivity,
                           Specificity = Specificity, AUC = AUC, 
                           Most_Importance_Variables = Most_Importance_Variables)

formattable(Final_result, align =c("l","c","c","c","c", "c", "c"), list(
  `Moldel` = formatter("span", style = ~ style(color = "grey",font.weight = "bold")), 
  `Accuracy`= color_tile("lightskyblue", "lightsteelblue1"),
  `Sensitivity`= color_tile("#71CA97", '#DeF7E9'),
  `Specificity`= color_tile("lightskyblue", "lightsteelblue1"),
  `AUC`= color_tile("#71CA97", '#DeF7E9'),
  `Most_Importance_Variables` = color_tile("lightskyblue", "lightsteelblue1")))




## Final Confusion Matrix results for four models 
ggarrange(rf_confusion_graph, boosting_confusion_graph,
          net_confusion_graph+ rremove("x.text"), 
          labels = c("A", "B", "C"),
          ncol = 1, nrow = 3) 


## Final ROC and AUC results for four models 
ggarrange(graph_new, graph_1_new,graph_3_new + rremove("x.text"), 
          labels = c("A", "B", "C"),
          ncol = 1, nrow = 3) 

## Final Importance of variables results for four models 
ggarrange(rf_Imp_graph, boosting_Imp_graph, net_Imp_graph + rremove("x.text"), 
          labels = c("A", "B", "C"),
          ncol = 1, nrow = 3)



