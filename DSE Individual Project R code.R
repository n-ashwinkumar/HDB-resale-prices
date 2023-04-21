#DSE1101 Individual Project



rm(list = ls())  # Clear everything to start clean

# Read in the data set
install.packages("readxl")
library(readxl)
data = read_excel("C:/Users/nashwinkumar/OneDrive - National University of Singapore/Desktop/DSE1101/HDB_resale_prices/HDB_data_2021_sample.xlsx")
df = data

# Split data set into train and test set
set.seed(678349) 
ntrain=4000
tr = sample(1:nrow(df),ntrain)  
train = df[tr,]   
test = df[-tr,] 

# Correlation matrix of key predictors
install.packages('corrplot')
library(corrplot)
corrplot(cor(train[,c('resale_price','floor_area_sqm','Remaining_lease','max_floor_lvl','Dist_CBD','Dist_nearest_station','mature')]),method = 'number',type = 'lower')

# Box plots of resale price against remaining lease and floor area
boxplot(resale_price/1000~Remaining_lease,data=train,
        xlab = "Remaining Lease (in years)", ylab = "HDB Resale Prices",notch=FALSE,varwidth=FALSE,pch=19)

boxplot(resale_price/1000~floor_area_sqm,data=train,
        xlab = "Floor Area (in square metres)", ylab = "HDB Resale Prices",notch=FALSE,varwidth=FALSE,pch=19)

####################################
## Simple linear regression model ##
####################################

# We have used the floor area as the predictor here due to its highest pairwise correlation with resale prices
lm1 = lm(resale_price/1000~floor_area_sqm, data = train)
summary(lm1)

# Scatter plot with fitted line
plot(x = train$floor_area_sqm, y = train$resale_price/1000,
     xlab = "Floor Area (in square metres)", ylab = "HDB Resale Prices", col = "red", pch = 20)
abline(lm1, col = "blue", lwd =2)

# Prediction on test data
predlm=predict.lm(lm1, newdata=test)

# Evaluation of model's performance
mean((test$resale_price/1000-predlm)^2) # MSE = 16309.84, RMSE = 127.71

######################################
## Multiple linear regression model ##
######################################

# Here we have choosed 6 key predictors for this model which we believe provide the strongest signals for the resale prices
lm2 = lm(resale_price/1000~floor_area_sqm+Remaining_lease+max_floor_lvl+Dist_CBD+Dist_nearest_station+mature, data = train)
summary(lm2)

# Prediction on test data
predlm=predict.lm(lm2, newdata=test)

# Evaluation of model's performance
mean((test$resale_price/1000-predlm)^2) # MSE = 4998.726, RMSE = 70.70167

##########################################
## K-nearest neighbors regression model ##
##########################################

library(kknn)

knncvall=train.kknn(resale_price/1000~.,data=train,kmax=100, kernel = "gaussian")

# Find the best K:
kbestall=knncvall$best.parameters$k
plot((1:100),knncvall$MEAN.SQU, type="l", col = "blue", main="LOOCV MSE", xlab="Complexity: K", ylab="MSE")
abline(v=2)

# We find K=2 works best according to LOOCV

# Fit for the selected K:
knnall = kknn(resale_price/1000~.,train,test,k=kbestall,kernel = "gaussian")

# Evaluation of model's performance
knnmseall=mean((test$resale_price/1000-knnall$fitted.values)^2) # MSE = 5331.827, RMSE = 73.01936

###########################
## Tree regression model ##
###########################

library(rpart)

# We first form a big enough tree before pruning it down so as to not miss out on important splits
big.tree = rpart(resale_price/1000~.,method="anova",data=train, minsplit=5,cp=.0005)

# Size of current tree
length(unique(big.tree$where)) # 99 leaves
nrow(big.tree$frame) # 197 nodes

# We use cv to obtain the optimal cp value that determines the size of the tree 
# which minimises loss function
plotcp(big.tree)
bestcpbigtree=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]

# We prune the original tree to the cv optimised tree
best.tree = prune(big.tree,cp=bestcpbigtree) 

# Size of pruned tree
length(unique(best.tree$where)) # 97 leaves
nrow(best.tree$frame) # 193 nodes

# Plot of pruned tree
plot(best.tree,uniform=FALSE) 
text(best.tree,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue')

# Prediction on test data
besttreefit=predict(best.tree,newdata=test,type="vector")

# Evaluation of model's performance
mean((test$resale_price/1000-besttreefit)^2) # MSE = 3336.901, RMSE = 57.76592

##########################################
## Principal component regression model ##
##########################################

library(pls)

# First, we do a PCA on the key predictors we have selected from earlier
prall = prcomp(train[,c('floor_area_sqm','Remaining_lease','max_floor_lvl','Dist_CBD','Dist_nearest_station','mature')], scale = TRUE)

# Next, we make a scree plot to get an idea of variance shares explained by components
prall.s = summary(prall)
prall.s$importance
scree = prall.s$importance[2,] 
plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b', cex = .8)

# Perform PCR
pcr.fit=pcr(train$resale_price/1000~floor_area_sqm+Remaining_lease+max_floor_lvl+Dist_CBD+Dist_nearest_station+mature,data=train, scale=TRUE, validation="CV")

# We can determine the optimal no. of components to use based on cross validation
validationplot(pcr.fit, val.type="MSEP", main="CV",legendpos = "topright")

# While the validation plot shows that using all 6 components would produce
# the minimum relative error, the scree plot helps us here to show that
# the elbow occurs at M = 4 thus we can go with 4 components here to
# demonstrate the application of PCR in dimension reduction.

# Prediction on test data
pcr.pred=predict(pcr.fit, newdata=test, ncomp=4)

# Evaluation of model's performance
mean((test$resale_price/1000-pcr.pred)^2) # MSE = 5483.754, RMSE = 74.05237

