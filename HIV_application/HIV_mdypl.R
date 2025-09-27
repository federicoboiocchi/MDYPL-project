# Author: Federico Boiocchi

# Diaconis-Ylvisaker logistic regression on a binary dataset

#rm(list = ls())

library(tidyverse)

set.seed(123)

# setwd("C:\\Users\\andre\\Desktop\\MDYPL_R_warwick\\HIV_mdypl")

y_X <- read.csv("data_hiv")
dim(y_X) # it adds a column automatically to index rows

# we have to make the drug resistance response variable binary.
# we arboitrarily choose 30 as the boundary between low drug resistance level and
# high drug resistance.
lim <- 30 

y <- y_X$y
X <- y_X[, -c(1, 2)]

y_01 <- ifelse(y < lim, 0, 1)
# hist(y_01)

# Remark: we have unbalanced classes, we should take this into account when
# estimating the logistic regression

# Resampling to adjust for unbalanced classes:

size_adj <- sum(y_01 == 0) - sum(y_01 == 1)

yX <- as.data.frame(cbind(y_01, X))
yX_add <- slice_sample(yX[yX$y_01 == 1, ], n = size_adj)
yX_tot <- rbind(yX, yX_add)
N <- dim(yX_tot)[1]
shuff_ind <- sample(1:N, size = N, replace = FALSE) # default (shown to be clear)

# Binary response and sparse design matrix

yX_tot <- yX_tot[shuff_ind, ]
y <- as.data.frame(yX_tot[, 1]); dim(y);
X <- as.data.frame(yX_tot[, -1])

# some checks
dim(X);anyNA(y);anyNA(X)

# Importing the brglm2 library, needed to use the method "mdyplFit" for logistic regression

library(brglm2)

# splitting in training and test (50/50)
train_id <- sample(1:N, size = round(N / 2), replace = FALSE)

# train set
X_tr <- as.matrix(X[train_id,])
X_tr_std <- scale(X_tr,center=TRUE,scale=TRUE) # standardization
# checking for NAs
anyNA(X_tr_std)
image(t(is.na(X_tr_std)))
sum(is.na(X_tr_std))
# defining a function to give the position of NAs
whichNA <- function(X){
  n <- dim(X)[1]
  p <- dim(X)[2]
  pos <- matrix(data=NA,ncol=2,nrow=1)
  for(i in 1:n){
    for(j in 1:p){
      if(is.na(X[i,j])){
        pos <- rbind(c(i,j),pos)
      }   
    }   
  }
  return(pos)
}
# In the columns 30 and 167 we have NaN's, derinving from the operation of standardization. Therefore we remove them
# since the standardization will help later. Alternatively, one could have avoided to delete an entire mutation (feature)
# at the cost of not being able to standardize and having issues later with the hd_correction. 
whichNA(X_tr_std)
dim(X_tr_std)
X_tr_std[,30]
X_tr_std[,167]
X_tr <- X_tr_std[,-c(30,167)] 
anyNA(X_tr)

y_tr <- as.matrix(y[train_id,])
y_tr <- factor(y_tr) # response is a binary (categorical variable)

# Test set
X_ts <- as.matrix(X[-train_id,-c(30,167)])
y_ts <- as.matrix(y[-train_id,])
y_ts <- factor(y_ts)

X_ts_std <- scale(X_ts,center=TRUE,scale=TRUE)
anyNA(X_ts_std)
whichNA(X_ts_std)
X_ts <- X_ts_std[,-195] # index associated with NaN's
anyNA(X_ts)

# Standard logistic regression model

# we get warnings about convergence and probabilities fitted
# to 0 or 1. The IRLS inside the logistic regression fails to converge

mod <- glm(y_tr ~ X_tr, 
           family = binomial(link = "logit"))

# Penalized likelihood approach and high-dimensional correction (Kosmidis,Sterzinger 2024)
# DY penalized logistic regression model estimation:

mod_DY <- glm(y_tr ~ X_tr, 
    family = binomial(link = "logit"), 
    method = "mdyplFit")

# Estimated regression coefficients (beta_DY_hat) using the mdypl fitter:

betas <- coef(mod_DY) # it might contains NAs 
ind_NA <- as.numeric(which(is.na(betas)==TRUE))
betas <- betas[-ind_NA]

b_DY <- matrix(data=betas,ncol=1,nrow=length(betas)) 

n_tr <- dim(X_tr)[1]
ones <- matrix(data = rep(1,n_tr),ncol=1,nrow=n_tr)
X_tr <- X_tr[,-ind_NA]
eta_hat <- cbind(ones,X_tr)%*%b_DY  

# rowSums(X_tr) how many mutations for each patient

# Inverse logistic link function 

link <- function(x){
  1/(1+exp(-x))
}

# estimated probabilities on the training set 

probs_DY <- link(eta_hat)
probs_fit <- fitted(mod_DY)

# equivalently it could have been used probs_DY <- fitted(mod_DY), but we wanted to show the whole
# computation of the linear predictor and regression coefficients, and that even though we don't see it sometimes certain covariates are completely eliminated
# in the computation of the predicted probabilities. On the contrary, here, we have full control of what we are doing.

# Obviously, the goodness of the model must be assessed on the test set, namely, we use the DY regression coefficients estimated on the training set
# to do binary classification on the test set (we don't train the model on the test, but just evaluate it).

n_ts <- dim(X_ts)[1]
X_ts <- X_ts[,-ind_NA]
dim(X_ts)
eta_hat_test <- cbind(ones,X_ts)%*%b_DY[-197] 

probs_DY_ts <- link(eta_hat_test)
est_labels <- ifelse(probs_DY_ts>0.5,1,0)
ground_truth <- y_ts
# accuracy function
acc <- function(x,y){
  if(length(x)==length(y)){
    out <- sum(x==y)/length(x)
    return(out)
  } else {
    return("Different lengths not allowed")
  }
}

# accuracy on the test set
acc(est_labels,y_ts) # approx: 0.851

# High-Dimensionality correction:

# we avoid to use the summary below, in order to give a more detailed presentation 
# on how the hd_correction works.
# summary(mod_DY,hd_correction=TRUE)

k <- dim(X_tr)[2]/dim(X_tr)[1] # aspect ratio
alpha <- 1/(1+k) # hyperparameter controlling the variability of the Diaconis-Ylvisaker prior
ss <- sloe(mod_DY) # sloe estimator (it turns out not to work with this data), we unpack it to 
# discover the problem and fix it manually.

# Unpacking sloe function (Signal-Strength-Leave-One-Out Estimator)
# getting the function from in the R console
getAnywhere(sloe)
# recomputing the quantities
mu <- fitted(mod_DY)
v <- mu * (1 - mu)
h <- hatvalues(mod_DY)
S <- mod_DY$linear.predictors - (mod_DY$y_adj - mu)/v * (h/(1-h))
# (h/(1-h)) the ratio between hat values creates + and -Inf. Hence we restrict the values
# of signal strenghts that are allowed.
S_feasible <- S[abs(S)<Inf]  # we fix it by considering only finite s_i 
ss <- sd(S_feasible)
inter <- b_DY[1]

# we apply the function solve_se() (Author: Federico Boiocchi based on previous Julia code by Philipp Sterzinger), in order to solve 
# the AMP state evolution system and find the parameter mu_* used to correct the estimated regression coefficients. 

se_pars <- solve_se(kappa = k, ss = ss, alpha = alpha,
                        intercept = inter,
                        start = c(0.5,1,1,inter),
                        corrupted = FALSE, gh = NULL, prox_tol = 1e-10,
                        transform = TRUE, init_method = "Nelder-Mead",
                        init_iter = 10)

# correction parameter for b_DY
mu_star <- se_pars[1]

# corrected estimated using AMP theory
b_hd_corr <- b_DY[-197]/mu_star

eta_hat_corr <- cbind(ones,X_ts)%*%b_hd_corr 
probs_DY_corr <- link(eta_hat_corr)

est_labels_corr <- ifelse(probs_DY_corr>0.5,1,0)
acc(est_labels_corr,y_ts) 

# we are not able to appreciate the difference
# in terms of accuracy, since the accuracy doesn't take into account predicted probabilities
# but just labels.

# Predicted probabilities on the test set graph:

n_ts <- dim(X_ts)[1]

ind_0 <- which(y_ts==0)
n_0 <- length(ind_0)
ind_1 <- which(y_ts==1)
n_1 <- length(ind_1)
par(mgp = c(2.5, 0.7, 0))
par(mfrow=c(1,2))
plot(1:n_0,probs_DY_ts[ind_0],xlim=c(0,n_ts),
     pch="0",cex=0.5,main="Predicted probabilities (no hd_corr)",
     ylab=expression(hat(p)),xlab="statistical units index",cex.main=0.8)

points((n_0+1):(n_0+n_1),probs_DY_ts[ind_1],
       col="red",pch="1",cex=0.5)
abline(h=0.5,col="blue",lwd=1.2,lty="dashed")

# Plot with hd_correction

plot(1:n_0,probs_DY_corr[ind_0],xlim=c(0,n_ts),
     pch="0",cex=0.5,main="Predicted probabilities (with hd_corr)",
     ylab=expression(hat(p)),xlab="statistical units index",cex.main=0.8)

points((n_0+1):(n_0+n_1),probs_DY_corr[ind_1],
       col="red",pch="1",cex=0.5)
abline(h=0.5,col="blue",lwd=1.2,lty="dashed")

# plot availavle in this folder under the name of pred_prob_comp.jpeg

# Classifier relative entropy for logistic regression
# with hd_correction and withouth hd_correction

# predicted probability matrix
# each row is a statistical unit (patient) in the test set while
# each of the two columns represent the probability of the patient
# being in the class of high nelfinanvir resistance given the 
# linear predictor composed by mutations' occurrences. 

pp_no_hd <- cbind(probs_DY_ts,1-probs_DY_ts)
pp_hd <- cbind(probs_DY_corr,1-probs_DY_corr)

rEN<-function(pp){
  n <- dim(pp)[1]
  k <- dim(pp)[2]
  return(-sum(apply(pp,1,function(t) sum(t*log(t))))/(n*log(k)))
}

rEN(pp_no_hd) # we have 68% of the entropy we would have in the case of random allocations
# in the two groups (worst entropy level possible)
rEN(pp_hd) # we have 50% of the entropy

# a lower value of rEN means a better classification. 



