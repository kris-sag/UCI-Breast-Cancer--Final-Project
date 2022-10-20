rm(list = ls()) # clear environment
# exploratory analysis

library(dplyr)
library(magrittr)
library(ggplot2)

cancer = read.csv("wdbc.data",header=FALSE)
names = c("ID_Number", "Diagnosis", 
    "Mean_Radius", "Mean_Texture", "Mean_Permieter", "Mean_Area", "Mean_Smoothness", "Mean_Compactness", "Mean_Concavity", "Mean_Concave_Points", "Mean_Symmetry", "Mean_Fractal_Dimension",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE", "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "Symmetry_SE", "Fractal_Dimension_SE",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness", "Worst_Compactness", "Worst_Concavity", "Worst_Concave_Points", "Worst_Symmetry", "Worst_Fractal_Dimension")
colnames(cancer) <- names

cancer[1] <- NULL # remove ID_Number as it is irrelevant

cancer %>% group_by(Diagnosis)  %>% summarize(n(),
    mean(Mean_Radius), 
    mean(Mean_Texture),
    mean(Mean_Permieter),
    mean(Mean_Area),
    mean(Mean_Smoothness),
    mean(Mean_Compactness),
    mean(Mean_Concavity),
    mean(Mean_Concave_Points),
    mean(Mean_Symmetry),
    mean(Mean_Fractal_Dimension))

cancer %>% group_by(Diagnosis) %>% summarize(n(),
    mean(Radius_SE),
    mean(Texture_SE),
    mean(Perimeter_SE),
    mean(Area_SE),
    mean(Smoothness_SE),
    mean(Compactness_SE),
    mean(Concavity_SE),
    mean(Concave_Points_SE),
    mean(Symmetry_SE),
    mean(Fractal_Dimension_SE))


malignant = cancer %>% filter(Diagnosis == "M")
benign = cancer %>% filter(Diagnosis == "B")

## uncomment to see boxplots
# par(mfrow=c(2,2))
# boxplot(malignant$Mean_Radius)
# boxplot(benign$Mean_Radius)

# # par(mfrow=c(1,2))
# boxplot(malignant$Mean_Fractal_Dimension)
# boxplot(benign$Mean_Fractal_Dimension)

# ---- FUNCTION CREATION ----
## RIDGE REGULARIZED LOGISTIC REGRESSION
objectiveValue <- function(X, Y, beta, lambda){

    obj = sum(-Y * (X %*% beta) + log(1 + exp(X %*% beta)) +
        lambda * sqrt(sum(beta * beta)))

    return(obj)

}

logisticReg_gradientDescent <- function(X, Y, lambda) {
    n = nrow(X)
    p = ncol(X)

    beta_prev = rep(0, p)
    maxiter = 100000

    stepsize = .01/n

    for (t in 1:maxiter){
        xb = X %*% beta_prev

        gradient = - t(X) %*% (Y - exp(xb)/(1 + exp(xb)) ) + lambda * 2 * beta_prev

        beta_next = beta_prev - stepsize * gradient


        if (norm(gradient, '2') < 1e-4) {
            print(paste("Converged at iteration: ", t))
            break
        } else beta_prev = beta_next

        cur_obj = objectiveValue(X, Y, beta_next, lambda)

        if (t %% 5000 == 0)
            print(paste("Current iter: ", t, " Current objective value: ", cur_obj))

        if (t == maxiter){
            print("Reached maxiter; did not converge.")
        }
    }

    print(paste("Final gradient norm: ", norm(gradient, '2')))
    return(beta_next)
}


# ----- logistic regression analysis -----
n = nrow(cancer)
cancer = cancer %>% mutate(Diagnosis = ifelse(Diagnosis == "M", 1, 0),
    .keep = "unused", .before = 1) # set malignant = 1, benign = 0

features = names[-(1:2)]

Y = cancer$Diagnosis
X = cbind(rep(1,n), as.matrix(cancer[, features]))
p = ncol(X)

# standardize
for (j in 2:p){
    X[, j] = (X[, j] - mean(X[, j]))/sd(X[, j])
}

## uncomment to reorder the data
# set.seed(1)

# reorder = sample(1:n, n, replace=FALSE)

# X = X[reorder, ]
# Y = Y[reorder]

n_test = 69
n_learn = 500

Y_test = Y[1:n_test]
Y_learn = Y[(n_test+1):(n_test+n_learn)]

X_test = X[1:n_test, ]
X_learn = X[(n_test+1):(n_test+n_learn), ]

# ## Using our functions to compute logistic regression

beta_lambda = logisticReg_gradientDescent(X_learn, Y_learn, 10)


## Making predictions on test data and computing the
## misclassification error

Ypred = (X_test %*% beta_lambda > 0)

print("Logistic Regression Test Error:")
lr_TE = mean( Y_test != Ypred )
print(lr_TE)

print("Logistic Regression Coefficients:")
print(beta_lambda)

# ----- LASSO analysis with glmnet package -----
library(glmnet)
## automatic cross-validation from glmnet package
cv_result = cv.glmnet(x = X_learn, y = Y_learn, alpha = 1,
    family = "binomial")

## lambda that attains smallest cross-validation error
lambda_min = cv_result$lambda.min

glmnet_result = glmnet(X_learn,Y_learn, lambda = lambda_min, alpha = 1,
    family = "binomial")

beta = glmnet_result$beta[, 1]
intercept = glmnet_result$a0

Ypred = (X_test %*% beta + intercept > 0)
testerr = mean(Y_test != Ypred)
print("LASSO Test Error and minimizing lambda")
print(c(testerr, lambda_min))

print("LASSO Coefficients:")
print(glmnet_result$beta)


## preliminary LASSO plotting, needs to be rewritten in ggplot

par(mfrow = c(1,2))
lammod <- cv.glmnet(X_learn, Y_learn, alpha = 1, family = "binomial")
glmmod <- glmnet(X_learn, Y_learn, alpha = 1, family = "binomial")


plot(glmmod, xvar="lambda")
grid(col = "lightgray", lty = "dotted")

plot(lammod)
grid(col = "lightgray", lty = "dotted")