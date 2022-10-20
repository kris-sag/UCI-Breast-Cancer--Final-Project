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
    grads = as.array(0)

    for (t in 1:maxiter){
        xb = X %*% beta_prev

        gradient = - t(X) %*% (Y - exp(xb)/(1 + exp(xb)) ) + lambda * 2 * beta_prev
        grads = append(grads, norm(gradient,'2'))

        beta_next = beta_prev - stepsize * gradient

        if (norm(gradient, '2') < 1e-4) {
            print(paste("Converged at iteration: ", t))
            break
        } else beta_prev = beta_next

        # cur_obj = objectiveValue(X, Y, beta_next, lambda)

        # if (t %% 5000 == 0)
        #     print(paste("Current iter: ", t, " Current objective value: ", cur_obj))

        if (t == maxiter){
            print("Reached maxiter; did not converge.")
        }
    }

    print(paste("Final gradient norm: ", norm(gradient, '2')))
    grads = grads[-1]
    return(list(ridge_beta = beta_next,
        grads = grads,
        iters = t))
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

#uncomment to reorder the data
set.seed(893476)

reorder = sample(1:n, n, replace=FALSE)

X = X[reorder, ]
Y = Y[reorder]

n_test = 119
n_learn = 450

Y_test = Y[1:n_test]
Y_learn = Y[(n_test+1):(n_test+n_learn)]

X_test = X[1:n_test, ]
X_learn = X[(n_test+1):(n_test+n_learn), ]

# ## Using our functions to compute logistic regression

output = logisticReg_gradientDescent(X_learn, Y_learn, 10)


## Making predictions on test data and computing the
## misclassification error

ridge_Ypred = (X_test %*% output$ridge_beta > 0)

## baseline prediction
if (mean(ridge_Ypred) > 0.5) {
    Ybaseline = rep(1, n_test)
} else {
    Ybaseline = rep(0, n_test)
}

baseline_err = mean( Y_test != Ybaseline )

print("Ridge Regularized Logistic Regression Test Error + Baseline:")
ridge_TE = mean( Y_test != ridge_Ypred )
print(sprintf("Ridge error: %.8f,   Baseline error: %.8f",
              ridge_TE, baseline_err))

# print("Logistic Regression Coefficients:")
# print(output$ridge_beta)

linebreak = "================================="
print(linebreak)

# ----- LASSO analysis with glmnet package -----
library(glmnet)
## automatic cross-validation from glmnet package
cv_result = cv.glmnet(x = X_learn, y = Y_learn, alpha = 1,
    family = "binomial")

## lambda that attains smallest cross-validation error
lambda_min = cv_result$lambda.min

glmnet_result = glmnet(X_learn,Y_learn, lambda = lambda_min, alpha = 1,
    family = "binomial")

lasso_beta = glmnet_result$beta
intercept = glmnet_result$a0

lasso_Ypred = (X_test %*% lasso_beta + intercept > 0)
lasso_TE = mean(Y_test != lasso_Ypred)

## baseline prediction
if (mean(lasso_Ypred) > 0.5) {
    Ybaseline = rep(1, n_test)
} else {
    Ybaseline = rep(0, n_test)
}

baseline_err = mean( Y_test != Ybaseline )


print("LASSO Test Error, minimum lambda, and baseline error")
print(sprintf("LASSO error: %.8f,   Minimum lambda: %.8f,   Baseline error: %.8f",
              lasso_TE, lambda_min, baseline_err))

# print("LASSO Coefficients:")
# print(lasso_beta)

print(linebreak)

## ----- PRINTS AND COMPARISONS -----

## comparing the predictions of ridge and lasso vs. the actual test data
# NB: This is important in the context of breast cancer slides, as a
# false negative (negative diagnosis when it should actually be positive)
# is worse than a false positive (positive diagnosis when actually negative)

algs = c("Ridge Regression", "LASSO Regression")
predictions = cbind(ridge_Ypred[, 1], lasso_Ypred[, 1], Y_test)
colnames(predictions) <- c(algs, "Test Data")


## creating a sort-of confusion matrix
falses = c("False Positive", "False Negative")
false_comparisons = matrix(0, 4, 2)
colnames(false_comparisons) <- algs
rownames(false_comparisons) <- c(falses, "Sensitivity (%)", "Specificity (%)")

n_benign = sum(Y_test == 0)
n_malignant = sum(Y_test == 1)

for (i in 1:2){
    false_neg = 0
    false_pos = 0

    ixs = predictions[, i] != Y_test
    ixs = which(ixs)

    if (length(ixs) > 0){
        for (j in 1:length(ixs)){
            if (Y_test[j] == 1){
                false_neg = false_neg + 1
            }

            else if (Y_test[j] == 0) {
                false_pos = false_pos + 1
            }
        }
    }
    false_comparisons[1, i] = false_pos
    false_comparisons[2, i] = false_neg
    false_comparisons[3, i] = round(100 * (n_malignant - false_neg) 
        / n_malignant, digits = 1)
    false_comparisons[4, i] = round(100 * (n_benign - false_pos) 
        / n_benign, digits = 1)
}


## comparisons
# comparison of test error
print("FINAL ERRORS")
print(sprintf("Ridge Test Error: %.8f,   LASSO Test Error: %.8f", ridge_TE, lasso_TE))
print(linebreak)

# comparison of coefficients
beta_comp = cbind(output$ridge_beta, lasso_beta)
colnames(beta_comp) <- algs

print("Ridge vs. LASSO Coefficients comparison:")
print(beta_comp)
print(linebreak)

# comparison of false positives and negatives
print("Pseudo confusion matrix:")
print(false_comparisons)

## Ridge plotting
grad_data = as.data.frame(cbind(1:length(output$grads), output$grads))
colnames(grad_data) <- c("Iteration", "Gradient_Norms")

grad_plot <- ggplot(grad_data,
    aes(x = Iteration, y = Gradient_Norms, color = Iteration)) +
    geom_line(size = 1) +
    scale_y_log10() +
    labs(title = "Gradient Norm vs. Iterations") +
    theme_minimal() +
    scale_fill_manual(values="#9999CC")

png(filename = "grad_plot.png", width = 800, height = 600)
plot(grad_plot)
dev.off()


## LASSO plotting

lammod <- cv.glmnet(X_learn, Y_learn, alpha = 1, family = "binomial")
glmmod <- glmnet(X_learn, Y_learn, alpha = 1, family = "binomial")

glmmod_beta = coef(glmmod)

tmp <- as.data.frame(as.matrix(glmmod_beta))
tmp$coef <- row.names(tmp)
tmp <- reshape::melt(tmp, id = "coef")
tmp$variable <- as.numeric(gsub("s", "", tmp$variable))
tmp$lambda <- glmmod$lambda[tmp$variable+1] # extract the lambda values
tmp$norm <- apply(abs(glmmod_beta[-1,]), 2, sum)[tmp$variable+1]

lasso_coefs <- ggplot(tmp[tmp$coef != "(Intercept)",], 
    aes(lambda, value, group = coef, color = coef, linetype = coef)) + 
    geom_line(size = 1) + 
    geom_point(size = 1) +
    facet_wrap(~ coef) +
    scale_x_log10() + 
    xlab("log(lambda)") + 
    guides(color = guide_legend(title = ""), 
           linetype = guide_legend(title = "")) +
    theme_bw() + 
    theme(legend.key.width = unit(1,"lines")) +
    scale_y_continuous(limits=c(-20,20)) +
    labs(title="LASSO beta coefficients")

png(filename="LASSO_beta_coefficients.png", width = 1000, height = 600)

plot(lasso_coefs)
dev.off()

png(filename = "cross_validation_lambda_curve.png", width = 800, height = 600)
plot(lammod)
grid(col = "lightgray", lty = "dotted")
title("Cross Validation Lambda Curve", line = 2.5)
dev.off()
