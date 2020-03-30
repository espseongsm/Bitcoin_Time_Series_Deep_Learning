# Call required libraries
library(dplyr)
library(MASS)
library(glmnet)
library(ggplot2)
library(reshape)
library(gridExtra)

# call required libraries for DL
install_keras()
library(keras)
install.packages("kerasR")
library(kerasR)
install_tensorflow()
library(tensorflow)

x = read.csv("BTC-USD.csv")
x = x[, 2]

# function that create time series data from history data of bitcoin
time_series_data = function(history_data) {
  data_size = readline(prompt = "Enter the data size: ")
  the_number_of_variables = readline(prompt = "Enter the number of variables: ")
  
  # convert character into integer
  data_size = as.integer(data_size)
  the_number_of_variables = as.integer(the_number_of_variables)
  
  if (length(history_data) < data_size + the_number_of_variables)
  {
    print("we need more history data")
  } else{
    TSD = matrix(0, nrow = data_size, ncol = the_number_of_variables + 1)
    for (i in 1:data_size) {
      TSD[i,] = history_data[(i):(i + the_number_of_variables)]
    }
  }
  TSD
}

a = time_series_data(x)

dim(a)

t.data = a

# data structure and rates
n = dim(t.data)[1]
p = dim(t.data)[2] - 1

# Modelling factors
iterations = 100
CV_iterations = 30
Dlearn_rate = 0.8
sampling.rate = 1

# train and test error rate matrix
train_error = matrix(0, nrow = iterations, ncol = 3)
colnames(train_error) = c("Deep Learning", "LASSO", "Ridge")

cv_error = matrix(0, nrow = CV_iterations, ncol = 3)
colnames(cv_error) = c("Deep Learning", "LASSO", "Ridge")

test_error = matrix(0, nrow = iterations, ncol = 3)
colnames(test_error) = c("Deep Learning", "LASSO", "Ridge")

lasso.coef = matrix(0, ncol = iterations, nrow = p + 1)
ridge.coef = matrix(0, ncol = iterations, nrow = p + 1)

# convert to data frame
train_error = data.frame(train_error)
test_error = data.frame(test_error)

# time of cv and fit
time.cv = matrix(0, nrow = CV_iterations, ncol = 3)
colnames(time.cv) = c("Deep Learning", "LASSO", "Ridge")

time.fit = matrix(0, nrow = iterations, ncol = 3)
colnames(time.fit) = c("Deep Learning", "LASSO", "Ridge")

# sampling from t.data
sampling = sample(n, n * sampling.rate)
sampling.data = data.frame(t.data[sampling, ])
sampling.n = dim(sampling.data)[1]

num_epochs_min = matrix(0, ncol = 1, nrow = CV_iterations)
lasso.lambda.min = matrix(0, ncol = 1, nrow = CV_iterations)
ridge.lambda.min = matrix(0, ncol = 1, nrow = CV_iterations)
# preparation for lasso and ridge
# X = model.matrix(sampling.data[,1] ~ ., sampling.data)[, -1]
# y = sampling.data[,1]

# early stopping for DL
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
# possible number of epochs
num_epochs = 500 

# tune hyperparameters by 100 iterations
for (m in 1:CV_iterations) {
  cat("processing cross-validation iteration #", m, "for tuning hyperparameters\n")
  Sys.sleep(5)
  # create a training data vector for dividing the data set.
  train = sample(sampling.n, sampling.n * Dlearn_rate)
  
  train_data = sampling.data[train, -1]
  test_data = sampling.data[-train, -1]
  
  mean = apply(train_data, 2, mean) 
  std = apply(train_data, 2, sd)
  
  train_data = scale(train_data, center = mean, scale = std) 
  test_data = scale(test_data, center = mean, scale = std)
  
  train_target = sampling.data[train,1]
  test_target = sampling.data[-train,1]
  
  # define neural nets
  build_DL_model = function() {
    model = keras_model_sequential() %>%
      layer_dense(units = 64, activation = "relu",
                  input_shape = dim(train_data)[[2]]) %>%
      layer_dense(units = 128, activation = "relu") %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 1)
    model %>% compile(
      optimizer = "rmsprop",
      loss = "mse",
      metrics = c("mae")
    ) }
  
  # cross-validation for neural nets
  ptm = proc.time()
  k = 5
  indices = sample(1:nrow(train_data))
  folds = cut(indices, breaks = k, labels = FALSE)
  
  all_mae_histories = NULL 
  
  for (i in 1:k) {
    cat("processing cross-validation fold #", i, "for neural nets\n")
    
    val_indices = which(folds == i, arr.ind = TRUE) 
    val_data = train_data[val_indices,]
    val_target = train_target[val_indices]
    
    partial_train_data = train_data[-val_indices,] 
    partial_train_target = train_target[-val_indices]
    
    DL = build_DL_model()
    
    history = DL %>% fit(
      partial_train_data, partial_train_target,
      validation_data = list(val_data, val_target),
      epochs = num_epochs, batch_size = 4, callbacks = list(early_stop)
    )
    mae_history = history$metrics$val_mae 
    all_mae_histories = rbind(all_mae_histories, mae_history)
  }
  ptm = proc.time() - ptm
  time.cv[m, 1]  = ptm["elapsed"]
  
  cv_error[m, 1] = min(colMeans(all_mae_histories))
  num_epochs_min[m, 1] = which.min(colMeans(all_mae_histories))
  
  # cross validation for lasso
  cat("processing cross-validation for lasso\n")
  Sys.sleep(5)
  ptm = proc.time()
  cv.lasso = cv.glmnet(
    train_data,
    train_target,
    family = "gaussian",
    intercept = T,
    type.measure = "mae",
    nfolds = 5
  )
  ptm = proc.time() - ptm
  time.cv[m, 2]  = ptm["elapsed"]
  
  cv_error[m, 2] = min(cv.lasso$cvm)
  lasso.lambda.min[m, 1] = cv.lasso$lambda.min
  
  # cross validation for ridge
  cat("processing cross-validation for ridge\n")
  Sys.sleep(5)
  ptm = proc.time()
  cv.ridge = cv.glmnet(
    train_data,
    train_target,
    alpha = 0,
    family = "gaussian",
    intercept = T,
    type.measure = "mae",
    nfolds = 5
  )
  ptm = proc.time() - ptm
  time.cv[m, 3]  = ptm["elapsed"]
  
  cv_error[m, 3] = min(cv.ridge$cvm)
  ridge.lambda.min[m, 1] = cv.ridge$lambda.min
  
}

# best hyperparameters with average: num_epochs for DL and lambda for lasso&ridge
best_num_epochs = round(mean(num_epochs_min))
best_lasso_lambda = mean(lasso.lambda.min)
best_ridge_lambda = mean(ridge.lambda.min)

# best hyperparameters with least mae: num_epochs for DL and lambda for lasso&ridge
best_num_epochs = num_epochs_min[which.min(cv_error[,1]),]
best_lasso_lambda = lasso.lambda.min[which.min(cv_error[,2]),]
best_ridge_lambda = ridge.lambda.min[which.min(cv_error[,3]),]

# 100 iteration for perforamnce analysis: mae, time, and coefficients
for (m in 1:iterations) {
  cat("processing iteration #", m, "for performance analysis\n")
  Sys.sleep(5)
  # create a training data vector for dividing the data set.
  train = sample(sampling.n, sampling.n * Dlearn_rate)
  
  train_data = sampling.data[train, -1]
  test_data = sampling.data[-train, -1]
  
  mean = apply(train_data, 2, mean) 
  std = apply(train_data, 2, sd)
  
  train_data = scale(train_data, center = mean, scale = std) 
  test_data = scale(test_data, center = mean, scale = std)
  
  train_target = sampling.data[train,1]
  test_target = sampling.data[-train,1]
  
  # neural nets fit
  DL = build_DL_model()
  
  cat("fit the neural nets\n")
  ptm = proc.time()
  DL %>% fit(
    train_data, train_target,
    epochs = best_num_epochs, batch_size = 4, callbacks = list(early_stop))
  ptm = proc.time() - ptm
  time.fit[m, 1]  = ptm["elapsed"]
  
  # neural net performance from train and test data
  train_result = DL %>% evaluate(train_data, train_target)
  train_error[m, 1] = train_result$mae
  test_result = DL %>% evaluate(test_data, test_target)
  test_error[m, 1] = test_result$mae
  
  # lasso fit
  cat("fit the lasso\n")
  ptm = proc.time()
  lasso.mod = glmnet(
    train_data,
    train_target,
    alpha = 1,
    family = "gaussian",
    intercept = T,
    lambda = best_lasso_lambda,
    standardize = F
  )
  ptm = proc.time() - ptm
  time.fit[m, 2]  = ptm["elapsed"]
  
  # lasso performance from train data
  lasso.coef[, m] = coef(lasso.mod)[, 1]
  lasso.pred = predict(lasso.mod,
                       s = best_lasso_lambda,
                       newx = train_data,
                       type = "response")
  lasso.pred
  train_error[m, 2] = mean(abs(train_target - lasso.pred))
  
  # lasso performance from test data
  lasso.pred = predict(lasso.mod,
                       s = best_lasso_lambda,
                       newx = test_data,
                       type = "response")
  test_error[m, 2] = mean(abs(test_target - lasso.pred))
  
  # ridge fit
  cat("fit the ridge\n")
  ptm = proc.time()
  ridge.mod = glmnet(
    train_data,
    train_target,
    alpha = 0,
    family = "gaussian",
    intercept = T,
    lambda = best_ridge_lambda,
    standardize = F
  )
  ptm = proc.time() - ptm
  time.fit[m, 3]  = ptm["elapsed"]

  # ridge performance from train data
  ridge.coef[, m] = as.matrix(coef(ridge.mod))
  ridge.pred = predict(ridge.mod,
                       s = best_ridge_lambda,
                       newx = train_data,
                       type = "response")
  ridge.pred
  train_error[m, 3] = mean(abs(train_target - ridge.pred))

  # ridge performance from test data
  ridge.pred = predict(ridge.mod,
                       s = best_ridge_lambda,
                       newx = test_data,
                       type = "response")
  test_error[m, 3] = mean(abs(test_target - ridge.pred))
}

############################################
############################################

# store error rate and coef
write.csv(ridge.coef, file = "D80_rdige_coef.csv")
write.csv(lasso.coef, file = "D80_lasso_coef.csv")

write.csv(cv_error, file = "D80_cv_error.csv")
write.csv(test_error, file = "D80_test_error.csv")
write.csv(train_error, file = "D80_train_error.csv")

write.csv(time.cv, file = "D80_time_cv.csv")
write.csv(time.fit, file = "D80_time_fit.csv")

write.csv(num_epochs_min, file = "D80_num_epochs_min.csv")
write.csv(lasso.lambda.min, file = "D80_lasso_lambda_min.csv")
write.csv(ridge.lambda.min, file = "D80_ridge_lambda_min.csv")

write.csv(batch_size_all_mae_histories, file = "D80_batch_size_test mae")
