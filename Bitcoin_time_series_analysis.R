rm(list = ls())    #delete objects
cat("\014")
library(ggplot2)
library(dplyr)
library(glmnet)
library(reshape)
library(gridExtra)

# call required libraries for DL
install.packages("kerasR")
library(kerasR)
library(keras)
library(tensorflow)


time_series = read.csv("BTC-USD.csv")
time_series = time_series[, 2]

#time_series = scale(time_series) #standardize the data so that all standardized variables are given a mean of zero and a standard deviation of one.
p = 30
n = length(time_series) - p - 1
X = matrix(0, nrow = n, ncol = p)
y = matrix(0, nrow = n, ncol = 1)
k = 1
for (i in (p + 1):(n + p)) {
  X[k,]  = time_series[(i - 1):(i - p)] # first row of X is time_series[p:1]
  y[k, 1] = time_series[i]
  k      = k + 1
}

n.test  =   floor(0.2 * n)
n.train =   n  - n.test


X.train = X[1:n.train, ]
X.test  = X[-(1:n.train), ]
y.train = y[1:n.train]
y.test  = y[-(1:n.train)]

mean    = apply(X.train, 2, mean)
std     = apply(X.train, 2, sd)

X.train = scale(X.train, center = mean, scale = std)
X.test  = scale(X.test, center = mean, scale = std)


# early stopping for DL
early_stop = callback_early_stopping(monitor = "val_mae",
                                     min_delta = 3,
                                     patience = 3)

# possible number of epochs
num_epochs = 500

# step 1
# testing the number of nodes in a hidden layer
num.testing.nodes = 128
num.nodes_mae     = matrix(0, ncol = num.testing.nodes, nrow = 2)

for (j in 1:num.testing.nodes) {
  num.nodes = 2 * j
  
  cat("testing the number of nodes = " , num.nodes, "\n")
  Sys.sleep(3)
  
  # define neural nets
  build_DL_model = function() {
    model = keras_model_sequential() %>%
      layer_dense(
        units = p,
        activation = "relu",
        input_shape = dim(X.train)[[2]]
      ) %>%
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = num.nodes, activation = "relu") %>%
      layer_dense(units = 1)
    model %>% compile(optimizer = "rmsprop",
                      loss = "mse",
                      metrics = c("mae"))
  }
  
  DL = build_DL_model()
  
  # cross-validation
  k       = 5
  indices = sample(1:nrow(X.train))
  folds   = cut(indices, breaks = k, labels = FALSE)
  
  all_mae_histories = NULL
  
  for (i in 1:k) {
    cat(
      "processing cross-validation fold #",
      i,
      "for a hidden layer with",
      num.nodes,
      "nodes \n"
    )
    
    val_indices = which(folds == i, arr.ind = TRUE)
    val_data    = X.train[val_indices,]
    val_target  = y.train[val_indices]
    
    partial_train_data   = X.train[-val_indices,]
    partial_train_target = y.train[-val_indices]
    
    DL = build_DL_model()
    
    history = DL %>% fit(
      partial_train_data,
      partial_train_target,
      validation_data = list(val_data, val_target),
      epochs = num_epochs,
      batch_size = 32,
      callbacks = list(early_stop)
    )
    mae_history       = history$metrics$val_mae
    all_mae_histories = rbind(all_mae_histories, mae_history)
    cv.mae            = apply(all_mae_histories, 2, mean)
    cv.mae.std        = apply(all_mae_histories, 2, sd)
    cv.min.mae        = min(cv.mae)
    cv.min.mae.std    = cv.mae.std[which.min(cv.mae)]
  }
  num.nodes_mae[1, j] = cv.min.mae
  num.nodes_mae[2, j] = cv.min.mae.std
  
}

num.nodes_mae           = t(num.nodes_mae)
num.nodes_mae           = cbind(num.nodes_mae, c(seq(2, 256, 2)))
colnames(num.nodes_mae) = c("cv.min.mae", "cv.min.mae.std", "number.of.nodes")
num.nodes_mae           = data.frame(num.nodes_mae)

arrange(num.nodes_mae, cv.min.mae, cv.min.mae.std)

ggplot(num.nodes_mae,
       aes(x = number.of.nodes, y = cv.min.mae, color = "red")) +
  geom_line() +
  theme(legend.position = "none") + ylim(0, 1000) +
  labs(
    x     = element_blank(),
    y     = "CV Mean Absolute Error",
    title = expression(Number ~ of ~ Nodes ~ at ~ Hidden ~ Layer ~ Three)
  )

# stpe 2
# define optimized architecture of the neural nets
build_DL_model = function() {
  model = keras_model_sequential() %>%
    layer_dense(
      units = p,
      activation = "relu",
      input_shape = dim(X.train)[[2]]
    ) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 22, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metrics = c("mae"))
}

# step 3
# testing the batch size
num.testing.batch.size = 10
batch.size_mae         = matrix(0, ncol = num.testing.batch.size, nrow = 2)

for (j in 1:num.testing.batch.size) {
  testing.batch.size = 2 ^ (j - 1)
  
  cat("testing the batch size = " , testing.batch.size, "\n")
  Sys.sleep(3)
  
  DL = build_DL_model()
  
  # cross-validation for neural nets
  k       = 5
  indices = sample(1:nrow(X.train))
  folds   = cut(indices, breaks = k, labels = FALSE)
  
  all_mae_histories = NULL
  
  for (i in 1:k) {
    cat(
      "processing cross-validation fold #",
      i,
      "for the batch size = ",
      testing.batch.size,
      "\n"
    )
    
    val_indices = which(folds == i, arr.ind = TRUE)
    val_data    = X.train[val_indices,]
    val_target  = y.train[val_indices]
    
    partial_train_data   = X.train[-val_indices,]
    partial_train_target = y.train[-val_indices]
    
    DL = build_DL_model()
    
    history = DL %>% fit(
      partial_train_data,
      partial_train_target,
      validation_data = list(val_data, val_target),
      epochs = num_epochs,
      batch_size = 16,
      callbacks = list(early_stop)
    )
    mae_history       = history$metrics$val_mae
    all_mae_histories = rbind(all_mae_histories, mae_history)
    cv.mae            = apply(all_mae_histories, 2, mean)
    cv.mae.std        = apply(all_mae_histories, 2, sd)
    cv.min.mae        = min(cv.mae)
    cv.min.mae.std    = cv.mae.std[which.min(cv.mae)]
  }
  batch.size_mae[1, j] = cv.min.mae
  batch.size_mae[2, j] = cv.min.mae.std
  
}

batch.size_mae           = t(batch.size_mae)
batch.size_mae           = cbind(batch.size_mae, c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
colnames(batch.size_mae) = c("cv.min.mae", "cv.min.mae.std", "batch.size")
batch.size_mae           = data.frame(batch.size_mae)

ggplot(batch.size_mae) +
  geom_line(aes(x = batch.size, y = cv.min.mae, color = "blue")) +
  theme(legend.position = "none") +
  labs(x = element_blank(),
       y = "CV Mean Absolute Error",
       title = expression(Batch ~ Size))

# step 4
# testing the number of epochs
all_mae_histories = NULL

# cross validation
k       = 5
indices = sample(1:nrow(X.train))
folds   = cut(indices, breaks = k, labels = FALSE)

for (i in 1:k) {
  cat("processing cross-validation fold #",
      i,
      "for the number of epochs \n")
  
  val_indices = which(folds == i, arr.ind = TRUE)
  val_data    = X.train[val_indices,]
  val_target  = y.train[val_indices]
  
  partial_train_data   = X.train[-val_indices,]
  partial_train_target = y.train[-val_indices]
  
  DL = build_DL_model()
  
  history = DL %>% fit(
    partial_train_data,
    partial_train_target,
    validation_data = list(val_data, val_target),
    epochs = 500,
    batch_size = 8
  )
  mae_history       = history$metrics$val_mae
  all_mae_histories = rbind(all_mae_histories, mae_history)
}

average_mae_history = data.frame(epoch = seq(1:ncol(all_mae_histories)),
                                 cv_mae = apply(all_mae_histories, 2, mean))

ggplot(average_mae_history, aes(x = epoch, y = cv_mae)) + geom_smooth() +
  theme(legend.position = "none") +
  labs(x = element_blank(),
       y = "CV Mean Absolute Error",
       title = expression(Number ~ of ~ Epochs))

# step 5
# train the DL
best.num.epoch = 130

DL = build_DL_model()

DL %>% fit(X.train, y.train,
           epochs = best.num.epoch, batch_size = 8)

y.train.hat.DL = DL %>% predict(X.train)
y.test.hat.DL  = DL %>% predict(X.test)

time.series.DL                =     data.frame(c(rep("observed", n),
                                                 rep("train.hat", n.train),
                                                 rep("test.hat", n.test)),
                                               c(1:n, 1:n),
                                               c(y, y.train.hat.DL, y.test.hat.DL))
colnames(time.series.DL)      =     c("state", "time", "value")
time.series.DL.plot           =     ggplot(time.series.DL, 
                                           aes(x = time, y =value, colour = state)) + 
                                      geom_line() + 
                                      labs(title = expression(Deep ~ Learning))
time.series.DL.plot

res.time.series.DL            =     data.frame(c(rep("train", n.train), rep("test", n.test)),
                                               c(1:n),
                                               c(y.train.hat.DL - y.train, y.test.hat.DL - y.test))
colnames(res.time.series.DL)  =     c("state", "time", "residual")
res.time.series.DL.plot       =     ggplot(res.time.series.DL, 
                                           aes(x = time, y =residual, colour = state)) + 
                                      geom_line() + ylim(-5000, 5000) + 
                                      labs(title = expression(Deep ~ Learning))
res.time.series.DL.plot

# step 6
# cross validation for lasso
cat("processing cross-validation for lasso\n")
cv.lasso                         =     cv.glmnet(
  X.train,
  y.train,
  alpha = 1,
  family = "gaussian",
  intercept = T,
  type.measure = "mae",
  nfolds = 20
)
lasso.fit                        =     glmnet(
  X.train,
  y.train,
  alpha = 1,
  family = "gaussian",
  intercept = T,
  lambda = cv.lasso$lambda.min
)
y.train.hat.lasso                =     X.train %*% lasso.fit$beta + lasso.fit$a0  #same as: y.train.hat_  =    predict(lasso.fit, newx = X.train, type = "response", cv.lasso$lambda.min)
y.test.hat.lasso                 =     X.test %*% lasso.fit$beta  + lasso.fit$a0  #same as: y.test.hat_  =    predict(lasso.fit, newx = X.test, type = "response", cv.lasso$lambda.min)
y.train.hat.lasso                =     as.vector(y.train.hat.lasso)
y.test.hat.lasso                 =     as.vector(y.test.hat.lasso)

time.series.lasso                =     data.frame(c(
  rep("observed", n),
  rep("train.hat", n.train),
  rep("test.hat", n.test)
),
c(1:n, 1:n),
c(y, y.train.hat.lasso, y.test.hat.lasso))
colnames(time.series.lasso)      =     c("state", "time", "value")
time.series.lasso.plot           =     ggplot(time.series.lasso, aes(x =
                                                                       time, y = value, colour = state)) + geom_line() + labs(title = expression(LASSO))
time.series.lasso.plot

res.time.series.lasso            =     data.frame(c(rep("train", n.train), rep("test", n.test)),
                                                  c(1:n),
                                                  c(y.train.hat.lasso - y.train, y.test.hat.lasso - y.test))
colnames(res.time.series.lasso)  =     c("state", "time", "residual")
res.time.series.lasso.plot       =     ggplot(res.time.series.lasso, aes(x =
                                                                           time, y = residual, colour = state)) + geom_line() + ylim(-5000, 5000) + labs(title = expression(LASSO))
res.time.series.lasso.plot

# step 7
# cross validation for ridge
cat("processing cross-validation for ridge\n")
cv.ridge                         =     cv.glmnet(
  X.train,
  y.train,
  alpha = 0,
  family = "gaussian",
  intercept = T,
  type.measure = "mae",
  nfolds = 20
)
ridge.fit                        =     glmnet(
  X.train,
  y.train,
  alpha = 0,
  family = "gaussian",
  intercept = T,
  lambda = cv.lasso$lambda.min
)
y.train.hat.ridge                =     X.train %*% ridge.fit$beta + ridge.fit$a0  #same as: y.train.hat_  =    predict(lasso.fit, newx = X.train, type = "response", cv.lasso$lambda.min)
y.test.hat.ridge                 =     X.test %*% ridge.fit$beta  + ridge.fit$a0  #same as: y.test.hat_  =    predict(lasso.fit, newx = X.test, type = "response", cv.lasso$lambda.min)
y.train.hat.ridge                =     as.vector(y.train.hat.ridge)
y.test.hat.ridge                 =     as.vector(y.test.hat.ridge)

time.series.ridge                =     data.frame(c(
  rep("observed", n),
  rep("train.hat", n.train),
  rep("test.hat", n.test)
),
c(1:n, 1:n),
c(y, y.train.hat.ridge, y.test.hat.ridge))
colnames(time.series.ridge)      =     c("state", "time", "value")
time.series.ridge.plot           =     ggplot(time.series.ridge, aes(x =
                                                                       time, y = value, colour = state)) + geom_line() + labs(title = expression(Ridge))
time.series.ridge.plot

res.time.series.ridge            =     data.frame(c(rep("train", n.train), rep("test", n.test)),
                                                  c(1:n),
                                                  c(y.train.hat.ridge - y.train, y.test.hat.ridge - y.test))
colnames(res.time.series.ridge)  =     c("state", "time", "residual")
res.time.series.ridge.plot       =     ggplot(res.time.series.ridge, aes(x =
                                                                           time, y = residual, colour = state)) + geom_line() + ylim(-5000, 5000) + labs(title = expression(Ridge))
res.time.series.ridge.plot

# step 8
# cross validation for elastic
cat("processing cross-validation for elastic net\n")
cv.elastic                         =     cv.glmnet(
  X.train,
  y.train,
  alpha = 0.5,
  family = "gaussian",
  intercept = T,
  type.measure = "mae",
  nfolds = 20
)
elastic.fit                        =     glmnet(
  X.train,
  y.train,
  alpha = 0.5,
  family = "gaussian",
  intercept = T,
  lambda = cv.lasso$lambda.min
)
y.train.hat.elastic                =     X.train %*% elastic.fit$beta + elastic.fit$a0  #same as: y.train.hat_  =    predict(lasso.fit, newx = X.train, type = "response", cv.lasso$lambda.min)
y.test.hat.elastic                 =     X.test %*% elastic.fit$beta  + elastic.fit$a0  #same as: y.test.hat_  =    predict(lasso.fit, newx = X.test, type = "response", cv.lasso$lambda.min)
y.train.hat.elastic                =     as.vector(y.train.hat.elastic)
y.test.hat.elastic                 =     as.vector(y.test.hat.elastic)

time.series.elastic                =     data.frame(c(
  rep("observed", n),
  rep("train.hat", n.train),
  rep("test.hat", n.test)
),
c(1:n, 1:n),
c(y, y.train.hat.elastic, y.test.hat.elastic))
colnames(time.series.elastic)      =     c("state", "time", "value")
time.series.elastic.plot           =     ggplot(time.series.elastic, aes(x =
                                                                           time, y = value, colour = state)) + geom_line() + labs(title = expression(Elastic ~ Net))
time.series.elastic.plot

res.time.series.elastic            =     data.frame(c(rep("train", n.train), rep("test", n.test)),
                                                    c(1:n),
                                                    c(y.train.hat.elastic - y.train, y.test.hat.elastic - y.test))
colnames(res.time.series.elastic)  =     c("state", "time", "residual")
res.time.series.elastic.plot       =     ggplot(res.time.series.elastic,
                                                aes(x = time, y = residual, colour = state)) + geom_line() + ylim(-5000, 5000) + labs(title = expression(Elastic ~ Net))
res.time.series.elastic.plot

mean(abs(y.train.hat.lasso - y.train))
mean(abs(y.test.hat.lasso - y.test))

mean(abs(y.train.hat.ridge - y.train))
mean(abs(y.test.hat.ridge - y.test))

mean(abs(y.train.hat.elastic - y.train))
mean(abs(y.test.hat.elastic - y.test))

mean(abs(y.train.hat.DL - y.train))
mean(abs(y.test.hat.DL - y.test))
