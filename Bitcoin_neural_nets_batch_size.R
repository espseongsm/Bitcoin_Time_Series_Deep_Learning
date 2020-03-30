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
Dlearn_rate = 0.8
sampling.rate = 1

# sampling from t.data
sampling = sample(n, n * sampling.rate)
sampling.data = data.frame(t.data[sampling, ])
sampling.n = dim(sampling.data)[1]

# early stopping for DL
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
# possible number of epochs
num_epochs = 500 

batch_size_all_mae_histories = matrix(0, ncol = 6, nrow = 100)
batch_size_mae_history = rep(0,100)

# testing the batch size
for ( bs in c(0,1,2,3,4,5)){
  cat("testing the batch size = " , 2^bs, "\n")
  Sys.sleep(5)
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
  DL = build_DL_model()
  for (i in 1:100){
    cat("processing intereation #", i, "for testing the performance at the batch size = " , 2^bs, "\n")
    Sys.sleep(5)
    history = DL %>% fit(
    train_data, train_target,
    validation_data = list(test_data, test_target),
    epochs = num_epochs, batch_size = 2^bs, callbacks = list(early_stop)
  )
  batch_size_mae_history[i] = min(history$metrics$val_mae)
  }
  batch_size_all_mae_histories[,(bs+1)] = batch_size_mae_history
}

mean(batch_size_mae_history)
apply(batch_size_all_mae_histories, 2, mean)
apply(batch_size_all_mae_histories, 2, sd)
