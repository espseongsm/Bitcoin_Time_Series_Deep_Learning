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
