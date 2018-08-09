# a build a network for x^2
setwd('~/DeepLearning/')
library(keras)
install_keras()


num_train=8
num_test=4

num_hidden_units=3

#train
train_data=matrix(runif(num_train,-3,3))
train_target=train_data^2
# test
test_data=matrix(runif(num_test,-3,3))=seq(-3, 3, length.out = 4)

test_target=test_data^2

#without train dim it errors on summary
model_simple <- keras_model_sequential() %>% 
  layer_dense(units = num_hidden_units,input_shape = dim(train_data)[[2]],activation = "relu") %>% 
  layer_dense(units = 1) 

model_simple %>% compile(
  optimizer = "rmsprop", 
  loss = "mse", 
  metrics = c("mae")
)
model_simple$optimizer$lr = 1e-2

summary(model_simple)

hist_approx<-model_simple %>% fit(train_data, train_target,
              epochs = 10000, batch_size = 6, verbose = 0)

plot(hist_approx)

result <- model_simple %>% evaluate(test_data, test_target)


pred=predict(model_simple,test_data)

plot(test_data,test_target,col='blue',main=paste('Test Data num  layers=1, num hidden units=',num_hidden_units))
points(test_data, predict(model_simple,test_data), col = "red")
points(train_data,train_target,col='green')
#pch=c(15,17)
legend( x="topleft", 
        legend=c("Actual values","Predicted Values"),
        col=c("blue","red"), lwd=1, lty=c(1,2), 
         )