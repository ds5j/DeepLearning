# a build a network for x^2
setwd('~/DeepLearning/')
library(keras)
install_keras()


num_train=8
num_test=4
#train
train_data=matrix(runif(num_train,-3,3))
train_target=train_data^2
# test
test_data=matrix(runif(num_test,-3,3))=seq(-3, 3, length.out = 4)
test_target=test_data^2

gen_noise=function(x,a=-.25,b=.25)
{
  runif(x,a,b)
  
}

# lawrence and giles
train_data=matrix(unique(c((seq(from=0,to=314,by=314/5)/100)[2:6],seq(from=0,to=314,by=314/15)/100+3.14)))
y_function=function(x,a=-.25,b=.25)
{
  ifelse(x<3.14,-cos(x)+gen_noise(sum(x<3.14),a,b)  ,
        cos(3*(x-3.14))+gen_noise(sum(x>=3.14),a,b)  )
  
}
train_target=y_function(train_data)

test_data=matrix(seq(0,6.28,6.28/100))
test_target=y_function(test_data,0,0)
#####################################################################################################
####################################################################################################
predictions=vector(mode="list")
models=vector(mode="list")

num_hidden_units=100

#without train dim it errors on summary
model_simple <- keras_model_sequential() %>% 
  layer_dense(units = num_hidden_units,input_shape = dim(train_data)[[2]],activation='tanh'
           #   ,kernel_regularizer = regularizer_l2(l =.01) 
              
              ) %>%
  layer_dense(units = 1) 
#,clipnorm=1,,clipvalue=1
opt <- optimizer_sgd(lr = 0.01)

model_simple %>% compile(
  optimizer = opt, 
  loss = "mse", 
  metrics = c("mae")
)

lr_schedule <- function(epoch,lr) {  .01/(1+(epoch/5000)) }
lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

epochs=20000

summary(model_simple)
#,callback_model_checkpoint("checkpoints.h5")
hist_approx<-model_simple %>% fit(train_data, train_target,
              epochs = epochs, batch_size = 10, verbose = 0,list(lr_reducer)
              )

plot(hist_approx)
#, callback_tensorboard("logs/run_a")
#tensorboard("logs/run_a")


result <- model_simple %>% evaluate(test_data, test_target)
pred=predict(model_simple,test_data)
predictions[[num_hidden_units]]=pred
models[[num_hidden_units]]=model_simple

plot(test_data,test_target,col='blue',main=paste('Test Data num  layers=1, num hidden units=',num_hidden_units),type='l',ylim=c(-2,2))
lines(test_data, predict(model_simple,test_data), col = "red",lty=2)
points(train_data,train_target,col='green')
#pch=c(15,17)
legend( x="topleft", 
        legend=c("Actual values","Predicted Values"),
        col=c("blue","red"), lwd=1, lty=c(1,2), 
         )






############################################################
K  <- backend()
wt <- model$trainable_weights
grad <- K$gradients( model$output, wt)[[1]]

# Normalize gradients.
grads <- grads / k_maximum(k_mean(k_abs(grads)), k_epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
fetch_loss_and_grads <- k_function(list(dream), list(loss,grads))

eval_loss_and_grads <- function(image){
  outs <- fetch_loss_and_grads(list(image))
  list(
    loss_value = outs[[1]],
    grad_values = outs[[2]]
  )
}
##################################################################