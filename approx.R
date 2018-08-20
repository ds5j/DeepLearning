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
test_data=matrix(runif(num_test,-3,3))
test_data=seq(-3, 3, length.out = 4)
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




## get gradients
get_grad=function(model,train_data)
{
  K  <- backend()
  wt <- model$trainable_weights
  grad <- K$gradients( model$output, wt)
  
  sess=K$get_session()
  #print(sess$run( grad, feed_dict = dict(input_layer=x) ))
  g=sess$run( grad, feed_dict = dict(input_layer=train_data) )
  return(sum(unlist(g)^2))
}
#####################################################################################################
####################################################################################################
predictions=vector(mode="list")
models=vector(mode="list")

#train_data=normalize(train_data)
num_hidden_units=4
lr0=.5/num_hidden_units
#without train dim it errors on summary
l2reg=.00008
input_layer <- layer_input(shape = 1, name = 'input')
output_layer <- input_layer %>% layer_dense(units = num_hidden_units, activation = 'tanh',kernel_regularizer = regularizer_l2(l2reg))  %>% layer_dense(units = 1,,kernel_regularizer = regularizer_l2(l2reg)) 
model_simple <- keras_model(inputs = input_layer, outputs = output_layer )

#,clipnorm=1,,clipvalue=1
opt <- optimizer_sgd(lr = lr0)

model_simple %>% compile(
  optimizer = opt, 
  loss = "mse", 
  metrics = c("mae")
)

lr_schedule <- function(epoch,lr) {  lr0/(1+(epoch/5)) }
lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

# define custom callback class
LossHistory <- R6::R6Class("LossHistory",
                           inherit = KerasCallback,
                           
                           public = list(
                             
                             losses = NULL,
                             
                             on_epoch_end = function(epoch, logs = list()) {
                               self$losses <- c(self$losses, logs[["loss"]])
                               print(paste('losses:',self$losses[length(self$losses)]))
                             }
                           ))
history <- LossHistory$new()


epochs=20000
wgts=NA
summary(model_simple)
#,callback_model_checkpoint("checkpoints.h5")
hist_approx<-model_simple %>% fit(train_data, train_target,
              epochs = epochs, batch_size = 10, verbose = 0
#              ,callback_lambda(on_epoch_begin=(print(paste(get_weights(model_simple)))))
#              ,callback_reduce_lr_on_plateau, #,list(lr_reducer)
,callbacks= list(callback_lambda( on_epoch_end = function(epoch, logs=list()) {
  cat("Epoch Begin\n")
  print(paste('epoch:',epoch))
  wgts=get_weights(model_simple)
  print(paste('wgt norm:',sum(unlist(wgts)^2), 'gradient norm:',get_grad(model_simple,train_data),'loss:',logs[["loss"]]))

  
  #print(paste('losses:',logs[["loss"]][length(logs[["loss"]])]))
}
))

)
              

plot(hist_approx)
print(hist_approx)
#, callback_tensorboard("logs/run_a")
#tensorboard("logs/run_a")


result <- model_simple %>% evaluate(test_data, test_target)
pred=predict(model_simple,(test_data))
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

##################################################################