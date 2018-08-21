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

test_data=matrix(seq(0,pi*3,pi*3/100))
test_target=y_function(test_data,0,0)

############## replace train and test  ########################## ############## ############## 
X1 = seq(0,pi,length.out = 7)[c(-1,-7)]
X2 = seq(pi,2*pi,length.out=15)
Y1 = -cos(X1)+runif(length(X1),-.25,.25)
Y2 = cos(3*(X2-pi))+runif(length(X2),-.25,.25)
X = c(X1,X2)
dim(X)=c(length(X),1)
Y = c(Y1,Y2)
dim(Y)=c(length(Y),1)
x_grid=seq(0,2*pi,length.out = 99)
y_noiseless=c(-cos(seq(0,pi,length.out=50))[-1],cos(3*(seq(pi,2*pi,length.out = 50)-pi)))
############## ############## ############## ############## ############## ############## 
train_data=X
train_target=Y
test_data=x_grid
test_target=y_noiseless

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
num_hidden_units=100
#lr0=ifelse(num_hidden_units>20,.02,.07)

lr0=.07
#without train dim it errors on summary
l2reg=0
input_layer <- layer_input(shape = 1, name = 'input')
output_layer <- input_layer %>% layer_dense(units = num_hidden_units, activation = 'tanh',kernel_regularizer = regularizer_l2(l2reg),bias_initializer=initializer_random_uniform(minval = -0.1, maxval = 0.1, seed = 104), kernel_initializer=initializer_random_normal(mean=0,stddev=.1, seed = 104))  %>% layer_dense(units = 1,,kernel_regularizer = regularizer_l2(l2reg),,bias_initializer=initializer_random_uniform(minval = -0.1, maxval = 0.1, seed = 104), kernel_initializer=initializer_random_normal(mean=0,stddev=.1, seed = 104)) 

model_simple <- keras_model(inputs = input_layer, outputs = output_layer )

#,clipnorm=1,,clipvalue=1
opt <- optimizer_sgd(lr = lr0,momentum=0)

model_simple %>% compile(
  optimizer = opt, 
  loss = "mse", 
  metrics = c("mae")
)


epochs=20000
wgts=NA

lr_schedule <- function(epoch,lr) {  lr0/(1+(epoch/epochs)) }
lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

summary(model_simple)
#,callback_model_checkpoint("checkpoints.h5")
hist_approx<-model_simple %>% fit(train_data, train_target,
              epochs = epochs, batch_size = 10, verbose = 0
#              ,callback_lambda(on_epoch_begin=(print(paste(get_weights(model_simple)))))
#              ,callback_reduce_lr_on_plateau, #,list(lr_reducer)
,callbacks= list(callback_lambda( on_epoch_end = function(epoch, logs=list()) {
    if (epoch %% 1000==0)
    {
     cat("Epoch End\n")
     print(paste('epoch:',epoch))
     wgts=get_weights(model_simple)
     print(paste('wgt norm:',sum(unlist(wgts)^2), 'gradient norm:',get_grad(model_simple,train_data),'loss:',logs[["loss"]]))
  }
  
}
),callback_terminate_on_naan(),lr_reducer
,callback_reduce_lr_on_plateau(monitor = "loss", factor = .5)  

)
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