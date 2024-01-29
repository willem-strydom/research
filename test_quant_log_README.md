# test_quant_log
test quantized logisitc regression for convergence, accuracy, etc


problems:
My gradient descent for quantized logistic regression will not do more than one update before getting stuck in a loop. I think that this could be for a few reasons:
#1 - the most obvious, I just did not implement the quantized loss or descent algorithm correctly...
#2 - w won't be updated if the gradient is too small, since sign(w - c*gradient) is just going to be w if c*gradient is small. 
#3 - w won't be updated if the gradient has the opposite sign as w, since sign(w - c*gradient) is just going to be w if sign(gradient) = sign(w). This seems to be the case in my implementation
