## EuclideanLoss


### [EuclideanLoss](http://caffe.berkeleyvision.org/tutorial/layers/euclideanloss.html)
```
layer {
    name: "loss"
    type: "EuclideanLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
}
```


### [paddle.fluid.layers.square_error_cost](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-167-square_error_cost)
```python
paddle.fluid.layers.square_error_cost(
    input,
    label
)
```  

### 功能差异
#### 实现方式
Caffe：计算的是整个输入的欧氏距离除以两倍的样本个数，最终获得的标量输出。                                        

PaddlePaddle：使用elemenwise方式，计算`input`和`label`对应元素的L2距离，输入和输出`shape`一致：  
```python
inputs = paddle.fluid.layers.data(name = 'data1', shape = [2,3,227,227], append_batch_size = False, dtype = 'float32')
labels = paddle.fluid.layers.data(name = 'data1', shape = [2,3,227,227], append_batch_size = False, dtype = 'float32')
loss = paddle.fluid.layers.square_error_cost(input = inputs, label = labels)
sum = paddle.fluid.layers.sum(x = loss)
res = sum/(2*inputs.shape[0])
```
