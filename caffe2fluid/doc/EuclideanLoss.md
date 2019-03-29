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
Caffe：对整个输入的欧氏距离进行取和后除以两倍的样本个数，最终获得一个数值。                                        

PaddlePaddle：使用elemenwise方式，计算`input`和`label`对应元素的欧式距离，最终获得一个array（输入和输出`shape`一致）：  
```python
preds = paddle.fluid.layers.data(name = 'preds', shape = [2,10], append_batch_size = False, dtype = 'float32')
labels = paddle.fluid.layers.data(name = 'labels', shape = [2,10], append_batch_size = False, dtype = 'float32')
loss = paddle.fluid.layers.square_error_cost(input = preds, label = labels)
count =  paddle.fluid.layers.fill_constant(shape=[1],dtype='float32', value=4)
out = paddle.fluid.layers.elementwise_div(x=loss, y=count)
out = paddle.fluid.layers.reduce_sum(input=out, dim=0)
out = paddle.fluid.layers.reduce_sum(input=out, dim=0)
```
