import paddle

# Related to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dygraph/jit/load_cn.html#load
def load_model(path='output/model'):
    model = paddle.jit.load(path=path)

    return model
