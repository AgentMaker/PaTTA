import paddle

def load_model(path='output/model'):
    model = paddle.jit.load(path=path)

    return model