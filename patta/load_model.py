import paddle

def load_model(path='model', model_filename='__model__', params_filename='__params__'):
    model = paddle.jit.load(path=path,
                            model_filename=model_filename,
                            params_filename=params_filename)

    return model