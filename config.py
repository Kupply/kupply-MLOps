class Config():

    def __init__(self, model_fn, gpu_id, batch_size, lines):
        self.model_fn = model_fn
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.lines = lines
        self.top_k = 1