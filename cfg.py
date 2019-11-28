class config():
    def __init__(self):
        self.IMAGE_PATH = r'G:\image data\natural_images'
        self.INPUT_SIZE = (227,227,3)
        self.BATCH_SIZE = 64
        self.EPOCH = 500
        self.NUM_CLASS = 1000
        self.KEEP_PROP = 0.5
        self.LEARNING_RATE = 0.01
