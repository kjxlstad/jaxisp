# TODO: add typing
class ISPNode:
    def __init__(self, config):
        self.reconfigure(config)

    def reconfigure(self, config):
        self.execute = self.compile(config)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)
