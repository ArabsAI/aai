class Metric:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, *args: dict, **kwargs: dict):
        raise NotImplementedError
