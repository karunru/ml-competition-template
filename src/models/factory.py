from .cat import CatBoostModel

def catboost() -> CatBoostModel:
    return CatBoostModel()

def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
