# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from copy import deepcopy
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  LinearRegression, Ridge, RidgeClassifier,
                                  RidgeCV, RidgeClassifierCV,
                                  SGDRegressor, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB,
                                 GaussianNB, MultinomialNB)
from sklearn.dummy import DummyClassifier, DummyRegressor

from .. utils import raise_error, warn, logger
from . dynamic import DynamicSelection

_available_models = {
    'svm': {
        'regression': SVR,
        'classification': SVC,
    },
    'rf': {
        'regression': RandomForestRegressor,
        'classification': RandomForestClassifier,
    },
    'et': {
        'regression': ExtraTreesRegressor,
        'classification': ExtraTreesClassifier,
    },
    'dummy': {
        'regression': DummyRegressor,
        'classification': DummyClassifier,
    },
    'gauss': {
        'regression': GaussianProcessRegressor,
        'classification': GaussianProcessClassifier,
    },
    'logit': {
        'classification': LogisticRegression,
    },
    'logitcv': {
        'classification': LogisticRegressionCV,
    },
    'linreg': {
        'regression': LinearRegression,
    },
    'ridge': {
        'regression': Ridge,
        'classification': RidgeClassifier,
    },
    'ridgecv': {
        'regression': RidgeCV,
        'classification': RidgeClassifierCV,
    },
    'sgd': {
        'regression': SGDRegressor,
        'classification': SGDClassifier,
    },
    'adaboost': {
        'regression': AdaBoostRegressor,
        'classification': AdaBoostClassifier,
    },
    'bagging': {
        'regression': BaggingRegressor,
        'classification': BaggingClassifier,
    },
    'gradientboost': {
        'regression': GradientBoostingRegressor,
        'classification': GradientBoostingClassifier,
    },
    'nb_bernoulli': {
        'classification': BernoulliNB,
    },
    'nb_categorical': {
        'classification': CategoricalNB,
    },
    'nb_complement': {
        'classification': ComplementNB,
    },
    'nb_gaussian': {
        'classification': GaussianNB,
    },
    'nb_multinomial': {
        'classification': MultinomialNB,
    },
    'ds': {
        'classification': DynamicSelection,
    },
    "stacking": {
        "classification": StackingClassifier,
        "regression": StackingRegressor,
    }
}

_available_models_reset = deepcopy(_available_models)


def list_models():
    """List all the available model names

    Returns
    -------
    out : list(str)
        A list will all the available model names.

    """
    out = list(_available_models.keys())
    return out


def get_model(name, problem_type, **kwargs):
    """Get a model

    Parameters
    ----------
    name : str
        The model name
    problem_type : str
        The type of problem. See :func:`.run_cross_validation`.

    Returns
    -------
    out : scikit-learn compatible model
        The model object.

    """
    if name not in _available_models:
        raise_error(
            f'The specified model ({name}) is not available. '
            f'Valid options are: {list(_available_models.keys())}')

    if problem_type not in _available_models[name]:
        raise_error(
            f'The specified model ({name})) is not suitable for'
            f'{problem_type}')

    out = _available_models[name][problem_type](**kwargs)
    return out


def register_model(model_name,
                   classification_cls=None,
                   regression_cls=None,
                   overwrite=None
                   ):
    """Register a model to julearn.
    This function allows you to add a model or models for different
    problem_types to julearn.
    Afterwards, it behaves like every other julearn model and can
    be referred to by name. E.g. you can use inside of
     `run_cross_validation` unsig `model=model_name`.

    Parameters
    ----------
    model_name : str
        Name by which model will be referenced by
    classification_cls : object
        The class which will be used for
         classification problem_type.
    regression_cls : str
        The class which will be used for
         regression problem_type.
    overwrite : bool | None, optional
        decides whether overwrite should be allowed, by default None.
        Options are:

        * None : overwrite is possible, but warns the user
        * True : overwrite is possible without any warning
        * False : overwrite is not possible, error is raised instead

    """
    problem_types = [
        "classification",
        "regression"
    ]
    for cls, problem_type in zip(
            [classification_cls, regression_cls],
            problem_types):
        if cls is not None:
            if _available_models.get(model_name) is not None:
                if _available_models.get(model_name).get(problem_type):
                    if overwrite is None:
                        warn(
                            f'Model named {model_name} with'
                            ' problem type {problem_type}'
                            ' already exists. '
                            f'Therefore, {model_name} will be overwritten. '
                            'To remove this warning set overwrite=True. '
                            "If you won't to reset this use "
                            '`julearn.estimators.reset_model_register`.'
                        )
                    elif overwrite is False:
                        raise_error(

                            f'Model named {model_name} with '
                            'problem type {problem_type}'
                            ' already exists. '
                            f'Therefore, {model_name} will be overwritten. '
                            'overwrite is set to False, '
                            'therefore you cannot overwrite '
                            'existing models. Set overwrite=True'
                            ' in case you want to '
                            'overwrite existing models'
                        )

                    logger.info(f'registering model named {model_name} '
                                f'with problem_type {problem_type}'
                                )

                _available_models[model_name][problem_type] = cls
            else:

                logger.info(f'registering model named {model_name} '
                            f'with problem_type {problem_type}'
                            )
                _available_models[model_name] = {problem_type: cls}


def reset_model_register():
    global _available_models
    _available_models = deepcopy(_available_models_reset)
    return _available_models