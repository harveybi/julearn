# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
# TODO adjust license

from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils.validation import check_memory
from sklearn.utils import _print_elapsed_time
from sklearn.base import clone
from ..transformers import check_transformer


class FeatureTypedPipeline(Pipeline):

    def fit(self, X, y=None, X_names=None, X_types=None, **fit_params):

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y,
                       X_names=X_names, X_types=X_types,
                       **fit_params_steps)
        with _print_elapsed_time("Pipeline",
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def _fit(self, X, y=None, X_names=None, X_types=None, **fit_params_steps):
        self.X_names_ = X_names
        self.X_types_ = X_types
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        next_feature_names_in = self.X_names_
        next_feature_types_in = self.X_types_
        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time(
                        "Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location"):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, "cachedir"):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)

            cloned_transformer = check_transformer(cloned_transformer)

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                X_names=next_feature_names_in,
                X_types=next_feature_types_in,

                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
            next_feature_names_in = fitted_transformer.get_feature_names_out()
            next_feature_types_in = fitted_transformer.get_feature_types_out()

        return X

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {name: {}
                            for name, step in self.steps if step is not None}
        for pname, pval in fit_params.items():
            if pname in ["X_types", "X_names"]:
                fit_params_steps[pname] = pval
                continue
            if "__" not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname)
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps
