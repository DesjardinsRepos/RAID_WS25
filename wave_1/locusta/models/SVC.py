import sklearn
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING
from abc import ABC, ABCMeta, abstractmethod

class ClassifierMixin(ABC, metaclass=InputFilter):
    """
    Mixin abstract base class defining functionality for classifiers.
    """

    estimator_params = ["nb_classes"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self._nb_classes: int = -1

    @property
    def nb_classes(self) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        return self._nb_classes  # type: ignore

    @nb_classes.setter
    def nb_classes(self, nb_classes: int):
        """
        Set the number of output classes.
        """
        if nb_classes is None or nb_classes < 2:
            raise ValueError("nb_classes must be greater than or equal to 2.")

        self._nb_classes = nb_classes

class ScikitlearnEstimator(BaseEstimator):
    """
    Estimator class for scikit-learn models.
    """

    def _get_input_shape(self, model) -> Optional[Tuple[int, ...]]:
        _input_shape: Optional[Tuple[int, ...]]
        if hasattr(model, "n_features_"):
            _input_shape = (model.n_features_,)
        elif hasattr(model, "n_features_in_"):
            _input_shape = (model.n_features_in_,)
        elif hasattr(model, "feature_importances_"):
            _input_shape = (len(model.feature_importances_),)
        elif hasattr(model, "coef_"):
            if len(model.coef_.shape) == 1:
                _input_shape = (model.coef_.shape[0],)
            else:
                _input_shape = (model.coef_.shape[1],)
        elif hasattr(model, "support_vectors_"):
            _input_shape = (model.support_vectors_.shape[1],)
        elif hasattr(model, "steps"):
            _input_shape = self._get_input_shape(model.steps[0][1])
        else:
            logger.warning("Input shape not recognised. The model might not have been fitted.")
            _input_shape = None
        return _input_shape
        
def SklearnClassifier(
    model: "sklearn.base.BaseEstimator",
    clip_values: None,
    use_logits: bool = False,
) -> "ScikitlearnClassifier":
    """
    Create a `Classifier` instance from a scikit-learn Classifier model. This is a convenience function that
    instantiates the correct class for the given scikit-learn model.

    :param model: scikit-learn Classifier model.
    :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
            for features.
    """

    # This basic class at least generically handles `fit`, `predict` and `save`
    return ScikitlearnClassifier(
        model,
        clip_values,
        use_logits,
    )



class ScikitlearnClassifier(ClassifierMixin, ScikitlearnEstimator):
    """
    Class for scikit-learn classifier models.
    """

    estimator_params = ClassifierMixin.estimator_params + ScikitlearnEstimator.estimator_params + ["use_logits"]

    def __init__(
        self,
        model: "sklearn.base.BaseEstimator",
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        use_logits: bool = False,
    ) -> None:
        """
        Create a `Classifier` instance from a scikit-learn classifier model.

        :param model: scikit-learn classifier model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param use_logits: Determines whether predict() returns logits instead of probabilities if available. Some
               adversarial attacks (DeepFool) may perform better if logits are used.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
        )
        self._input_shape = self._get_input_shape(model)
        nb_classes = self._get_nb_classes()
        if nb_classes is not None:
            self.nb_classes = nb_classes
        self._use_logits = use_logits

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def use_logits(self) -> bool:
        """
        Return the Boolean for using logits.

        :return: Boolean for using logits.
        """
        return self._use_logits  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` classifier and will be passed to this function as such.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self.model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.model)
        self.nb_classes = self._get_nb_classes()

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :raises `ValueError`: If the classifier does not have methods `predict` or `predict_proba`.
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if self._use_logits:
            if callable(getattr(self.model, "predict_log_proba", None)):
                y_pred = self.model.predict_log_proba(x_preprocessed)
            else:  # pragma: no cover
                raise ValueError(
                    "Argument `use_logits` was True but classifier model does not have callable" "`predict_log_proba`."
                )
        elif callable(getattr(self.model, "predict_proba", None)):
            y_pred = self.model.predict_proba(x_preprocessed)
        elif callable(getattr(self.model, "predict", None)):
            y_pred = to_categorical(
                self.model.predict(x_preprocessed),
                nb_classes=self._get_nb_classes(),
            )
        else:  # pragma: no cover
            raise ValueError("The provided model does not have methods `predict_proba` or `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):  # pragma: no cover
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def clone_for_refitting(self) -> "ScikitlearnClassifier":
        """
        Create a copy of the classifier that can be refit from scratch.

        :return: new estimator
        """
        import sklearn

        clone = type(self)(sklearn.base.clone(self.model))
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.

        """
        # No need to do anything since scikitlearn models start from scratch each time fit() is called
        pass

    def _get_nb_classes(self) -> int:
        if hasattr(self.model, "n_classes_"):
            _nb_classes = self.model.n_classes_
        elif hasattr(self.model, "classes_"):
            _nb_classes = self.model.classes_.shape[0]
        else:
            logger.warning("Number of classes not recognised. The model might not have been fitted.")
            _nb_classes = None
        return _nb_classes

class ScikitlearnSVC(ClassGradientsMixin, LossGradientsMixin, ScikitlearnClassifier):
    """
    Class for scikit-learn C-Support Vector Classification models.
    """

    def __init__(
        self,
        model: Union["sklearn.svm.SVC", "sklearn.svm.LinearSVC"],
        clip_values: None,
    ) -> None:
        """
        Create a `Classifier` instance from a scikit-learn C-Support Vector Classification model.

        :param model: scikit-learn C-Support Vector Classification model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        """
        # pylint: disable=E0001
        import sklearn

        if not isinstance(model, sklearn.svm.SVC) and not isinstance(model, sklearn.svm.LinearSVC):
            raise TypeError(f"Model must be of type sklearn.svm.SVC or sklearn.svm.LinearSVC. Found type {type(model)}")

        super().__init__(
            model=model,
            clip_values=clip_values,
        )
        self._kernel = self._kernel_func()

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        # pylint: disable=E0001

        # Apply preprocessing
        

        num_samples, _ = x.shape

        if isinstance(self.model, sklearn.svm.SVC):
            if self.model.fit_status_:  # pragma: no cover
                raise AssertionError("Model has not been fitted correctly.")

            support_indices = [0] + list(np.cumsum(self.model.n_support_))

            if self.nb_classes == 2:
                sign_multiplier = -1
            else:
                sign_multiplier = 1

            if label is None:
                gradients = np.zeros(
                    (
                        x.shape[0],
                        self.nb_classes,
                        x.shape[1],
                    )
                )

                for i_label in range(self.nb_classes):  # type: ignore
                    for i_sample in range(num_samples):
                        for not_label in range(self.nb_classes):  # type: ignore
                            if i_label != not_label:
                                if not_label < i_label:
                                    label_multiplier = -1
                                else:
                                    label_multiplier = 1

                                for label_sv in range(
                                    support_indices[i_label],
                                    support_indices[i_label + 1],
                                ):
                                    alpha_i_k_y_i = self.model.dual_coef_[
                                        not_label if not_label < i_label else not_label - 1,
                                        label_sv,
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(label_sv, x[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

                                for not_label_sv in range(
                                    support_indices[not_label],
                                    support_indices[not_label + 1],
                                ):
                                    alpha_i_k_y_i = self.model.dual_coef_[
                                        i_label if i_label < not_label else i_label - 1,
                                        not_label_sv,
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif isinstance(label, int):
                gradients = np.zeros((x.shape[0], 1, x.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes):  # type: ignore
                        if label != not_label:
                            if not_label < label:
                                label_multiplier = -1
                            else:
                                label_multiplier = 1

                            for label_sv in range(support_indices[label], support_indices[label + 1]):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    not_label if not_label < label else not_label - 1,
                                    label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(
                                support_indices[not_label],
                                support_indices[not_label + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    label if label < not_label else label - 1,
                                    not_label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x.shape[0], 1, x.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes):  # type: ignore
                        if label[i_sample] != not_label:
                            if not_label < label[i_sample]:
                                label_multiplier = -1
                            else:
                                label_multiplier = 1

                            for label_sv in range(
                                support_indices[label[i_sample]],
                                support_indices[label[i_sample] + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    not_label if not_label < label[i_sample] else not_label - 1,
                                    label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(
                                support_indices[not_label],
                                support_indices[not_label + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    label[i_sample] if label[i_sample] < not_label else label[i_sample] - 1,
                                    not_label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            # gradients = self._apply_preprocessing_gradient(x, gradients * sign_multiplier)
            gradients=gradients*sign_multiplier

        elif isinstance(self.model, sklearn.svm.LinearSVC):
            if label is None:
                gradients = np.zeros(
                    (
                        x.shape[0],
                        self.nb_classes,
                        x.shape[1],
                    )
                )

                for i in range(self.nb_classes):  # type: ignore
                    for i_sample in range(num_samples):
                        if self.nb_classes == 2:
                            gradients[i_sample, i] = self.model.coef_[0] * (2 * i - 1)
                        else:
                            gradients[i_sample, i] = self.model.coef_[i]

            elif isinstance(label, int):
                gradients = np.zeros((x.shape[0], 1, x.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes == 2:
                        gradients[i_sample, 0] = self.model.coef_[0] * (2 * label - 1)
                    else:
                        gradients[i_sample, 0] = self.model.coef_[label]

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x.shape[0], 1, x.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes == 2:
                        gradients[i_sample, 0] = self.model.coef_[0] * (2 * label[i_sample] - 1)
                    else:
                        gradients[i_sample, 0] = self.model.coef_[label[i_sample]]

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            # gradients = self._apply_preprocessing_gradient(x, gradients)

        else:
            raise ValueError("Type of `self.model` not supported for class-gradients.")

        return gradients

    def _kernel_grad(self, sv: np.ndarray, x_sample: np.ndarray) -> np.ndarray:
        """
        Applies the kernel gradient to a support vector.

        :param sv: A support vector.
        :param x_sample: The sample the gradient is taken with respect to.
        :return: the kernel gradient.
        """
        # pylint: disable=W0212
        if self.model.kernel == "linear":
            grad = sv
        elif self.model.kernel == "poly":
            grad = (
                self.model.degree
                * (self.model._gamma * np.sum(x_sample * sv) + self.model.coef0) ** (self.model.degree - 1)
                * sv
            )
        elif self.model.kernel == "rbf":
            grad = (
                2
                * self.model._gamma
                * (-1)
                * np.exp(-self.model._gamma * np.linalg.norm(x_sample - sv, ord=2) ** 2)
                * (x_sample - sv)
            )
        elif self.model.kernel == "sigmoid":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Loss gradients for kernel '{self.model.kernel}' are not implemented.")
        return grad

    def _get_kernel_gradient_sv(self, i_sv: int, x_sample: np.ndarray) -> np.ndarray:
        """
        Applies the kernel gradient to all of a model's support vectors.

        :param i_sv: A support vector index.
        :param x_sample: A sample vector.
        :return: The kernelized product of the vectors.
        """
        x_i = self.model.support_vectors_[i_sv, :]
        return self._kernel_grad(x_i, x_sample)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        Following equation (1) with lambda=0.

        | Paper link: https://pralab.diee.unica.it/sites/default/files/biggio14-svm-chapter.pdf

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: Array of gradients of the same shape as `x`.
        """
        # pylint: disable=E0001
        import sklearn

        # Apply preprocessing
        # x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        num_samples, _ = x.shape
        gradients = np.zeros_like(x)
        y_index = np.argmax(x, axis=1)

        if isinstance(self.model, sklearn.svm.SVC):

            if self.model.fit_status_:
                raise AssertionError("Model has not been fitted correctly.")

            if y.shape[1] == 2:
                sign_multiplier = 1
            else:
                sign_multiplier = -1

            i_not_label_i = None
            label_multiplier = None
            support_indices = [0] + list(np.cumsum(self.model.n_support_))

            for i_sample in range(num_samples):
                i_label = y_index[i_sample]

                for i_not_label in range(self.nb_classes):  # type: ignore
                    if i_label != i_not_label:
                        if i_not_label < i_label:
                            i_not_label_i = i_not_label
                            label_multiplier = -1
                        elif i_not_label > i_label:
                            i_not_label_i = i_not_label - 1
                            label_multiplier = 1

                        for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                            alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_label_sv, x[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

                        for i_not_label_sv in range(
                            support_indices[i_not_label],
                            support_indices[i_not_label + 1],
                        ):
                            alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_not_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

        elif isinstance(self.model, sklearn.svm.LinearSVC):
            for i_sample in range(num_samples):
                i_label = y_index[i_sample]
                if self.nb_classes == 2:
                    i_label_i = 0
                    if i_label == 0:
                        label_multiplier = 1
                    elif i_label == 1:
                        label_multiplier = -1
                    else:
                        raise ValueError("Label index not recognized because it is not 0 or 1.")
                else:
                    i_label_i = i_label
                    label_multiplier = -1

                gradients[i_sample] = label_multiplier * self.model.coef_[i_label_i]
        else:
            raise TypeError("Model not recognized.")

        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

    def _kernel_func(self) -> Callable:
        """
        Return the function for the kernel of this SVM.

        :return: A callable kernel function.
        """
        # pylint: disable=E0001
        import sklearn
        from sklearn.metrics.pairwise import (
            polynomial_kernel,
            linear_kernel,
            rbf_kernel,
        )

        if isinstance(self.model, sklearn.svm.LinearSVC):
            kernel = "linear"
        elif isinstance(self.model, sklearn.svm.SVC):
            kernel = self.model.kernel
        else:
            raise NotImplementedError("SVM model not yet supported.")

        if kernel == "linear":
            kernel_func = linear_kernel
        elif kernel == "poly":
            kernel_func = polynomial_kernel
        elif kernel == "rbf":
            kernel_func = rbf_kernel
        elif callable(kernel):
            kernel_func = kernel
        else:
            raise NotImplementedError(f"Kernel '{kernel}' not yet supported.")

        return kernel_func

    def q_submatrix(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """
        Returns the q submatrix of this SVM indexed by the arrays at rows and columns.

        :param rows: The row vectors.
        :param cols: The column vectors.
        :return: A submatrix of Q.
        """
        submatrix_shape = (rows.shape[0], cols.shape[0])
        y_row = self.model.predict(rows)
        y_col = self.model.predict(cols)
        y_row[y_row == 0] = -1
        y_col[y_col == 0] = -1
        q_rc = np.zeros(submatrix_shape)
        for row in range(q_rc.shape[0]):
            for col in range(q_rc.shape[1]):
                q_rc[row][col] = self._kernel([rows[row]], [cols[col]])[0][0] * y_row[row] * y_col[col]

        return q_rc

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # pylint: disable=E0001
        import sklearn

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if isinstance(self.model, sklearn.svm.SVC) and self.model.probability:
            y_pred = self.model.predict_proba(X=x_preprocessed)
        else:
            y_pred_label = self.model.predict(X=x_preprocessed)
            targets = np.array(y_pred_label).reshape(-1)
            one_hot_targets = np.eye(self.nb_classes)[targets]
            y_pred = one_hot_targets

        return y_pred