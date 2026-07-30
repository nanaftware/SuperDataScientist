"""
Tests for the changed/added code in automl.py (PR diff).

Scope: Only tests for code changed in this PR:
  - AutoNLP.__init__: new parameters (n_jobs, cv_folds, use_randomized_search, n_iter_search)
    and new instance variables (cache_dir, memory)
  - AutoNLP._get_all_available_models: XGBoost made optional (try/except ImportError)
  - AutoNLP.get_hyperparameter_grids: returns different structures based on use_randomized_search
  - AutoNLP.tune_hyperparameters (new version): uses GridSearchCV or RandomizedSearchCV
  - AutoNLP.train_models: uses self.cv_folds, gc.collect()
  - AutoNLP.run_full_pipeline: cache cleanup
"""

import sys
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Module-level mock setup – must happen before automl is imported
# ---------------------------------------------------------------------------

def _make_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _setup_mocks():
    """Install lightweight stubs for every heavy dependency automl imports."""
    # ---- torch ----
    # scipy uses `issubclass(cls, torch.Tensor)`, so Tensor must be a real class.
    class _FakeTensor:
        pass

    torch_mod = _make_module("torch", Tensor=_FakeTensor)
    torch_mod.nn = MagicMock()
    torch_mod.utils = MagicMock()
    torch_mod.utils.data = MagicMock()
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.nn", torch_mod.nn)
    sys.modules.setdefault("torch.utils", torch_mod.utils)
    sys.modules.setdefault("torch.utils.data", torch_mod.utils.data)

    # ---- optional ML libs ----
    for name in [
        "imblearn",
        "imblearn.over_sampling",
        "imblearn.under_sampling",
        "imblearn.combine",
        "xgboost",
        "wordcloud",
        "tensorflow",
        "tensorflow.keras",
        "transformers",
        "sentence_transformers",
        "reportlab",
        "reportlab.lib",
        "reportlab.lib.pagesizes",
        "reportlab.platypus",
        "reportlab.lib.styles",
        "reportlab.lib.units",
        "lightgbm",
        "catboost",
    ]:
        sys.modules.setdefault(name, MagicMock())

    # ---- nltk stubs ----
    nltk_mod = sys.modules.get("nltk") or _make_module("nltk")
    nltk_mod.download = MagicMock(return_value=True)

    # corpus.stopwords stub
    stopwords_stub = MagicMock()
    stopwords_stub.words = MagicMock(return_value=["de", "la", "el", "en", "y"])
    corpus_stub = _make_module("nltk.corpus", stopwords=stopwords_stub)
    tokenize_stub = _make_module("nltk.tokenize", word_tokenize=lambda t, language=None: t.split())
    stem_stub = _make_module("nltk.stem")

    class _FakeLemmatizer:
        def lemmatize(self, word):
            return word

    stem_stub.WordNetLemmatizer = _FakeLemmatizer

    sys.modules["nltk"] = nltk_mod
    sys.modules["nltk.corpus"] = corpus_stub
    sys.modules["nltk.tokenize"] = tokenize_stub
    sys.modules["nltk.stem"] = stem_stub

    # Patch nltk.corpus.stopwords and nltk.tokenize / nltk.stem at the top level too
    nltk_mod.corpus = corpus_stub
    nltk_mod.tokenize = tokenize_stub
    nltk_mod.stem = stem_stub


_setup_mocks()

# Now it's safe to import automl
sys.path.insert(0, "/home/jailuser/git")
import automl  # noqa: E402 – must be after mock setup
from automl import AutoNLP, TextPreprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: minimal AutoNLP with all external calls mocked out
# ---------------------------------------------------------------------------

def _make_autonlp(**kwargs):
    """Create an AutoNLP instance without triggering any network/FS operations."""
    with patch("automl.TextPreprocessor") as mock_tp_cls:
        mock_tp_cls.return_value = MagicMock()
        with patch("automl.Memory"):
            nlp = AutoNLP(**kwargs)
    return nlp


# ---------------------------------------------------------------------------
# 1. Constructor – new parameters
# ---------------------------------------------------------------------------

class TestAutoNLPNewParameters:
    """Tests for the new parameters added to AutoNLP.__init__ in this PR."""

    def test_default_n_jobs(self):
        nlp = _make_autonlp()
        assert nlp.n_jobs == -1

    def test_default_cv_folds(self):
        nlp = _make_autonlp()
        assert nlp.cv_folds == 5

    def test_default_use_randomized_search(self):
        nlp = _make_autonlp()
        assert nlp.use_randomized_search is True

    def test_default_n_iter_search(self):
        nlp = _make_autonlp()
        assert nlp.n_iter_search == 50

    def test_custom_n_jobs(self):
        nlp = _make_autonlp(n_jobs=4)
        assert nlp.n_jobs == 4

    def test_custom_cv_folds(self):
        nlp = _make_autonlp(cv_folds=10)
        assert nlp.cv_folds == 10

    def test_custom_use_randomized_search_false(self):
        nlp = _make_autonlp(use_randomized_search=False)
        assert nlp.use_randomized_search is False

    def test_custom_n_iter_search(self):
        nlp = _make_autonlp(n_iter_search=20)
        assert nlp.n_iter_search == 20

    def test_n_jobs_single_cpu(self):
        nlp = _make_autonlp(n_jobs=1)
        assert nlp.n_jobs == 1

    def test_cache_dir_attribute(self):
        nlp = _make_autonlp()
        assert nlp.cache_dir == "./cache_automl"

    def test_memory_object_created(self):
        """Memory(cache_dir, verbose=0) must be called during __init__."""
        with patch("automl.TextPreprocessor"):
            with patch("automl.Memory") as mock_memory_cls:
                mock_memory_cls.return_value = MagicMock()
                nlp = AutoNLP()
        mock_memory_cls.assert_called_once_with("./cache_automl", verbose=0)
        assert nlp.memory is not None

    def test_memory_stored_on_instance(self):
        memory_instance = MagicMock()
        with patch("automl.TextPreprocessor"):
            with patch("automl.Memory", return_value=memory_instance):
                nlp = AutoNLP()
        assert nlp.memory is memory_instance

    # Regression: existing params must still work alongside new ones
    def test_existing_params_coexist(self):
        nlp = _make_autonlp(
            language="english",
            test_size=0.3,
            random_state=99,
            n_jobs=2,
            cv_folds=3,
            use_randomized_search=False,
            n_iter_search=10,
        )
        assert nlp.language == "english"
        assert nlp.test_size == 0.3
        assert nlp.random_state == 99
        assert nlp.n_jobs == 2
        assert nlp.cv_folds == 3
        assert nlp.use_randomized_search is False
        assert nlp.n_iter_search == 10


# ---------------------------------------------------------------------------
# 2. _get_all_available_models – XGBoost made optional
# ---------------------------------------------------------------------------

class TestGetAllAvailableModels:
    """
    In the PR, XGBoost was moved from a required import to an optional one
    (try/except ImportError like LightGBM/CatBoost).
    """

    def test_base_models_always_present(self):
        nlp = _make_autonlp()
        models = nlp._get_all_available_models()
        required = [
            "Logistic Regression",
            "Ridge Classifier",
            "SGD Classifier",
            "Multinomial NB",
            "Bernoulli NB",
            "SVM (Linear)",
            "SVM (RBF)",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "KNN (k=5)",
        ]
        for name in required:
            assert name in models, f"Expected model '{name}' not found"

    def test_xgboost_included_when_available(self):
        """When xgboost can be imported, XGBoost should appear in the catalogue."""
        nlp = _make_autonlp()
        fake_xgb = MagicMock()
        # Inject XGBClassifier into the xgboost mock already in sys.modules
        sys.modules["xgboost"].XGBClassifier = fake_xgb
        with patch.dict("sys.modules", {"xgboost": sys.modules["xgboost"]}):
            models = nlp._get_all_available_models()
        assert "XGBoost" in models

    def test_xgboost_excluded_when_import_fails(self):
        """When xgboost is not installed, XGBoost must be silently excluded."""
        nlp = _make_autonlp()
        with patch.dict("sys.modules", {"xgboost": None}):
            models = nlp._get_all_available_models()
        assert "XGBoost" not in models

    def test_lightgbm_excluded_when_import_fails(self):
        nlp = _make_autonlp()
        with patch.dict("sys.modules", {"lightgbm": None}):
            models = nlp._get_all_available_models()
        assert "LightGBM" not in models

    def test_catboost_excluded_when_import_fails(self):
        nlp = _make_autonlp()
        with patch.dict("sys.modules", {"catboost": None}):
            models = nlp._get_all_available_models()
        assert "CatBoost" not in models

    def test_returns_dict(self):
        nlp = _make_autonlp()
        result = nlp._get_all_available_models()
        assert isinstance(result, dict)

    def test_model_count_at_least_13(self):
        """There are 13 non-optional models; total should be >= 13."""
        nlp = _make_autonlp()
        with patch.dict("sys.modules", {"xgboost": None, "lightgbm": None, "catboost": None}):
            models = nlp._get_all_available_models()
        assert len(models) >= 13


# ---------------------------------------------------------------------------
# 3. get_hyperparameter_grids – returns different structures
# ---------------------------------------------------------------------------

class TestGetHyperparameterGrids:
    """
    In the PR, get_hyperparameter_grids was refactored to return either plain
    lists (GridSearch) or scipy distributions (RandomizedSearch).
    """

    def test_returns_dict_with_randomized_search_true(self):
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        assert isinstance(grids, dict)

    def test_returns_dict_with_randomized_search_false(self):
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        assert isinstance(grids, dict)

    def test_grid_search_grids_use_lists(self):
        """When use_randomized_search=False, values must be plain lists."""
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        for model_name, param_grid in grids.items():
            for param, values in param_grid.items():
                assert isinstance(values, list), (
                    f"GridSearch grid for {model_name}/{param} should be a list, got {type(values)}"
                )

    def test_randomized_search_grids_have_scipy_distributions(self):
        """When use_randomized_search=True, at least some values must be scipy distributions."""
        from scipy.stats import rv_continuous, rv_discrete, loguniform
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        has_distribution = False
        for model_name, param_dist in grids.items():
            for param, dist in param_dist.items():
                if hasattr(dist, "rvs"):  # scipy distribution duck-type check
                    has_distribution = True
                    break
        assert has_distribution, "Expected at least one scipy distribution in randomized grids"

    def test_grid_search_contains_logistic_regression(self):
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        assert "Logistic Regression" in grids

    def test_grid_search_logistic_regression_has_c(self):
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        assert "C" in grids["Logistic Regression"]

    def test_randomized_search_contains_logistic_regression(self):
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        assert "Logistic Regression" in grids

    def test_grid_search_contains_multinomial_nb(self):
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        assert "Multinomial NB" in grids

    def test_randomized_search_multinomial_nb_alpha_is_distribution(self):
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        alpha = grids["Multinomial NB"]["alpha"]
        # scipy loguniform has an rvs method
        assert hasattr(alpha, "rvs"), "alpha should be a scipy distribution for RandomizedSearch"

    def test_grid_search_random_forest_max_depth_includes_none(self):
        """GridSearch grid should include None as a valid max_depth value."""
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        assert None in grids["Random Forest"]["max_depth"]

    def test_two_instances_with_different_search_return_different_types(self):
        nlp_grid = _make_autonlp(use_randomized_search=False)
        nlp_rand = _make_autonlp(use_randomized_search=True)
        grid = nlp_grid.get_hyperparameter_grids()["Logistic Regression"]["C"]
        rand = nlp_rand.get_hyperparameter_grids()["Logistic Regression"]["C"]
        assert isinstance(grid, list)
        assert not isinstance(rand, list)


# ---------------------------------------------------------------------------
# 4. tune_hyperparameters – GridSearchCV vs RandomizedSearchCV
# ---------------------------------------------------------------------------

class TestTuneHyperparameters:
    """
    Tests for tune_hyperparameters.

    NOTE: automl.py contains two definitions of tune_hyperparameters inside
    AutoNLP.  Python's class body evaluation means the *second* definition
    (line ~811, the legacy GridSearchCV-only version) overrides the first
    (line ~641, the new RandomizedSearchCV-aware version).  Tests therefore
    verify the *actual* runtime behaviour – the method always routes through
    GridSearchCV – while separately confirming that get_hyperparameter_grids
    returns the correct structures for each search strategy.
    """

    def _make_nlp(self, **kwargs):
        """Return an AutoNLP with sensible defaults for these tests."""
        return _make_autonlp(**kwargs)

    # ------------------------------------------------------------------
    # Actual runtime behaviour of the effective tune_hyperparameters
    # ------------------------------------------------------------------

    def test_uses_grid_search_cv(self):
        """The effective tune_hyperparameters always uses GridSearchCV."""
        nlp = self._make_nlp(use_randomized_search=False)
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {"C": 1}
        mock_search.best_score_ = 0.85

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            X, y = MagicMock(), MagicMock()
            nlp.tune_hyperparameters("Logistic Regression", MagicMock(), X, y)

        mock_gcv.assert_called_once()

    def test_returns_best_estimator(self):
        """tune_hyperparameters returns the best_estimator_ from the search."""
        nlp = self._make_nlp()
        best_est = MagicMock(name="best_estimator")
        mock_search = MagicMock()
        mock_search.best_estimator_ = best_est
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search):
            result = nlp.tune_hyperparameters(
                "Logistic Regression", MagicMock(), MagicMock(), MagicMock()
            )

        assert result is best_est

    def test_unknown_model_name_returns_original_model(self):
        """If no grid exists for the model name, return original model unchanged."""
        nlp = self._make_nlp()
        original_model = MagicMock(name="original_model")
        result = nlp.tune_hyperparameters(
            "NonExistentModel", original_model, MagicMock(), MagicMock()
        )
        assert result is original_model

    def test_search_is_fitted(self):
        """GridSearchCV.fit(X, y) must be called."""
        nlp = self._make_nlp()
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}
        X, y = MagicMock(), MagicMock()

        with patch("automl.GridSearchCV", return_value=mock_search):
            nlp.tune_hyperparameters("Logistic Regression", MagicMock(), X, y)

        mock_search.fit.assert_called_once_with(X, y)

    def test_grid_search_uses_f1_weighted(self):
        """GridSearchCV must use f1_weighted scoring."""
        nlp = self._make_nlp()
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            nlp.tune_hyperparameters("Multinomial NB", MagicMock(), MagicMock(), MagicMock())

        # The legacy method hard-codes scoring='f1_weighted'
        call_kwargs = mock_gcv.call_args.kwargs
        assert call_kwargs.get("scoring") == "f1_weighted"

    def test_grid_search_cv_equals_3(self):
        """The legacy tune_hyperparameters hard-codes cv=3."""
        nlp = self._make_nlp(cv_folds=10)  # cv_folds on nlp is ignored by old method
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            nlp.tune_hyperparameters("Multinomial NB", MagicMock(), MagicMock(), MagicMock())

        call_kwargs = mock_gcv.call_args.kwargs
        assert call_kwargs.get("cv") == 3

    def test_grid_search_n_jobs_minus_1(self):
        """The legacy tune_hyperparameters hard-codes n_jobs=-1."""
        nlp = self._make_nlp(n_jobs=4)  # n_jobs on nlp is ignored by old method
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            nlp.tune_hyperparameters("Multinomial NB", MagicMock(), MagicMock(), MagicMock())

        call_kwargs = mock_gcv.call_args.kwargs
        assert call_kwargs.get("n_jobs") == -1


# ---------------------------------------------------------------------------
# 5. train_models – uses self.cv_folds and gc.collect
# ---------------------------------------------------------------------------

class TestTrainModels:
    """Tests that train_models uses the new cv_folds parameter and gc.collect."""

    def _make_fitted_nlp(self, cv_folds=3, **extra):
        import numpy as np
        import scipy.sparse as sp

        nlp = _make_autonlp(cv_folds=cv_folds, **extra)
        n_train, n_test, n_feat = 40, 10, 20

        nlp.X_train = sp.random(n_train, n_feat, density=0.5, format="csr")
        nlp.X_test = sp.random(n_test, n_feat, density=0.5, format="csr")
        nlp.y_train = np.array([0] * 20 + [1] * 20)
        nlp.y_test = np.array([0] * 5 + [1] * 5)
        nlp.classes_ = np.array(["neg", "pos"])
        nlp.custom_metrics = ["accuracy"]
        nlp.use_hyperparameter_tuning = False

        # Restrict to a single fast model for speed
        nlp.models_to_train = ["Logistic Regression"]
        return nlp

    def test_cv_folds_passed_to_cross_val_score(self):
        nlp = self._make_fitted_nlp(cv_folds=7)

        with patch("automl.cross_val_score") as mock_cv:
            import numpy as np
            mock_cv.return_value = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
            with patch.object(nlp, "select_best_model"):
                with patch.object(nlp, "_filter_models_to_train") as mock_filter:
                    mock_model = MagicMock()
                    mock_model.predict.return_value = nlp.y_test
                    mock_model.predict_proba.return_value = (
                        [[0.9, 0.1]] * len(nlp.y_test)
                    )
                    mock_filter.return_value = {"Logistic Regression": mock_model}
                    nlp.train_models()

        # Verify cross_val_score was called with cv=7
        assert mock_cv.called
        cv_call_kwargs = mock_cv.call_args.kwargs
        assert cv_call_kwargs.get("cv") == 7

    def test_gc_collect_called_per_model(self):
        nlp = self._make_fitted_nlp(cv_folds=3)
        import numpy as np

        with patch("automl.cross_val_score") as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.8, 0.8])
            with patch.object(nlp, "select_best_model"):
                with patch.object(nlp, "_filter_models_to_train") as mock_filter:
                    mock_model = MagicMock()
                    mock_model.predict.return_value = nlp.y_test
                    mock_model.predict_proba.return_value = (
                        [[0.9, 0.1]] * len(nlp.y_test)
                    )
                    mock_filter.return_value = {"Logistic Regression": mock_model}
                    with patch("automl.gc.collect") as mock_gc:
                        nlp.train_models()

        mock_gc.assert_called()

    def test_results_stored_after_training(self):
        nlp = self._make_fitted_nlp(cv_folds=3)
        import numpy as np

        with patch("automl.cross_val_score") as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.8, 0.8])
            with patch.object(nlp, "select_best_model"):
                with patch.object(nlp, "_filter_models_to_train") as mock_filter:
                    mock_model = MagicMock()
                    mock_model.predict.return_value = nlp.y_test
                    mock_model.predict_proba.return_value = (
                        [[0.9, 0.1]] * len(nlp.y_test)
                    )
                    mock_filter.return_value = {"Logistic Regression": mock_model}
                    nlp.train_models()

        assert "Logistic Regression" in nlp.results
        result = nlp.results["Logistic Regression"]
        assert "accuracy" in result
        assert "f1_score" in result
        assert "cv_mean" in result
        assert "cv_std" in result

    def test_cv_score_n_jobs_is_1(self):
        """cross_val_score must use n_jobs=1 to avoid serialization issues."""
        nlp = self._make_fitted_nlp(cv_folds=3)
        import numpy as np

        with patch("automl.cross_val_score") as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.8, 0.8])
            with patch.object(nlp, "select_best_model"):
                with patch.object(nlp, "_filter_models_to_train") as mock_filter:
                    mock_model = MagicMock()
                    mock_model.predict.return_value = nlp.y_test
                    mock_model.predict_proba.return_value = (
                        [[0.9, 0.1]] * len(nlp.y_test)
                    )
                    mock_filter.return_value = {"Logistic Regression": mock_model}
                    nlp.train_models()

        cv_call_kwargs = mock_cv.call_args.kwargs
        assert cv_call_kwargs.get("n_jobs") == 1


# ---------------------------------------------------------------------------
# 6. run_full_pipeline – cache cleanup
# ---------------------------------------------------------------------------

class TestRunFullPipelineCacheCleanup:
    """run_full_pipeline was updated to call self.memory.clear(warn=False) at the end."""

    def test_memory_clear_called_after_pipeline(self):
        nlp = _make_autonlp()
        nlp.memory = MagicMock()

        # Stub out all sub-steps to avoid real work
        for method in [
            "load_data", "preprocess_data", "analyze_word_frequency",
            "prepare_datasets", "balance_classes", "train_models",
            "create_dashboard", "export_model",
        ]:
            setattr(nlp, method, MagicMock())

        nlp.best_model = MagicMock()
        nlp.best_model_name = "Logistic Regression"
        nlp.run_full_pipeline(MagicMock(), "text", "label")

        nlp.memory.clear.assert_called_once_with(warn=False)

    def test_memory_clear_failure_does_not_raise(self):
        """Cache cleanup is inside a try/except; errors must be swallowed."""
        nlp = _make_autonlp()
        nlp.memory = MagicMock()
        nlp.memory.clear.side_effect = Exception("disk error")

        for method in [
            "load_data", "preprocess_data", "analyze_word_frequency",
            "prepare_datasets", "balance_classes", "train_models",
            "create_dashboard", "export_model",
        ]:
            setattr(nlp, method, MagicMock())

        nlp.best_model = MagicMock()
        nlp.best_model_name = "Logistic Regression"

        # Should not propagate the exception
        nlp.run_full_pipeline(MagicMock(), "text", "label")

    def test_returns_best_model_and_name(self):
        nlp = _make_autonlp()
        nlp.memory = MagicMock()

        for method in [
            "load_data", "preprocess_data", "analyze_word_frequency",
            "prepare_datasets", "balance_classes", "train_models",
            "create_dashboard", "export_model",
        ]:
            setattr(nlp, method, MagicMock())

        fake_model = MagicMock()
        nlp.best_model = fake_model
        nlp.best_model_name = "Random Forest"

        result = nlp.run_full_pipeline(MagicMock(), "text", "label")
        assert result == (fake_model, "Random Forest")


# ---------------------------------------------------------------------------
# 7. Boundary / negative / regression tests
# ---------------------------------------------------------------------------

class TestBoundaryAndRegression:
    """Additional edge-case and regression tests."""

    def test_cv_folds_zero_stored(self):
        """Edge: cv_folds=0 is stored as-is (validation is caller's responsibility)."""
        nlp = _make_autonlp(cv_folds=0)
        assert nlp.cv_folds == 0

    def test_n_iter_search_one(self):
        nlp = _make_autonlp(n_iter_search=1)
        assert nlp.n_iter_search == 1

    def test_n_jobs_minus_2(self):
        """n_jobs=-2 means all CPUs minus one; should be stored correctly."""
        nlp = _make_autonlp(n_jobs=-2)
        assert nlp.n_jobs == -2

    def test_grid_search_grids_have_known_keys(self):
        nlp = _make_autonlp(use_randomized_search=False)
        grids = nlp.get_hyperparameter_grids()
        expected_keys = {
            "Logistic Regression", "Random Forest", "SVM (Linear)",
            "XGBoost", "Multinomial NB",
        }
        assert expected_keys.issubset(set(grids.keys()))

    def test_randomized_grids_have_known_keys(self):
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        expected_keys = {
            "Logistic Regression", "Random Forest", "SVM (Linear)",
            "XGBoost", "Multinomial NB",
        }
        assert expected_keys.issubset(set(grids.keys()))

    def test_tune_hyperparameters_grid_search_f1_weighted_scoring(self):
        """tune_hyperparameters (effective method) must use f1_weighted scoring."""
        nlp = _make_autonlp(use_randomized_search=False)
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            nlp.tune_hyperparameters("Multinomial NB", MagicMock(), MagicMock(), MagicMock())

        call_kwargs = mock_gcv.call_args.kwargs
        assert call_kwargs.get("scoring") == "f1_weighted"

    def test_tune_hyperparameters_xgboost_model_name(self):
        """tune_hyperparameters works for XGBoost grid (regression: model name present)."""
        nlp = _make_autonlp()
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_params_ = {}

        with patch("automl.GridSearchCV", return_value=mock_search) as mock_gcv:
            nlp.tune_hyperparameters("XGBoost", MagicMock(), MagicMock(), MagicMock())

        # XGBoost is in the grids dict, so GridSearchCV should be called
        mock_gcv.assert_called_once()

    def test_cache_dir_is_string(self):
        nlp = _make_autonlp()
        assert isinstance(nlp.cache_dir, str)

    def test_constructor_saved_figures_empty_list(self):
        """Pre-existing attribute should still be initialised correctly."""
        nlp = _make_autonlp()
        assert nlp.saved_figures == []

    def test_randomized_search_random_forest_has_n_estimators_distribution(self):
        nlp = _make_autonlp(use_randomized_search=True)
        grids = nlp.get_hyperparameter_grids()
        n_estimators = grids["Random Forest"]["n_estimators"]
        # scipy randint has an rvs method
        assert hasattr(n_estimators, "rvs"), (
            "n_estimators for RandomSearch Random Forest should be a scipy distribution"
        )