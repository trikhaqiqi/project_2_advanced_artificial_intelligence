"""Implementasi from scratch semua model Tugas Akhir IF5150 (Kelompok G02).

Isi modul:
  - DecisionTreeClassifier (ID3) dan RandomForestClassifier
  - KNNScratch dan LogisticRegressionScratch, dibungkus ScratchPipeline (preprocessing + SMOTE)
  - Ann (jaringan saraf satu hidden layer)
  - load_model / load_all_models untuk memuat model yang disimpan notebook di saved_models/

Contoh:
    from ml_models import load_model, load_all_models
    knn = load_model("saved_models/knn_scratch.pkl")
    models = load_all_models("saved_models")
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
KOLOM_NOMINAL_SCRATCH = ["Application mode", "Course", "Previous qualification"]
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Decision Tree (ID3) dan Random Forest (bagian 2.1 dan 2.2 notebook)
# ---------------------------------------------------------------------------
class DecisionTreeClassifier():
    def __init__(self, max_depth = None, min_samples_split = None, min_samples_leaf = None, max_features = None):
        self.__features = None
        self.__target = None
        self.__tree = None
        self.__weights = None
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_features = max_features
        self.__thresholds = {}

    def fit(self, features, target, discretize = None, discretization_max_depth = None):
        self.__features = features.reset_index(drop = True)
        self.__target = target.reset_index(drop = True)
        self.__weights = pd.Series(np.ones(len(self.__features)))

        if discretize is not None:
            self.__features = self.__discretize(self.__features, self.__target, self.__weights, discretize, discretization_max_depth)

        if self.__max_depth is not None:
            self.__tree = self.__id3(self.__features, self.__target, self.__weights, 1)
        else:
            self.__tree = self.__id3(self.__features, self.__target, self.__weights)


    def predict(self, features, voting = 'hard'):
        features = self.__transform(features)
        predictions = [self.__tree_traversal(self.__tree, features.iloc[i]) for i in range(len(features))]

        if voting != 'hard':
            return predictions
        else:
            return [max(prediction, key=prediction.get) for prediction in predictions]

    def save_to_pickle(self, model_name = None, save_to = None):
        if model_name is None:
            raise ValueError("Please provide model name through model_name argument. Example: model.pkl")

        model_data = {
            "features": self.__features,
            "target": self.__target,
            "tree": self.__tree,
            "weights": self.__weights,
            "max_depth": self.__max_depth,
            "min_samples_split": self.__min_samples_split,
            "min_samples_leaf": self.__min_samples_leaf,
            "max_features": self.__max_features,
            "thresholds": self.__thresholds
        }
        
        if save_to is None:
            with open(model_name, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            if save_to[-1] == '/':
                with open(save_to + '/' + model_name, 'w') as f:
                    pickle.dump(model_data, f)
            else:
                with open(save_to + model_name, 'w') as f:
                    pickle.dump(model_data, f)

    def load_from_pickle(self, pickle_dir):
        with open(pickle_dir, 'rb') as f:
            model_data = pickle.load(f)

        self.__features = model_data["features"]
        self.__target = model_data["target"]
        self.__tree = model_data["tree"]
        self.__weights = model_data["weights"]
        self.__max_depth = model_data["max_depth"]
        self.__min_samples_split = model_data["min_samples_split"]
        self.__min_samples_leaf = model_data["min_samples_leaf"]
        self.__max_features = model_data["max_features"]
        self.__thresholds = model_data["thresholds"]

    def __entropy(self, features, weights):
        unique_elements = np.unique(features)
        counts = np.array([np.sum(weights[features == value]) for value in unique_elements])

        entropy_value = -np.sum([
            (counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(unique_elements)) if counts[i] > 0])
        
        return entropy_value

    def __information_gain(self, feature, target, weights):
        assert len(feature) == len(target), 'length of feature and target is not matched'

        not_missing_feature = feature[feature.notna()]
        not_missing_target = target[feature.notna()]
        not_missing_weights = weights[feature.notna()]

        if np.sum(not_missing_weights) == 0:
            return 0

        total_entropy = self.__entropy(not_missing_target, not_missing_weights)
        values= np.unique(not_missing_feature)
        counts = np.array([np.sum(not_missing_weights[not_missing_feature == value]) for value in values])
        
        weighted_entropy = np.sum([
            (counts[i] / np.sum(counts)) * self.__entropy(not_missing_target[not_missing_feature == values[i]], 
                                                          not_missing_weights[not_missing_feature == values[i]])
            for i in range(len(values))])

        not_missing_fraction = np.sum(not_missing_weights) / np.sum(weights)

        return not_missing_fraction * (total_entropy - weighted_entropy)

    def __id3(self, features, target, weights, tree_depth = None, parent_node_class=None):
        assert len(features) == len(target), 'length of feature and target is not matched'

        if self.__min_samples_leaf is not None:
            if len(features) < self.__min_samples_leaf:
                return parent_node_class
            
        if (len(features) == 0):
            return parent_node_class

        if len(np.unique(target)) == 1:
            return np.unique(target)[0]

        unique_elements = np.unique(target)
        counts = np.array([np.sum(weights[target == value]) for value in unique_elements])
        
        parent_node_class = unique_elements[np.argmax(counts)]

        if tree_depth is not None:
            if tree_depth >= self.__max_depth:
                return parent_node_class

        if self.__min_samples_split is not None:
            if len(features) < self.__min_samples_split:
                return parent_node_class

        if features.shape[1] == 0:
            return parent_node_class


        if self.__max_features is not None:
            columns = np.random.choice(features.columns, size = min(self.__max_features, len(features.columns)), replace=False)
        else:
            columns = features.columns

        gains = [self.__information_gain(features[column], target, weights) for column in columns]
        if np.max(gains) < 1e-12:
            return parent_node_class
        
        best_feature = columns[np.argmax(gains)]

        tree = {best_feature: {}}
        not_best_feature = [f for f in features.columns if f != best_feature]
        not_missing_feature = features[best_feature].notna()
        missing_feature = features[best_feature].isna()

        values = np.unique(features.loc[not_missing_feature, best_feature])
        counts = np.array([np.sum(weights[not_missing_feature & (features[best_feature] == value)]) for value in values])
        probabilities = counts / np.sum(counts)
        tree["__probabilities__"] = {
            (value.item() if isinstance(value, np.generic) else value): float(probability) 
            for value, probability in zip(values, probabilities)
        }

        for i, value in enumerate(values):
            selected = (features[best_feature] == value) | missing_feature
            
            subset = features[selected][not_best_feature].reset_index(drop = True)
            subset_target =  target.loc[selected].reset_index(drop = True)
            subset_weights = weights.loc[selected].reset_index(drop = True)
            subset_weights.loc[missing_feature[selected].to_numpy()] *= probabilities[i]

            if tree_depth is not None:
                subtree = self.__id3(subset, subset_target, subset_weights, tree_depth + 1, parent_node_class)
            else:
                subtree = self.__id3(subset, subset_target, subset_weights, tree_depth, parent_node_class)
            
            key = value.item() if isinstance(value, np.generic) else value
            tree[best_feature][key] = subtree

        return tree

    def __tree_traversal(self, tree, features):
        if not isinstance(tree, dict):
            return {tree: 1.0}
    
        feature = next(key for key in tree if key != "__probabilities__")
        value = features[feature]

        if pd.notna(value) and value in tree[feature]:
            subtree = tree[feature][value]
            return self.__tree_traversal(subtree, features)

        predictions = {}

        for branch, subtree in tree[feature].items():
            result = self.__tree_traversal(subtree, features)
            probability = tree["__probabilities__"][branch]

            for label, score in result.items():
                predictions[label] = (
                    predictions.get(label, 0.0)
                    + probability * score
                )

        return predictions

    def __find_thresholds(self, feature, target, weights, discretization_max_depth, depth = 0):
        if discretization_max_depth is not None and depth >= discretization_max_depth:
            return []
        
        not_missing_feature = feature[feature.notna()]
        not_missing_target = target[feature.notna()]
        not_missing_weights = weights[feature.notna()]
        
        values = np.sort(not_missing_feature.unique())

        if len(values) < 2 or not_missing_target.nunique() <= 1:
            return []

        candidates = (values[:-1] + values[1:]) / 2

        gains = []

        for threshold in candidates:
            categorical = pd.Series(np.where(not_missing_feature <= threshold, 0, 1), index=not_missing_feature.index)
            gains.append(self.__information_gain(categorical, not_missing_target, not_missing_weights))

        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]

        if best_gain <= 1e-12:
            return []

        threshold = candidates[best_idx]

        left = not_missing_feature <= threshold
        right = ~left

        return (
            self.__find_thresholds(not_missing_feature.loc[left], not_missing_target.loc[left], 
                                   not_missing_weights.loc[left], discretization_max_depth, depth + 1) + [threshold] 
            + self.__find_thresholds(not_missing_feature.loc[right], not_missing_target.loc[right],
                                     not_missing_weights.loc[right], discretization_max_depth, depth + 1)
        )

    def __discretize(self, features, target, weights, columns, discretization_max_depth):
        features = features.copy()
        self.__thresholds = {}

        for column in columns:
            thresholds = sorted(
                self.__find_thresholds(features[column], target, weights, discretization_max_depth)
            )

            self.__thresholds[column] = thresholds

            bounds = np.array(thresholds + [np.inf])

            indices = np.searchsorted(
                thresholds,
                features.loc[features[column].notna(), column].to_numpy(),
                side="left"
            )

            features[column] = features[column].astype(float)
            features.loc[features[column].notna(), column] = bounds[indices]

        return features

    def __transform(self, features):
        features = features.copy()

        for column, thresholds in self.__thresholds.items():
            bounds = np.array(thresholds + [np.inf])

            indices = np.searchsorted(
                thresholds,
                features.loc[features[column].notna(), column].to_numpy(),
                side="left"
            )

            features[column] = features[column].astype(float)
            features.loc[features[column].notna(), column] = bounds[indices]

        return features

class RandomForestClassifier():
    def __init__(self, num_of_trees = None, tree_max_depth = None, tree_min_samples_split = None, tree_min_samples_leaf = None, tree_max_features = None, bootstrap_samples = None):
        self.__features = None
        self.__target = None
        self.__num_of_trees = num_of_trees
        self.__tree_max_depth = tree_max_depth
        self.__tree_min_samples_split = tree_min_samples_split
        self.__tree_min_samples_leaf = tree_min_samples_leaf
        self.__tree_max_features = tree_max_features
        self.__num_bootstrap_samples = bootstrap_samples
        self.__trees = []

    def fit(self, features, target, discretize, discretization_max_depth):
        self.__features = features
        self.__target = target

        if self.__num_of_trees is None:
            for _ in tqdm(range(100), desc = "creating trees"):
                self.__trees.append(self.__create_trees(discretize, discretization_max_depth))
        else:
            for _ in tqdm(range(self.__num_of_trees), desc = "creating trees"):
                self.__trees.append(self.__create_trees(discretize, discretization_max_depth))

    def predict(self, features, voting = 'hard'):
        if voting != 'hard':
            predictions = np.array([tree.predict(features, voting = 'soft') for tree in self.__trees])
            return [max(self.__target.unique(), key=lambda label: np.mean([p.get(label, 0) for p in predictions[:, i]])) for i in range(len(features))]
        else:
            predictions = np.array([tree.predict(features) for tree in self.__trees])
            return [pd.Series(predictions[:, i]).mode().iloc[0] for i in range(len(features))]

    def save_to_pickle(self, model_name = None, save_to = None):
        if model_name is None:
            raise ValueError("Please provide model name through model_name argument. Example: model.pkl")

        model_data = {
            "features": self.__features,
            "target": self.__target,
            "num_of_tree": self.__num_of_trees,
            "tree_max_depth": self.__tree_max_depth,
            "tree_min_samples_split": self.__tree_min_samples_split,
            "tree_min_samples_leaf": self.__tree_min_samples_leaf,
            "tree_max_features": self.__tree_max_features,
            "bootstrap_samples": self.__num_bootstrap_samples,
            "trees": self.__trees
        }
        
        if save_to is None:
            with open(model_name, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            if save_to[-1] == '/':
                with open(save_to + '/' + model_name, 'w') as f:
                    pickle.dump(model_data, f)
            else:
                with open(save_to + model_name, 'w') as f:
                    pickle.dump(model_data, f)

    def load_from_pickle(self, pickle_dir):
        with open(pickle_dir, 'rb') as f:
            model_data = pickle.load(f)

        self.__features = model_data["features"]
        self.__target = model_data["target"]
        self.__num_of_trees = model_data["num_of_tree"]
        self.__tree_max_depth = model_data["tree_max_depth"]
        self.__tree_min_samples_split = model_data["tree_min_samples_split"]
        self.__tree_min_samples_leaf = model_data["tree_min_samples_leaf"]
        self.__tree_max_features = model_data["tree_max_features"]
        self.__num_bootstrap_samples = model_data["bootstrap_samples"]
        self.__trees = model_data["trees"]

    def __create_trees(self, discretize, discretization_max_depth):
        tree = DecisionTreeClassifier(max_depth = self.__tree_max_depth, min_samples_leaf = self.__tree_min_samples_leaf, 
                                      min_samples_split = self.__tree_min_samples_split, max_features = self.__tree_max_features)
        tree_samples, tree_target = self.__bootstrap_samples()
        tree.fit(tree_samples, tree_target, discretize, discretization_max_depth)

        return tree

    def __bootstrap_samples(self):
        indexes = list(self.__features.index)

        if self.__num_bootstrap_samples is None:
            sample_indexes = np.random.choice(indexes, size = len(self.__features), replace=True)
        else:
            sample_indexes = np.random.choice(indexes, size = self.__num_bootstrap_samples, replace=True)
        
        sampled_features = self.__features.loc[sample_indexes].reset_index(drop = True)
        sampled_target = self.__target.loc[sample_indexes].reset_index(drop = True)

        return sampled_features, sampled_target


# ---------------------------------------------------------------------------
# KNN dan Logistic Regression (sama dengan bagian 2.3 dan 2.4 notebook)
# ---------------------------------------------------------------------------
class BaseClassifier:
    """Common API: fit, predict_proba, predict, score, get_params, save, load."""

    def _encode_labels(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes to train a classifier")
        return y_enc.astype(np.int64)

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """Accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_") and not k.startswith("_")}

    def save(self, path):
        """Pickle the model state together with its class name and a format version."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": FORMAT_VERSION, "class_name": type(self).__name__, "state": self.__dict__}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class_name") != cls.__name__:
            raise TypeError(f"{path} holds a {payload.get('class_name')}, not a {cls.__name__}")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} uses format version {payload.get('format_version')}, expected {FORMAT_VERSION}")
        obj = cls.__new__(cls)
        obj.__dict__.update(payload["state"])
        return obj

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


def check_array(X):
    """Convert DataFrame or array-like input to a finite 2-D float array."""
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("input contains NaN or infinity; impute missing values before KNN or logistic regression")
    return X


class ScratchPreprocessor:
    """One-hot untuk kolom nominal + z-score untuk kolom lain; statistik di-fit dari data latih saja.

    Urutan keluaran sama dengan ColumnTransformer kelompok: blok one-hot dulu, lalu numerik.
    Kategori yang belum pernah terlihat menjadi vektor nol (setara handle_unknown="ignore").
    """

    def __init__(self, nominal_cols):
        self.nominal_cols = list(nominal_cols)

    def fit(self, df):
        self.nominal_ = [c for c in self.nominal_cols if c in df.columns]
        self.numeric_ = [c for c in df.columns if c not in self.nominal_]
        self.categories_ = {c: np.unique(df[c].to_numpy()) for c in self.nominal_}
        num = df[self.numeric_].to_numpy(dtype=float)
        self.median_ = np.nanmedian(num, axis=0)
        num = np.where(np.isnan(num), self.median_, num)
        self.mean_, self.std_ = num.mean(axis=0), num.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.feature_names_ = [f"{c}={v}" for c in self.nominal_ for v in self.categories_[c]] + self.numeric_
        return self

    def transform(self, df):
        blocks = [(df[c].to_numpy()[:, None] == self.categories_[c][None, :]).astype(float) for c in self.nominal_]
        num = df[self.numeric_].to_numpy(dtype=float)
        blocks.append((np.where(np.isnan(num), self.median_, num) - self.mean_) / self.std_)
        return np.hstack(blocks)


def smote(X, y, k=5, random_state=None):
    """SMOTE (Chawla dkk., 2002): baris sintetis kelas minoritas diinterpolasi ke salah satu k tetangga sekelas."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    X_new, y_new = [X], [y]
    for c, n in zip(classes, counts):
        need = counts.max() - n
        if need == 0:
            continue
        Xc = X[y == c]
        sq = np.einsum("ij,ij->i", Xc, Xc)
        d2 = sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T
        np.fill_diagonal(d2, np.inf)
        kk = min(k, len(Xc) - 1)
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        base = rng.integers(0, len(Xc), need)
        neigh = nn[base, rng.integers(0, kk, need)]
        X_new.append(Xc[base] + rng.random((need, 1)) * (Xc[neigh] - Xc[base]))
        y_new.append(np.full(need, c, dtype=y.dtype))
    return np.vstack(X_new), np.concatenate(y_new)


class ScratchPipeline(BaseClassifier):
    """Preprocessing -> (opsional) SMOTE pada data latih -> model scratch. Menerima DataFrame mentah seperti model anggota lain."""

    def __init__(self, model, nominal_cols=KOLOM_NOMINAL_SCRATCH, resampling=None, random_state=SEED):
        self.model = model
        self.nominal_cols = list(nominal_cols)
        self.resampling = resampling
        self.random_state = random_state

    def training_matrix(self, X_df, y):
        Xm, ym = self.pre_.transform(X_df), np.asarray(y)
        if self.resampling == "SMOTE":
            Xm, ym = smote(Xm, ym, k=5, random_state=self.random_state)
        return Xm, ym

    def fit(self, X_df, y):
        self.pre_ = ScratchPreprocessor(self.nominal_cols).fit(X_df)
        self.model.fit(*self.training_matrix(X_df, y))
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X_df):
        return self.model.predict_proba(self.pre_.transform(X_df))


class KNNScratch(BaseClassifier):
    """KNN classifier.

    metric:  'euclidean', 'manhattan', 'chebyshev', 'minkowski' (order ``p``) or 'cosine'.
    weights: 'uniform' (plain majority vote) or 'distance' (votes weighted by 1/d).

    KNN is a lazy learner: ``fit`` only stores the training set. Prediction computes the
    distance from every query row to every training row, so queries are processed in
    chunks of ``chunk_size`` rows to keep memory bounded. Euclidean distance uses the
    expansion ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab, one BLAS matrix product per chunk.
    Ties between classes go to the class that sorts first, as in scikit-learn. When several
    training rows sit at exactly the k-th distance, which of them is kept is unspecified.
    """

    METRICS = ("euclidean", "manhattan", "chebyshev", "minkowski", "cosine")

    def __init__(self, n_neighbors=5, metric="euclidean", p=2, weights="uniform", chunk_size=256):
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be at least 1")
        if metric == "minkowski" and p < 1:
            raise ValueError("minkowski order p must be >= 1 to be a metric")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p
        self.weights = weights
        self.chunk_size = chunk_size

    def fit(self, X, y):
        self.X_train_ = check_array(X)
        self.y_train_ = self._encode_labels(y)
        if len(self.X_train_) != len(self.y_train_):
            raise ValueError("X and y have different numbers of rows")
        if self.n_neighbors > len(self.X_train_):
            raise ValueError("n_neighbors is larger than the training set")
        self.n_features_in_ = self.X_train_.shape[1]
        self.sq_norms_ = np.einsum("ij,ij->i", self.X_train_, self.X_train_)
        return self

    def _distances(self, Q):
        T = self.X_train_
        if self.metric == "euclidean":
            q_sq = np.einsum("ij,ij->i", Q, Q)[:, None]
            d2 = q_sq + self.sq_norms_[None, :] - 2.0 * Q @ T.T
            return np.sqrt(np.maximum(d2, 0.0))
        if self.metric == "cosine":
            qn = np.linalg.norm(Q, axis=1, keepdims=True)
            tn = np.sqrt(self.sq_norms_)[None, :]
            return 1.0 - (Q @ T.T) / np.maximum(qn * tn, 1e-12)
        # Accumulate one feature at a time: memory stays chunk x n_train instead of chunk x n_train x d.
        D = np.zeros((len(Q), len(T)))
        for j in range(T.shape[1]):
            diff = np.abs(Q[:, j, None] - T[None, :, j])
            if self.metric == "manhattan":
                D += diff
            elif self.metric == "chebyshev":
                np.maximum(D, diff, out=D)
            else:
                D += diff ** self.p
        return D if self.metric != "minkowski" else D ** (1.0 / self.p)

    def kneighbors(self, X):
        """Return (distances, indices) of the k nearest training rows, nearest first."""
        Q = check_array(X)
        if Q.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {Q.shape[1]}")
        k = self.n_neighbors
        dist_out = np.empty((len(Q), k))
        idx_out = np.empty((len(Q), k), dtype=np.int64)
        # Per-feature metrics run fastest with small, cache-sized chunks; the BLAS metrics prefer large ones.
        step = self.chunk_size if self.metric in ("euclidean", "cosine") else min(self.chunk_size, 64)
        for start in range(0, len(Q), step):
            D = self._distances(Q[start:start + step])
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            pd_ = np.take_along_axis(D, part, axis=1)
            order = np.argsort(pd_, axis=1, kind="stable")
            idx_out[start:start + step] = np.take_along_axis(part, order, axis=1)
            dist_out[start:start + step] = np.take_along_axis(pd_, order, axis=1)
        return dist_out, idx_out

    @staticmethod
    def vote(dist, labels, n_classes, weights="uniform"):
        """Class probabilities from neighbour distances and encoded labels, both (n_query, k).

        Exposed so a search over K can compute neighbours once at the largest K and slice.
        """
        if weights == "uniform":
            w = np.ones_like(dist)
        else:
            # An exact match takes all the weight, matching scikit-learn.
            zero = dist <= 1e-12
            w = np.where(zero.any(axis=1, keepdims=True), zero.astype(np.float64), 1.0 / np.maximum(dist, 1e-12))
        votes = np.zeros((len(dist), n_classes))
        for c in range(n_classes):
            votes[:, c] = np.sum(w * (labels == c), axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return self.vote(dist, self.y_train_[idx], len(self.classes_), self.weights)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z):
    # tanh form avoids overflow in exp for large |z|.
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def class_weights(y_enc, n_classes, mode):
    """Per-sample weights; 'balanced' uses n / (K * count_c) like scikit-learn."""
    if mode is None:
        return np.ones(len(y_enc))
    counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
    if mode == "balanced":
        per_class = len(y_enc) / (n_classes * counts)
    else:
        per_class = np.array([mode[c] for c in range(n_classes)], dtype=np.float64)
    return per_class[y_enc]


class LogisticRegressionScratch(BaseClassifier):
    """Logistic regression for K classes.

    multi_class='softmax' (multinomial): P = softmax(XW + b), loss = weighted mean cross-entropy.
    multi_class='ovr' (one-vs-rest): K independent sigmoid models, each on its binary log-loss;
    the K scores are normalised to sum to 1 for ``predict_proba``.

    Objective:  L(W, b) = sum_i s_i * loss_i + (alpha / 2) * ||W||^2,  with s_i summing to 1.
    Gradient:   dL/dW = X^T (s * (P - Y)) + alpha * W,   dL/db = sum_i s_i (P_i - Y_i).
    The same P - Y form holds for softmax + cross-entropy and for sigmoid + binary log-loss.
    Training is full-batch gradient descent; it stops when the gradient norm falls below
    ``tol`` or after ``max_iter`` steps. The bias is not regularised.
    scikit-learn equivalent penalty: C = 1 / (alpha * n_samples).
    """

    def __init__(self, learning_rate=0.1, max_iter=3000, alpha=1e-3, multi_class="softmax",
                 class_weight=None, tol=1e-5, fit_intercept=True, random_state=None):
        if multi_class not in ("softmax", "ovr"):
            raise ValueError("multi_class must be 'softmax' or 'ovr'")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _forward(self, X, W, b):
        Z = X @ W + b
        return softmax(Z) if self.multi_class == "softmax" else sigmoid(Z)

    def _loss_and_grads(self, X, Y, sw, W, b):
        """Objective value and its gradients; sw are per-sample weights that sum to 1."""
        P = self._forward(X, W, b)
        if self.multi_class == "softmax":
            loss = -np.sum(sw * np.log(np.clip(np.sum(P * Y, axis=1), 1e-15, None)))
        else:
            Pc = np.clip(P, 1e-15, 1 - 1e-15)
            loss = -np.sum(sw[:, None] * (Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)))
        loss += 0.5 * self.alpha * np.sum(W * W)
        G = sw[:, None] * (P - Y)
        dW = X.T @ G + self.alpha * W
        db = G.sum(axis=0) if self.fit_intercept else np.zeros_like(b)
        return loss, dW, db

    def fit(self, X, y):
        X = check_array(X)
        y_enc = self._encode_labels(y)
        n, d = X.shape
        K = len(self.classes_)
        self.n_features_in_ = d
        sw = class_weights(y_enc, K, self.class_weight)
        sw = sw / sw.sum()
        Y = np.zeros((n, K))
        Y[np.arange(n), y_enc] = 1.0

        rng = np.random.default_rng(self.random_state)
        W = rng.normal(0.0, 0.01, size=(d, K))
        b = np.zeros(K)
        self.loss_history_, self.grad_norm_history_ = [], []
        self.converged_ = False
        it = -1
        for it in range(self.max_iter):
            loss, dW, db = self._loss_and_grads(X, Y, sw, W, b)
            grad_norm = float(np.sqrt(np.sum(dW * dW) + np.sum(db * db)))
            self.loss_history_.append(float(loss))
            self.grad_norm_history_.append(grad_norm)
            if grad_norm < self.tol:
                self.converged_ = True
                break
            W -= self.learning_rate * dW
            b -= self.learning_rate * db
        self.coef_ = W
        self.intercept_ = b
        self.n_iter_ = it + 1
        return self

    def decision_function(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        Z = self.decision_function(X)
        if self.multi_class == "softmax":
            return softmax(Z)
        P = sigmoid(Z)
        return P / P.sum(axis=1, keepdims=True)



# ---------------------------------------------------------------------------
# ANN (sama dengan bagian 2.5 notebook)
# ---------------------------------------------------------------------------
class Ann():
  def __init__(self, hidden_size=32, epochs=10000, learning_rate=0.1, seed=42, verbose=True):
      self.hidden_size = hidden_size
      self.epochs = epochs
      self.learning_rate = learning_rate
      self.seed = seed
      self.verbose = verbose
      self.parameters = None
      self.mean = None
      self.std = None
      self.n_kelas = None
      self.cost_history = []

  def sigmoid(self, Z):
    return 1 / (1 + np.exp(-Z))

  def relu(self, Z):
      return np.maximum(0, Z)

  def relu_derivative(self, Z):
      return (Z > 0).astype(float)

  def standardisasi(self, X):
      X = np.asarray(X, dtype=float)
      return ((X - self.mean) / self.std).T     # (n, fitur) -> (fitur, n)

  def one_hot(self, y):
      Y = np.zeros((self.n_kelas, len(y)))
      Y[y, np.arange(len(y))] = 1
      return Y

  def initialize_parameters(self, input_size, hidden_size, output_size):
    np.random.seed(self.seed)
    parameters = {
        "W1": np.random.randn(hidden_size, input_size) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.randn(output_size, hidden_size) * 0.01,
        "b2": np.zeros((output_size, 1))
    }
    return parameters

  def forward_propagation(self, X, parameters):
      W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

      Z1 = np.dot(W1, X) + b1
      A1 = self.relu(Z1)
      Z2 = np.dot(W2, A1) + b2
      A2 = self.sigmoid(Z2)

      cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
      return A2, cache

  def compute_cost(self, Y, A2):
      m = Y.shape[1]
      A2 = np.clip(A2, 1e-12, 1 - 1e-12)
      cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m   # binary cross-entropy
      return np.squeeze(cost)

  def backward_propagation(self, X, Y, parameters, cache):
      m = X.shape[1]
      W2 = parameters["W2"]

      dZ2 = cache["A2"] - Y
      dW2 = np.dot(dZ2, cache["A1"].T) / m
      db2 = np.sum(dZ2, axis=1, keepdims=True) / m

      dZ1 = np.dot(W2.T, dZ2) * self.relu_derivative(cache["Z1"])
      dW1 = np.dot(dZ1, X.T) / m
      db1 = np.sum(dZ1, axis=1, keepdims=True) / m

      grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
      return grads

  def update_parameters(self, parameters, grads):
      for key in parameters.keys():
          parameters[key] -= self.learning_rate * grads["d" + key]
      return parameters

  def fit(self, X, y):
      X_arr = np.asarray(X, dtype=float)
      y = np.asarray(y).astype(int)

      # simpan mean & std data latih (dipakai lagi saat predict)
      self.mean = X_arr.mean(axis=0)
      self.std = X_arr.std(axis=0)
      self.std[self.std == 0] = 1
      self.n_kelas = int(y.max()) + 1

      X_t = self.standardisasi(X_arr)
      Y = self.one_hot(y)
      self.parameters = self.initialize_parameters(X_t.shape[0], self.hidden_size, self.n_kelas)
      self.cost_history = []

      for i in range(self.epochs):
          A2, cache = self.forward_propagation(X_t, self.parameters)
          cost = self.compute_cost(Y, A2)
          grads = self.backward_propagation(X_t, Y, self.parameters, cache)
          self.parameters = self.update_parameters(self.parameters, grads)
          self.cost_history.append(cost)

          if self.verbose and i % 200 == 0:
              print(f"Epoch {i}: Cost = {cost:.4f}")
      return self

  def predict_proba(self, X):
      A2, cache = self.forward_propagation(self.standardisasi(X), self.parameters)
      return A2.T

  def predict(self, X):
      return np.argmax(self.predict_proba(X), axis=1) 

  def save_to_pickle(self, path):
      with open(path, "wb") as f:
          pickle.dump(self.__dict__, f)

  def load_from_pickle(self, path):
      with open(path, "rb") as f:
          self.__dict__.update(pickle.load(f))


# ---------------------------------------------------------------------------
# Memuat semua model yang disimpan notebook
# ---------------------------------------------------------------------------
class _NotebookUnpickler(pickle.Unpickler):
    """Pickle dari notebook mencatat kelasnya sebagai __main__.X; arahkan ke kelas di modul ini."""

    def find_class(self, module, name):
        if module == "__main__" and name in globals():
            return globals()[name]
        return super().find_class(module, name)


def _read_pickle(path):
    with open(path, "rb") as f:
        return _NotebookUnpickler(f).load()


def load_model(path):
    """Muat model apa pun dari saved_models/ (DTL, RF, KNN, Logistic Regression, ANN).

    Format dikenali dari isinya:
      - KNN / Logistic Regression: payload BaseClassifier.save() -> ScratchPipeline
      - DTL: dict dengan kunci "tree";  RF: dict dengan kunci "trees"
      - ANN: dict atribut dengan kunci "parameters"
    Semua model hasil load_model menerima DataFrame fitur yang sama dengan x_train di notebook.
    """
    data = _read_pickle(path)
    if not isinstance(data, dict):
        return data
    if "class_name" in data and "state" in data:
        cls = globals()[data["class_name"]]
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} uses format version {data.get('format_version')}, expected {FORMAT_VERSION}")
        obj = cls.__new__(cls)
        obj.__dict__.update(data["state"])
        return obj
    if "tree" in data:
        obj = DecisionTreeClassifier()
        prefix = "_DecisionTreeClassifier__"
    elif "trees" in data:
        obj = RandomForestClassifier()
        prefix = "_RandomForestClassifier__"
        data = {("num_of_trees" if k == "num_of_tree" else "num_bootstrap_samples" if k == "bootstrap_samples" else k): v
                for k, v in data.items()}
    elif "parameters" in data:
        obj = Ann()
        obj.__dict__.update(data)
        return obj
    else:
        raise ValueError(f"format pickle {path} tidak dikenali (kunci: {sorted(data)})")
    for k, v in data.items():
        setattr(obj, prefix + k, v)
    return obj


def load_all_models(folder="saved_models"):
    """Muat semua file .pkl di folder; hasilnya dict {nama_file_tanpa_.pkl: model}."""
    return {p.stem: load_model(p) for p in sorted(Path(folder).glob("*.pkl"))}
