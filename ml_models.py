import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle

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

        return self.__tree

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
