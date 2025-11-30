import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    
    if len(np.unique(feature_vector)) <= 1:
        return np.array([]), np.array([]), None, float('inf')

    sorted_indices = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]

    unique_feature_vals = np.unique(sorted_feature)
    if len(unique_feature_vals) == 1:
        return np.array([]), np.array([]), None, float('inf')

    thresholds_all = (sorted_feature[1:] + sorted_feature[:-1]) / 2.0

    unique_threshold_mask = sorted_feature[1:] != sorted_feature[:-1]
    thresholds_unique = thresholds_all[unique_threshold_mask]

    if len(thresholds_unique) == 0:
        return np.array([]), np.array([]), None, float('inf')

    ginis_list = []
    best_gini = float('inf')
    best_threshold = None
    n_total = len(sorted_target)

    for threshold in thresholds_unique:
        left_mask = sorted_feature < threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            ginis_list.append(float('inf')) 
            continue

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        left_target = sorted_target[left_mask]
        right_target = sorted_target[right_mask]

        p1_left = np.mean(left_target) if n_left > 0 else 0
        p1_right = np.mean(right_target) if n_right > 0 else 0

        gini_left = 1 - p1_left**2 - (1-p1_left)**2 if n_left > 0 else 0
        gini_right = 1 - p1_right**2 - (1-p1_right)**2 if n_right > 0 else 0

        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

        ginis_list.append(gini)

        if gini < best_gini:
            best_gini = gini
            best_threshold = threshold

    if not ginis_list: 
        return np.array([]), np.array([]), None, float('inf')

    ginis_array = np.array(ginis_list)

    if best_threshold is None:
        best_gini = float('inf')

    return thresholds_unique, ginis_array, best_threshold, best_gini

def find_best_categorical_split(feature_vector, target_vector):
    unique_vals = np.unique(feature_vector)
    if len(unique_vals) <= 1:
        return None, None, float('inf')
    
    category_means = {}
    for val in unique_vals:
        mask = feature_vector == val
        if np.sum(mask) > 0:
            category_means[val] = np.mean(target_vector[mask])
    
    sorted_categories = sorted(unique_vals, key=lambda x: category_means.get(x, 0))
    
    best_gini = float('inf')
    best_split = None
    
    for i in range(1, len(sorted_categories)):
        left_categories = set(sorted_categories[:i])
        
        left_mask = np.array([x in left_categories for x in feature_vector])
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
            
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(target_vector)
        
        left_counts = Counter(target_vector[left_mask])
        gini_left = 1 - sum((count / n_left) ** 2 for count in left_counts.values())
        
        right_counts = Counter(target_vector[right_mask])
        gini_right = 1 - sum((count / n_right) ** 2 for count in right_counts.values())
        
        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        
        if gini < best_gini:
            best_gini = gini
            best_split = left_categories
            
    return best_split, best_gini, "categorical"

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        valid_types = ["real", "categorical"]
        for ft in feature_types:
            if ft not in valid_types:
                raise ValueError(f"Unknown feature type: {ft}")
        
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        n_samples = len(sub_y)
        
        if n_samples == 0:
            node["type"] = "terminal"
            node["class"] = 0
            return
        
        class_counts = Counter(sub_y)
        most_common_class = class_counts.most_common(1)[0][0]
        node["most_common_class"] = most_common_class
        
        if (len(class_counts) == 1 or 
            (self._max_depth is not None and depth >= self._max_depth) or  
            n_samples < self._min_samples_split): 
            node["type"] = "terminal"
            node["class"] = most_common_class
            return
        
        best_feature = None
        best_gini = float('inf')
        best_split = None
        best_type = None
        best_mask = None
        
        for feature_idx in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature_idx]
            feature_vector = sub_X[:, feature_idx]
            
            if feature_type == "real":
                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
                if threshold is not None and gini < best_gini:
                    mask = feature_vector < threshold
                    if (np.sum(mask) >= self._min_samples_leaf and 
                        np.sum(~mask) >= self._min_samples_leaf):
                        best_feature = feature_idx
                        best_gini = gini
                        best_split = threshold
                        best_type = "real"
                        best_mask = mask
                        
            elif feature_type == "categorical":
                split_categories, gini, _ = find_best_categorical_split(feature_vector, sub_y)
                if split_categories is not None and gini < best_gini:
                    mask = np.array([x in split_categories for x in feature_vector])
                    if (np.sum(mask) >= self._min_samples_leaf and 
                        np.sum(~mask) >= self._min_samples_leaf):
                        best_feature = feature_idx
                        best_gini = gini
                        best_split = split_categories
                        best_type = "categorical"
                        best_mask = mask
        
        if best_feature is None:
            node["type"] = "terminal"
            node["class"] = most_common_class
            return
        
        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        node["feature_type"] = best_type
        node["threshold"] = best_split
        node["gini"] = best_gini
        
        node["left_child"] = {}
        node["right_child"] = {}
        
        left_mask = best_mask
        right_mask = ~best_mask
        
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = node["feature_type"]
        feature_val = x[feature_idx]
        
        if feature_type == "real":
            threshold = node["threshold"]
            if feature_val < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            categories_split = node["threshold"]
            if feature_val in categories_split:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        
        return node["most_common_class"]

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        if len(X.shape) == 1:
            return self._predict_node(X, self._tree)
        else:
            return np.array([self._predict_node(x, self._tree) for x in X])

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types.copy() if deep else self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, f"_{key}", value)
        return self