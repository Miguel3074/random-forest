import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from io import StringIO

# None para output aleatorio
RANDOM_SEED = None

N_TREES = 100
TREE_DEPTH = 10

class Node:
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, *, value=None, info_gain=None, num_samples=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.info_gain = info_gain 
        self.value = value 
        self.num_samples = num_samples

class DecisionTree:
    def __init__(self, min_samples_to_split=2, max_depth=100, n_features_per_split=None):
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.n_features_per_split = n_features_per_split 
        self.root_node = None
        self.actual_n_features_per_split = None

    def _is_stopping_criteria_met(self, depth, num_samples_in_node, num_unique_labels):
        if (depth >= self.max_depth or
            num_samples_in_node < self.min_samples_to_split or
            num_unique_labels == 1):
            return True
        return False

    def fit(self, X, y):
        if self.n_features_per_split is None or self.n_features_per_split == 'all':
            self.actual_n_features_per_split = X.shape[1]
        elif self.n_features_per_split == 'sqrt':
            self.actual_n_features_per_split = int(np.sqrt(X.shape[1]))
        else:
            self.actual_n_features_per_split = min(self.n_features_per_split, X.shape[1])
        
        if X.shape[1] > 0 and self.actual_n_features_per_split == 0 :
             self.actual_n_features_per_split = 1

        self.root_node = self._build_tree_recursively(X, y, current_depth=0)

    def _build_tree_recursively(self, X, y, current_depth):
        num_samples, num_features = X.shape
        num_unique_labels = len(np.unique(y))

        if self._is_stopping_criteria_met(current_depth, num_samples, num_unique_labels):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, num_samples=num_samples)

        if num_features == 0: 
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, num_samples=num_samples)
        
        n_features_to_select = min(self.actual_n_features_per_split, num_features)
        if n_features_to_select <=0 and num_features > 0:
            n_features_to_select = num_features


        feature_indices_to_consider = np.random.choice(num_features, n_features_to_select, replace=False)
        
        best_split = self._get_best_split(X, y, feature_indices_to_consider)

        if best_split['info_gain'] <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, num_samples=num_samples)

        left_X, left_y = X[best_split['left_indices'], :], y[best_split['left_indices']]
        right_X, right_y = X[best_split['right_indices'], :], y[best_split['right_indices']]

        left_subtree = self._build_tree_recursively(left_X, left_y, current_depth + 1)
        right_subtree = self._build_tree_recursively(right_X, right_y, current_depth + 1)
        
        return Node(feature_index=best_split['feature_index'], 
                    threshold=best_split['threshold'], 
                    left_child=left_subtree, 
                    right_child=right_subtree, 
                    info_gain=best_split['info_gain'],
                    num_samples=num_samples)

    def _get_best_split(self, X, y, feature_indices):
        raise NotImplementedError

    def _calculate_leaf_value(self, y):
        raise NotImplementedError

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root_node) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)

class DecisionTreeRegressor(DecisionTree):
    def _calculate_mse(self, y):
        if len(y) == 0: return 0
        return np.mean((y - np.mean(y))**2)

    def _get_best_split(self, X, y, feature_indices):
        best_split = {'info_gain': -float('inf')} 
        parent_mse = self._calculate_mse(y)
        n_samples = len(y)

        if n_samples == 0: 
              return best_split

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                mse_left = self._calculate_mse(y_left)
                mse_right = self._calculate_mse(y_right)
                
                weighted_mse_children = (len(y_left) / n_samples) * mse_left + \
                                        (len(y_right) / n_samples) * mse_right
                
                mse_reduction = parent_mse - weighted_mse_children

                if mse_reduction > best_split['info_gain']:
                    best_split = {
                        'feature_index': feature_idx,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'info_gain': mse_reduction
                    }
        return best_split

    def _calculate_leaf_value(self, y):
        return np.mean(y) if len(y) > 0 else 0 

class DecisionTreeClassifier(DecisionTree):
    def _calculate_gini_impurity(self, y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _get_best_split(self, X, y, feature_indices):
        best_split = {'info_gain': -float('inf')}
        parent_gini = self._calculate_gini_impurity(y)
        n_samples = len(y)

        if n_samples == 0:
            return best_split

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                y_left, y_right = y[left_indices], y[right_indices]
                gini_left = self._calculate_gini_impurity(y_left)
                gini_right = self._calculate_gini_impurity(y_right)

                weighted_gini_children = (len(y_left) / n_samples) * gini_left + \
                                         (len(y_right) / n_samples) * gini_right
                
                information_gain = parent_gini - weighted_gini_children

                if information_gain > best_split['info_gain']:
                    best_split = {
                        'feature_index': feature_idx,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'info_gain': information_gain
                    }
        return best_split

    def _calculate_leaf_value(self, y):
        most_common = Counter(y).most_common(1)
        return most_common[0][0] if most_common else None

class RandomForest:
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, 
                     n_features_per_tree='sqrt', tree_model_class=None, random_state=None):
        self.n_trees = n_trees
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.n_features_per_tree = n_features_per_tree 
        self.tree_model_class = tree_model_class
        self.trees = []
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)


    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X_df, y_series):
        self.trees = []
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        y = y_series.values if isinstance(y_series, pd.Series) else y_series

        for i in range(self.n_trees):
            if self.random_state is not None:
                np.random.seed(self.random_state + i)
            
            tree = self.tree_model_class(
                min_samples_to_split=self.min_samples_to_split,
                max_depth=self.max_depth,
                n_features_per_split=self.n_features_per_tree 
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X_df):
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        tree_predictions_list = []
        for tree in self.trees:
            tree_predictions_list.append(tree.predict(X))
        
        if not tree_predictions_list:
            return np.array([])

        tree_preds_stacked = np.stack(tree_predictions_list, axis=0)
        tree_preds_transposed = tree_preds_stacked.T
        
        return self._aggregate_predictions(tree_preds_transposed)

    def _aggregate_predictions(self, tree_preds_transposed):
        raise NotImplementedError

class RandomForestRegressor(RandomForest):
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, n_features_per_tree='sqrt', random_state=None):
        super().__init__(n_trees, min_samples_to_split, max_depth, n_features_per_tree, 
                         tree_model_class=DecisionTreeRegressor, random_state=random_state)

    def _aggregate_predictions(self, tree_preds_transposed):
        return np.mean(tree_preds_transposed, axis=1)

class RandomForestClassifier(RandomForest):
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, n_features_per_tree='sqrt', random_state=None):
        super().__init__(n_trees, min_samples_to_split, max_depth, n_features_per_tree,
                         tree_model_class=DecisionTreeClassifier, random_state=random_state)

    def _aggregate_predictions(self, tree_preds_transposed):
        y_pred_aggregated = []
        for sample_predictions in tree_preds_transposed:
            most_common = Counter(sample_predictions).most_common(1)
            y_pred_aggregated.append(most_common[0][0] if most_common else None)
        return np.array(y_pred_aggregated)

def run_random_forest_algorithm():
    print("--- Starting Random Forest Algorithm (Implementation) ---")

    col_names_hist_file = ['i', 'si1_pSist', 'si2_pDiast', 'si3_qPA', 'si4_pulso', 'si5_resp', 'g1_gravid', 'y1_classe']
    features_cols = ['si3_qPA', 'si4_pulso', 'si5_resp'] 
    target_reg_col = 'g1_gravid'
    target_cls_col = 'y1_classe'

    input_hist_filename = 'treino_sinais_vitais_com_label.csv'

    df_hist = pd.read_csv(input_hist_filename, names=col_names_hist_file, header=None, sep=',')
    print(f"\nFile '{input_hist_filename}' loaded successfully.")

    X_df = df_hist[features_cols]
    y_reg_series = df_hist[target_reg_col]
    y_cls_series = df_hist[target_cls_col].astype(int)

    unique_classes = sorted(y_cls_series.unique())
    class_names = [f"Class {c}" for c in unique_classes]
    
    if(RANDOM_SEED == None):
        seed_for_split=1
    seed_for_split = RANDOM_SEED

    stratify_option = y_cls_series if len(unique_classes) > 1 and y_cls_series.value_counts().min() > 1 else None
    X_train_df, X_val_df, y_train_reg, y_val_reg, y_train_cls, y_val_cls = train_test_split(
        X_df, y_reg_series, y_cls_series, test_size=0.2, random_state=seed_for_split, stratify=stratify_option
    )
    print(f"\nHistorical data split: {len(X_train_df)} for training, {len(X_val_df)} for validation.")


    n_trees = N_TREES
    max_depth = TREE_DEPTH
    min_samples_split = 2
    n_features_rf = 'sqrt' 
    random_state_rf = RANDOM_SEED 

    print("\n" + "="*55)
    print("### Regression Task (Random Forest) ###")
    print("="*55)

    rf_regressor = RandomForestRegressor(
        n_trees=n_trees, 
        max_depth=max_depth, 
        min_samples_to_split=min_samples_split,
        n_features_per_tree=n_features_rf,
        random_state=random_state_rf
    )
    rf_regressor.fit(X_train_df, y_train_reg)
    y_pred_reg_val = rf_regressor.predict(X_val_df)
    rmse_rf_reg = np.sqrt(mean_squared_error(y_val_reg, y_pred_reg_val))
    print(f"\nRandom Forest Regressor (Evaluation on Validation Set):")
    print(f"  RMSE: {rmse_rf_reg:.4f}")

    print("\n" + "="*60)
    print("### Classification Task (Random Forest) ###")
    print("="*60)

    rf_classifier = RandomForestClassifier(
        n_trees=n_trees, 
        max_depth=max_depth, 
        min_samples_to_split=min_samples_split,
        n_features_per_tree=n_features_rf,
        random_state=random_state_rf 
    )
    rf_classifier.fit(X_train_df, y_train_cls)
    y_pred_cls_val = rf_classifier.predict(X_val_df)
    
    print(f"\nRandom Forest Classifier (Evaluation on Validation Set):")
    y_pred_cls_val_int = np.array(y_pred_cls_val).astype(int) 
    y_val_cls_np_int = y_val_cls.values.astype(int) 

    print(f"  Accuracy: {accuracy_score(y_val_cls_np_int, y_pred_cls_val_int):.4f}")
    print("  Classification Report:")
    print(classification_report(y_val_cls_np_int, y_pred_cls_val_int, labels=unique_classes, target_names=class_names, zero_division=0))
    print("  Confusion Matrix:")
    cm_rf = confusion_matrix(y_val_cls_np_int, y_pred_cls_val_int, labels=unique_classes)
    print(pd.DataFrame(cm_rf, index=[f"Actual {c}" for c in unique_classes], columns=[f"Predicted {c}" for c in unique_classes]))
    
    print("\n" + "="*62)
    print("### Prediction for Blind Test Set ###")
    print("="*62)
    print("\nTraining final Random Forest models with all historical data...")

    final_rf_regressor = RandomForestRegressor(
        n_trees=n_trees, max_depth=max_depth, min_samples_to_split=min_samples_split, 
        n_features_per_tree=n_features_rf, random_state=random_state_rf
    )
    final_rf_regressor.fit(X_df, y_reg_series) 

    final_rf_classifier = RandomForestClassifier(
        n_trees=n_trees, max_depth=max_depth, min_samples_to_split=min_samples_split,
        n_features_per_tree=n_features_rf, random_state=random_state_rf
    )
    final_rf_classifier.fit(X_df, y_cls_series) 
    print("Final Random Forest models trained.")

    input_test_filename = 'treino_sinais_vitais_sem_label.csv' 
    
    actual_col_names_in_test_file = ['i', 'si1_pSist', 'si2_pDiast', 'si3_qPA', 'si4_pulso', 'si5_resp', 'g1_gravid']
    
    df_test = pd.read_csv(input_test_filename, names=actual_col_names_in_test_file, header=None, sep=',')
    print(f"\nFile '{input_test_filename}' loaded successfully.")

    X_test_df = df_test[features_cols] 
    print("Making predictions on the test set with models...")
    gravid_pred_test = final_rf_regressor.predict(X_test_df)
    classe_pred_test = final_rf_classifier.predict(X_test_df)
    results_df = pd.DataFrame({
        'si3_qPA_original': X_test_df['si3_qPA'].round(4), 
        'gravid_pred': gravid_pred_test,
        'classe_pred': classe_pred_test.astype(int)
    })
    results_df['gravid_pred'] = results_df['gravid_pred'].round(4)
    
    output_filename_predictions = 'sinaisvitais_predicoes.txt'
    results_df.to_csv(output_filename_predictions, header=False, index=False, float_format='%.4f')
    
    print(f"\nPrediction results for '{X_test_df.shape[0]}' samples from the test file:")
    print(results_df.head().to_string(index=False, header=False)) 
    print(f"\nPredictions saved to '{output_filename_predictions}'")


if __name__ == '__main__':
    run_random_forest_algorithm()