from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

class RFACV:
    def __init__(self, model=None, cv=5, scoring='neg_mean_squared_error', select_all=True):
        """
        Recursive Feature Addition with Cross-Validation (RFACV)
        
        Parameters:
        - model: The model to use for feature evaluation (default: RandomForestRegressor).
        - cv: Number of cross-validation folds.
        - scoring: Scoring metric for cross-validation.
        """
        self.model = model if model else xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.cv = cv
        self.scoring = scoring
        self.selected_features = []
        self.select_all = select_all  # New argument to select all features
        self.best_score = float("inf")
        feature_scores = {}
        self.cv_results_ = []  # Initialize cv_results_ to store scores
        

    def fit(self, X, y):
        """
        Fit the RFACV to the dataset.
        
        Parameters:
        - X: DataFrame containing features.
        - y: Series or array containing the target variable.
        
        Returns:
        - self: Fitted instance of the RFACV class.
        """
        available_features = list(X.columns)
        self.selected_features = []
        self.best_score = float("inf")
        feature_scores = {}
        
        while available_features:
            
            # Try adding each available feature to the selected feature set
            for feature in available_features:
                features_to_try = self.selected_features + [feature]
                X_subset = X[features_to_try]
                
                # Perform cross-validation and get score
                cv_score = -np.mean(cross_val_score(self.model, X_subset, y, cv=self.cv, scoring=self.scoring, n_jobs=-1))
                feature_scores[feature] = cv_score
                
                #print(f"Testing feature '{feature}': CV Score = {cv_score:.4f}")

            # Find the feature with the best score when added
            best_feature = min(feature_scores, key=feature_scores.get)
            best_feature_score = feature_scores[best_feature]
            
            # Check if adding this feature improves the model
            if best_feature_score < self.best_score:
                self.best_score = best_feature_score
                self.selected_features.append(best_feature)
                self.cv_results_.append(-self.best_score)  # Store the score for this step
                available_features.remove(best_feature)
                print(f">>>>>>>>>>>>>>Feature added: {best_feature}, {len(self.selected_features)} New CV Score: {self.best_score}")
            else:
                print("No improvement; stopping feature addition.")
                if not self.select_all:
                    break
        
        # Fit the final model with the selected features
        self.model.fit(X[self.selected_features], y)

        return self

    def transform(self, X):
        """
        Transform the dataset to include only the selected features.
        
        Parameters:
        - X: DataFrame containing the original features.
        
        Returns:
        - Transformed DataFrame containing only selected features.
        """
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit to the data and return the dataset with selected features.
        
        Parameters:
        - X: DataFrame containing the original features.
        - y: Series or array containing the target variable.
        
        Returns:
        - Transformed DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        """
        Get the list of selected features after fitting.
        
        Returns:
        - List of selected features.
        """
        return self.selected_features

    def get_best_score(self):
        """
        Get the best cross-validation score obtained after fitting.
        
        Returns:
        - Best cross-validation score.
        """
        return self.best_score
    
    def plot_scores(self):
        """
        Plot the cross-validation scores against the number of features selected.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cv_results_) + 1), self.cv_results_, marker='o', linestyle='-')
        plt.title("RFACV Feature Addition - Cross-Validation Score vs Number of Features")
        plt.xlabel("Number of Features")
        plt.ylabel("Cross-Validation Score")
        plt.grid(True)
        plt.show()