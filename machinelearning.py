import pandas as pd
import numpy as np
import argparse
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import kendalltau, spearmanr

# Import models
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import lightgbm as lgb

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide") # For kendalltau/spearmanr
warnings.filterwarnings("ignore", message="No further splits with positive gain.*") # LightGBM
warnings.filterwarnings("ignore", message="Accuracy may be bad since you didn't explicitly set num_leaves.*") # LightGBM


# --- Model Definitions and Hyperparameter Grids (from notebook) ---
CLASSIFICATION_MODELS_PARAMS = {
    'RandomForestClassifier': (RandomForestClassifier(random_state=42, class_weight='balanced'), {
        'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
    }),
    'SVC': (SVC(random_state=42, probability=True, class_weight='balanced'), {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }),
    'LogisticRegression': (LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'), {
        'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']
    }),
    'KNeighborsClassifier': (KNeighborsClassifier(), {
        'n_neighbors': list(range(1, 11)), 'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }),
    'GaussianNB': (GaussianNB(), {'var_smoothing': np.logspace(0, -9, num=100)}),
    'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {
        'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
    }),
    'AdaBoostClassifier': (AdaBoostClassifier(random_state=42), {
        'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.001, 0.01, 0.1, 1.0]
    }),
    'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0]
    }),
    'LGBMClassifier': (lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1), {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40], 'max_depth': [-1, 10, 20],
        'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9]
    })
}

REGRESSION_MODELS_PARAMS = {
    'RandomForestRegressor': (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
    }),
    'SVR': (SVR(), {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }),
    'LinearRegression': (LinearRegression(), {}), # No hyperparameters for RandomizedSearchCV
    'KNeighborsRegressor': (KNeighborsRegressor(), {
        'n_neighbors': list(range(1, 11)), 'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }),
    'DecisionTreeRegressor': (DecisionTreeRegressor(random_state=42), {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'AdaBoostRegressor': (AdaBoostRegressor(random_state=42), {
        'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }),
    'GradientBoostingRegressor': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0],
        'loss': ['squared_error', 'absolute_error', 'huber'] # Removed 'quantile' as it has specific alpha
    }),
    'LGBMRegressor': (lgb.LGBMRegressor(random_state=42, verbose=-1), {
        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40], 'max_depth': [-1, 10, 20],
        'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9]
    })
}

# --- Helper Functions ---
def transform_gt_labels(df_in, num_gt_labels):
    """Transforms the GT column based on the number of desired labels."""
    df = df_in.copy()
    if 'GT' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'GT' column.")
    
    df["GT_new"] = np.nan
    if num_gt_labels == 3:
        df.loc[df['GT'] == 0, 'GT_new'] = 0
        df.loc[df['GT'] == 1, 'GT_new'] = 0
        df.loc[df['GT'] == 2, 'GT_new'] = 1
        df.loc[df['GT'] == 3, 'GT_new'] = 2
        df.loc[df['GT'] == 4, 'GT_new'] = 2
        # Drop rows where GT_new could not be mapped (if any GT values were outside 0-4)
        df.dropna(subset=['GT_new'], inplace=True)
        df['GT_new'] = df['GT_new'].astype(int)
    elif num_gt_labels == 5:
        # Assuming original GT 0,1,2,3,4 are the 5 labels
        df.loc[df['GT'].isin([0, 1, 2, 3, 4]), 'GT_new'] = df['GT']
        df.dropna(subset=['GT_new'], inplace=True)
        df['GT_new'] = df['GT_new'].astype(int)
    else:
        raise ValueError("num_gt_labels must be 3 or 5.")
    return df

def calculate_metrics(y_true, y_pred, model_type):
    """Calculates evaluation metrics."""
    metrics = {}
    
    # Common metrics (handle potential issues with constant arrays for correlation)
    try:
        ktau, kpval = kendalltau(y_true, y_pred)
        metrics['kendall_tau'] = ktau if not np.isnan(ktau) else 0.0
        metrics['kendall_pvalue'] = kpval if not np.isnan(kpval) else 1.0
    except ValueError: # Can happen if y_true or y_pred are constant
        metrics['kendall_tau'] = 0.0
        metrics['kendall_pvalue'] = 1.0

    try:
        srho, spval = spearmanr(y_true, y_pred)
        metrics['spearman_rho'] = srho if not np.isnan(srho) else 0.0
        metrics['spearman_pvalue'] = spval if not np.isnan(spval) else 1.0
    except ValueError:
        metrics['spearman_rho'] = 0.0
        metrics['spearman_pvalue'] = 1.0
        
    if model_type == 'regression':
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        # Accuracy for regression: as per notebook (rounded predictions)
        metrics['accuracy'] = accuracy_score(y_true.astype(int), np.round(y_pred).astype(int))
    elif model_type == 'classification':
        metrics['mae'] = np.nan # MAE not for classification per request
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
    return metrics

def evaluate_model_performance(model_name, model_instance, param_distributions,
                               X_train, y_train, X_test, y_test,
                               model_type, cv_fold, n_iter_search=20): # Increased n_iter slightly
    """Trains model using RandomizedSearchCV (if params provided) and evaluates."""
    print(f"    Evaluating {model_name} ({model_type})...")
    
    best_model = model_instance
    if param_distributions: # If there are parameters to search
        try:
            random_search = RandomizedSearchCV(
                estimator=model_instance,
                param_distributions=param_distributions,
                n_iter=n_iter_search,
                cv=cv_fold,
                random_state=42,
                n_jobs=-1, # Use all available cores
                scoring='accuracy' if model_type == 'classification' else 'neg_mean_absolute_error'
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
        except Exception as e:
            print(f"      Error during RandomizedSearchCV for {model_name}: {e}. Using default model parameters.")
            # Fallback to default parameters if search fails
            try:
                best_model.fit(X_train, y_train)
            except Exception as fit_e:
                print(f"      Error during fallback model fit for {model_name}: {fit_e}.")
                return None # Cannot proceed with this model
    else: # No parameters to search, just fit the model
        try:
            best_model.fit(X_train, y_train)
        except Exception as e:
            print(f"      Error during model fit for {model_name} (no search): {e}.")
            return None

    try:
        y_pred = best_model.predict(X_test)
    except Exception as e:
        print(f"      Error during prediction for {model_name}: {e}.")
        return None

    # For regression, predictions might be float. For classification, they should be int class labels.
    if model_type == 'classification' and y_pred.dtype == float:
        y_pred = np.round(y_pred).astype(int) # Ensure class labels are int for metrics

    eval_metrics = calculate_metrics(y_test, y_pred, model_type)
    eval_metrics['model_name'] = model_name
    eval_metrics['model_type'] = model_type
    
    return eval_metrics

# --- Main Processing Function ---
def main(input_excel_file, output_csv_file):
    """Main function to load data, process, run models, and save results."""
    try:
        df_original = pd.read_excel(input_excel_file)
    except FileNotFoundError:
        print(f"Error: Input Excel file '{input_excel_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    all_run_results = []

    for num_labels in [3, 5]:
        print(f"\n--- Processing for {num_labels} GT labels scenario ---")
        df_processed = df_original.copy()
        
        try:
            df_processed = transform_gt_labels(df_processed, num_labels)
        except ValueError as e:
            print(f"  Error transforming GT labels for {num_labels} labels: {e}")
            continue # Skip to next scenario if label transformation fails

        if df_processed.empty or 'GT_new' not in df_processed.columns or df_processed['GT_new'].isnull().all():
            print(f"  No valid data after GT transformation for {num_labels} labels. Skipping.")
            continue
        
        # Define features (X) and target (y)
        # Assuming all other columns are features and numeric
        feature_cols = [col for col in df_processed.columns if col not in ['GT', 'GT_new']]
        if not feature_cols:
            print(f"  No feature columns found for {num_labels} labels scenario after removing GT and GT_new. Skipping.")
            continue
            
        X = df_processed[feature_cols]
        y = df_processed['GT_new']

        if X.empty:
            print(f"  Feature set X is empty for {num_labels} labels scenario. Skipping.")
            continue
        
        # Check for non-numeric features that StandardScaler can't handle
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            print(f"  Warning: Non-numeric feature columns found: {list(non_numeric_cols)}. These will cause errors with StandardScaler. Please preprocess them.")
            # Optionally, drop them for now or raise an error
            # X = X.select_dtypes(include=np.number)
            # if X.empty:
            #     print(f"  Feature set X is empty after removing non-numeric columns for {num_labels} labels. Skipping.")
            #     continue
            print(f"  Skipping {num_labels} labels scenario due to non-numeric features.")
            continue


        # Split data (stratify on y)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError as e: # Handles cases where stratification is not possible (e.g., too few samples in a class)
            print(f"  Could not stratify during train-test split for {num_labels} labels: {e}. Trying without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if X_train.empty or X_test.empty:
            print(f"  Train or test set is empty after split for {num_labels} labels. Skipping.")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        # Regression Models
        print(f"\n  Running Regression Models for {num_labels} labels...")
        reg_cv = KFold(n_splits=min(3, len(X_train_scaled_df)), shuffle=True, random_state=42) # Ensure n_splits <= n_samples
        if len(X_train_scaled_df) < 3 : reg_cv = KFold(n_splits=len(X_train_scaled_df))


        for model_name, (model_instance, params) in REGRESSION_MODELS_PARAMS.items():
            # Ensure y_train and y_test are float for regression if they became int due to GT_new
            current_y_train = y_train.astype(float)
            current_y_test = y_test.astype(float)

            metrics = evaluate_model_performance(model_name, model_instance, params,
                                                 X_train_scaled_df, current_y_train, X_test_scaled_df, current_y_test,
                                                 model_type='regression', cv_fold=reg_cv)
            if metrics:
                metrics['gt_scenario'] = f'{num_labels}_labels'
                all_run_results.append(metrics)

        # Classification Models
        print(f"\n  Running Classification Models for {num_labels} labels...")
        # Ensure y is suitable for StratifiedKFold (enough members per class)
        min_class_count_train = y_train.value_counts().min()
        min_class_count_test = y_test.value_counts().min() # Though CV is on train
        
        n_splits_class = min(3, min_class_count_train) if min_class_count_train > 1 else 2 # Adjust n_splits
        if n_splits_class <=1 and len(y_train.unique()) > 1 : # if min_class_count_train is 1 but multiple classes
             print(f"    Warning: Some classes in y_train for {num_labels}-label classification have only 1 member. StratifiedKFold might not work well or reduce n_splits.")
             n_splits_class = 2 # Try with 2, or fallback to KFold if still problematic.
        
        if len(y_train.unique()) == 1:
             print(f"    Skipping classification for {num_labels} labels as y_train has only one unique class.")
        else:
            try:
                class_cv = StratifiedKFold(n_splits=n_splits_class, shuffle=True, random_state=42)
                 # Test if split is possible
                for _, _ in class_cv.split(X_train_scaled_df, y_train.astype(int)):
                    break
            except ValueError as e_skf:
                print(f"    Warning: StratifiedKFold failed for {num_labels}-label classification (n_splits={n_splits_class}): {e_skf}. Using regular KFold.")
                class_cv = KFold(n_splits=n_splits_class if n_splits_class > 1 else 2, shuffle=True, random_state=42)

            for model_name, (model_instance, params) in CLASSIFICATION_MODELS_PARAMS.items():
                # Ensure y_train and y_test are int for classification
                current_y_train = y_train.astype(int)
                current_y_test = y_test.astype(int)
                metrics = evaluate_model_performance(model_name, model_instance, params,
                                                    X_train_scaled_df, current_y_train, X_test_scaled_df, current_y_test,
                                                    model_type='classification', cv_fold=class_cv)
                if metrics:
                    metrics['gt_scenario'] = f'{num_labels}_labels'
                    all_run_results.append(metrics)
    
    if not all_run_results:
        print("\nNo results were generated. Output CSV will be empty or not created.")
        return

    results_df = pd.DataFrame(all_run_results)
    
    # Ensure all requested columns exist, fill with NaN if not (e.g., mae for classification)
    cols_order = ['gt_scenario', 'model_name', 'model_type', 'mae', 'accuracy',
                  'kendall_tau', 'kendall_pvalue', 'spearman_rho', 'spearman_pvalue']
    for col in cols_order:
        if col not in results_df.columns:
            results_df[col] = np.nan
            
    results_df = results_df[cols_order] # Reorder columns

    try:
        results_df.to_csv(output_csv_file, index=False, float_format='%.4f')
        print(f"\nResults successfully saved to {output_csv_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML models on Excel data and evaluate.")
    parser.add_argument("input_excel_file", help="Path to the input Excel file.")
    parser.add_argument("output_csv_file", help="Path to save the results CSV file.")
    
    args = parser.parse_args()
    
    main(args.input_excel_file, args.output_csv_file)