import pandas as pd
import numpy as np
import argparse
import warnings
from IPython.display import display, Image
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold,GroupShuffleSplit

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import kendalltau, spearmanr

from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="No further splits with positive gain.*")
warnings.filterwarnings("ignore", message="Accuracy may be bad since you didn't explicitly set num_leaves.*")


# --- Model Definitions and Hyperparameter Grids ---
CLASSIFICATION_MODELS_PARAMS = {
    'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [None, 10, 20, 30,40,50],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']
    }),
    'SVC': (SVC(random_state=42), {
        'C': np.logspace(-4, 4, 20),
        'gamma': np.logspace(-4, 4, 20),
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }),
    'LogisticRegression': (LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs'), {}),
    'LGBMClassifier': (LGBMClassifier(random_state=42), {
        'num_leaves': [31, 62, 127, 255],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_child_samples': [10, 20, 30, 40, 50],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    })
}

REGRESSION_MODELS_PARAMS = {
    'RandomForestRegressor': (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'SVR': (SVR(), {
        'C': np.logspace(-4, 4, 20),
        'gamma': np.logspace(-4, 4, 20),
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }),
    'LinearRegression': (LinearRegression(), {}), 
    'LGBMRegressor': (LGBMRegressor(random_state=42), {
        'num_leaves': [31, 62, 127, 255],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_child_samples': [10, 20, 30, 40, 50],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }),
    'XGBRegressor': (XGBRegressor(random_state=42), {
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    })
}

def transform_gt_labels(df_in, num_gt_labels):
    """Transforms the GT column based on the number of desired labels."""
    df = df_in.copy()
    df["GT_new"] = np.nan
    if num_gt_labels == 3:
        df.loc[df['GT'] == 0, 'GT_new'] = 0
        df.loc[df['GT'] == 1, 'GT_new'] = 0
        df.loc[df['GT'] == 2, 'GT_new'] = 1
        df.loc[df['GT'] == 3, 'GT_new'] = 2
        df.loc[df['GT'] == 4, 'GT_new'] = 2
        df.dropna(subset=['GT_new'], inplace=True)
        df['GT_new'] = df['GT_new'].astype(int)

    elif num_gt_labels == 5:
        df.loc[df['GT'].isin([0, 1, 2, 3, 4]), 'GT_new'] = df['GT']
        df.dropna(subset=['GT_new'], inplace=True)
        df['GT_new'] = df['GT_new'].astype(int)
    return df

def classify_regression_output_3_class(y_pred_values):
    """Classifies regression output into 3 classes based on thresholds."""
    def classify_single_value(value):
        if value < 0.5:
            return 0
        elif 0.5 <= value < 1.5:
            return 1
        else:
            return 2
    return np.array([classify_single_value(val) for val in y_pred_values])

def classify_regression_output_5_class(y_pred_values):
    """Classifies regression output into 5 classes based on thresholds."""
    def classify_single_value(value):
        if value < 0.5:
            return 0
        elif 0.5 <= value < 1.5:
            return 1
        elif 1.5 <= value < 2.5:
            return 2
        elif 2.5 <= value < 3.5:
            return 3
        else:
            return 4
    return np.array([classify_single_value(val) for val in y_pred_values])


def calculate_metrics(y_true, y_pred, model_type, num_gt_labels=None):
    """Calculates evaluation metrics."""
    metrics = {}

    try:
        y_true_rank = np.asarray(y_true).flatten()
        y_pred_rank = np.asarray(y_pred).flatten()
        if len(np.unique(y_pred_rank)) < 2 or len(np.unique(y_true_rank)) < 2:
             metrics['kendall_tau'] = 0.0
             metrics['kendall_pvalue'] = 1.0
        else:
            ktau, kpval = kendalltau(y_true_rank, y_pred_rank)
            metrics['kendall_tau'] = ktau if not np.isnan(ktau) else 0.0
            metrics['kendall_pvalue'] = kpval if not np.isnan(kpval) else 1.0
    except ValueError:
        metrics['kendall_tau'] = 0.0
        metrics['kendall_pvalue'] = 1.0

    try:
        if len(np.unique(y_pred_rank)) < 2 or len(np.unique(y_true_rank)) < 2: 
            metrics['spearman_rho'] = 0.0
            metrics['spearman_pvalue'] = 1.0
        else:
            srho, spval = spearmanr(y_true_rank, y_pred_rank)
            metrics['spearman_rho'] = srho if not np.isnan(srho) else 0.0
            metrics['spearman_pvalue'] = spval if not np.isnan(spval) else 1.0
    except ValueError:
        metrics['spearman_rho'] = 0.0
        metrics['spearman_pvalue'] = 1.0

    if model_type == 'regression':
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        y_pred_for_accuracy = y_pred 
        if num_gt_labels == 3:
            y_pred_classified = classify_regression_output_3_class(y_pred_for_accuracy)
        elif num_gt_labels == 5:
            y_pred_classified = classify_regression_output_5_class(y_pred_for_accuracy)
        metrics['accuracy'] = accuracy_score(y_true.astype(int), y_pred_classified.astype(int))
    elif model_type == 'classification':
        metrics['mae'] = np.nan
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

    return metrics

def evaluate_model_performance(model_name, model_instance, param_distributions,
                               X_train, y_train, X_test, y_test,
                               model_type, num_gt_labels, cv_fold=5, n_iter_search=10): 
    """Trains model using RandomizedSearchCV (if params provided) and evaluates."""
    print(f"    Evaluating {model_name} ({model_type} for {num_gt_labels} labels)...") 
    print()

    best_model = model_instance

    random_search = RandomizedSearchCV(
        estimator=model_instance,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        cv= 5 ,
        random_state=42,
        error_score='raise' 
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    eval_metrics = calculate_metrics(y_test, y_pred, model_type, num_gt_labels=num_gt_labels) 
    eval_metrics['model_name'] = model_name
    eval_metrics['model_type'] = model_type

    return eval_metrics

# --- Main Processing Function ---
def main(input_features_csv_file, output_csv_file):
    df_original = pd.read_csv(input_features_csv_file)

    all_run_results = []
    for num_labels in [3, 5]:
        print(f"\n--- Processing for {num_labels} GT labels scenario ---")
        df_processed = df_original.copy()
        df_processed = transform_gt_labels(df_processed, num_labels)

        df_processed['video_group'] = df_processed['video_name'].str.extract(r'(\d+)', expand=False)
        df_processed.dropna(subset=['video_group'], inplace=True) 
        if df_processed.empty:
            print(f"Skipping {num_labels} labels scenario as no data remains after group processing.")
            continue

        groups = df_processed['video_group']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        train_df, test_df = None, None 
        for train_idx, test_idx in gss.split(df_processed, df_processed["GT_new"], groups=groups):
            train_df = df_processed.iloc[train_idx]
            test_df = df_processed.iloc[test_idx]

        feature_cols = ['aperiodicity', 'fatigue_norm','freeze_durations',
        'speed_median', 'speed_quartile_range', 'speed_min', 'speed_max',
        'acc_median',  'acc_min', 'acc_max', 'acc_quartile_range',
        'amplitude_median', 'amplitude_quartile_range', 'amplitude_min','amplitude_max', 'amplitude_entropy',
        'period_median','period_quartile_range', 'period_min', 'period_max', 'period_entropy'
        ]
        X_train=train_df.loc[:,feature_cols]
        y_train=train_df["GT_new"]
        X_test=test_df.loc[:,feature_cols]
        y_test=test_df["GT_new"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        # Regression Models
        print(f"\n  Running Regression Models for {num_labels} labels...")

        for model_name, (model_instance, params) in REGRESSION_MODELS_PARAMS.items():
            current_y_train = y_train.astype(float) 
            current_y_test = y_test.astype(float)   

            metrics = evaluate_model_performance(model_name, model_instance, params,
                                                    X_train_scaled_df, current_y_train, X_test_scaled_df, current_y_test,
                                                    model_type='regression', num_gt_labels=num_labels, cv_fold=5) 
            if metrics:
                metrics['gt_scenario'] = f'{num_labels}_labels'
                all_run_results.append(metrics)

        # Classification Models
        print(f"\n  Running Classification Models for {num_labels} labels...")
        min_class_count_train = y_train.value_counts().min() 
        min_class_count_test = y_test.value_counts().min()

        y_train_class_counts = y_train.value_counts()
        n_splits_cv = 5 

        for model_name, (model_instance, params) in CLASSIFICATION_MODELS_PARAMS.items():
            current_y_train = y_train.astype(int) 
            current_y_test = y_test.astype(int)   
            metrics = evaluate_model_performance(model_name, model_instance, params,
                                                X_train_scaled_df, current_y_train, X_test_scaled_df, current_y_test,
                                                model_type='classification', num_gt_labels=num_labels, cv_fold=n_splits_cv) # Pass num_labels
            if metrics:
                metrics['gt_scenario'] = f'{num_labels}_labels'
                all_run_results.append(metrics)

    results_df = pd.DataFrame(all_run_results)

    cols_order = ['gt_scenario', 'model_name', 'model_type', 'mae', 'accuracy',
                'kendall_tau', 'kendall_pvalue', 'spearman_rho', 'spearman_pvalue']
    for col in cols_order:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = results_df[cols_order]
    results_df.to_csv(output_csv_file, index=False, float_format='%.4f')
    print(f'{output_csv_file} Saved!')

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML models on CSV data and evaluate.")
    parser.add_argument("--input_features_csv_file", required=True, help="Path to the input CSV file containing extracted features, GT, and video_name.") # Corrected help string
    parser.add_argument("--output_csv_file", required=True, help="Path to save the results CSV file.")

    args = parser.parse_args()

    main(args.input_features_csv_file, args.output_csv_file)
