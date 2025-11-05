import pandas as pd
import numpy as np
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from utils import save_plotly_fig 
from sklearn.metrics import roc_curve, auc # Import ROC functions
from sklearn.preprocessing import label_binarize


warnings.filterwarnings('ignore')

ARTIFACTS_DIR = '../artifacts/'

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create annotated heatmap
    # Plotly's figure factory is great for this
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(labels),
        y=list(labels),
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    save_plotly_fig(fig, filename)

def plot_feature_importance(model, feature_names, filename):
    """Generates and saves a feature importance bar chart."""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return
        
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    fig = px.bar(importance_df, x='importance', y='feature', 
                 title="Model Feature Importance",
                 orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    save_plotly_fig(fig, filename)


def plot_roc_auc(y_true, y_score, class_names, filename):

    print(f"Generating ROC-AUC plot for {len(class_names)} classes...")
    
    # Binarize the true labels
    n_classes = len(class_names)
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Initialize Plotly figure
    fig = go.Figure()
    
    # Add a dashed "no-skill" line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        # Get probabilities for this class
        class_scores = y_score[:, i]
        
        # Get true labels for this class
        class_true = y_true_binarized[:, i]
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(class_true, class_scores)
        roc_auc = auc(fpr, tpr)
        
        # Add trace to the figure
        name = f"{class_names[i]} (AUC = {roc_auc:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        title='ROC-AUC Curve (One-vs-Rest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Classes'
    )
    save_plotly_fig(fig, filename)



def load_data():
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')             
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')

    target_encoder = joblib.load('../artifacts/target_encoder.joblib')
    class_names = target_encoder.classes_.tolist()

    return X_train, y_train, X_test, y_test, class_names, X_train.columns

def run_model():
    X_train, y_train, X_test, y_test, class_names, feature_names = load_data()


    models = {
        "Baseline (Stratified)": DummyClassifier(strategy='stratified', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    params_grid = {
        "Baseline (Stratified)": {
        }, # No params to tune
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        },
        "K-Nearest Neighbors": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        "Random Forest": {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "XGBoost": {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    }

    results = {}

    print("Starting model training and evaluation...")
    for name, model in tqdm(models.items(), desc="Overall Model Training"):
        print(f"\nTraining {name}")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params_grid[name],
            cv=5,
            scoring='f1_macro',
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        
        y_score = None
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        results[name] = {
            "accuracy": accuracy,
            "f1-score (macro)": report['macro avg']['f1-score'],
            "model": best_model, 
            "predictions": y_pred,
            "probabilities": y_score, 
            "best_params": grid_search.best_params_
        }
        
        print(f"--- {name} Results ---")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV F1-Macro Score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

    best_model_name = max(results, key=lambda k: results[k]['f1-score (macro)'])
    best_model = results[best_model_name]['model']
    best_predictions = results[best_model_name]['predictions']

    print(f"\n--- Best Model Selection ---")
    print(f"Best model: {best_model_name}")
    print(f"Test Macro F1: {results[best_model_name]['f1-score (macro)']:.4f}")
    print(f"Best Hyperparameters: {results[best_model_name]['best_params']}")

    # Save results table for README
    results_summary = {
        name: {
            'Accuracy': res['accuracy'], 
            'Macro F1-Score': res['f1-score (macro)'],
            'Best Params': str(res['best_params'])
        } 
        for name, res in results.items()
    }
    results_df = pd.DataFrame.from_dict(results_summary, orient='index')
    print("\nModel Comparison:")
    print(results_df)
    results_df.to_csv('../model_comparison.csv')

    # 4. Generate and Save Evaluation Plots for Best Model
    print("Generating evaluation plots for best model...")
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, best_predictions, class_names, 
                          '../visuals/06_confusion_matrix.png')
                          
    # Feature Importance (if applicable)
    plot_feature_importance(best_model, feature_names, 
                            '../visuals/07_feature_importance.png')

    # ROC-AUC Curve (if applicable)
    best_probabilities = results[best_model_name]['probabilities']
    if best_probabilities is not None:
        plot_roc_auc(y_test, best_probabilities, class_names,
                     '../visuals/08_roc_auc_curve.png')
    else:
        print(f"Could not generate ROC-AUC curve; '{best_model_name}' does not have 'predict_proba'.")

    # 5. Save the Best Model
    model_save_path = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model ({best_model_name}) saved to {model_save_path}")

    print("--- Modeling Complete ---")

if __name__ == "__main__":
    run_model()