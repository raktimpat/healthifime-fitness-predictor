

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils import save_plotly_fig 

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
