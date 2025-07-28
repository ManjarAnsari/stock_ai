import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data(show_spinner=False)
def shap_feature_importance(model, X_train, max_display=10):
    """
    Generate SHAP feature importance for a trained model.

    Parameters:
    - model: Trained tree-based model (like XGBoost or LightGBM)
    - X_train: Training features (DataFrame)
    - max_display: Number of top features to display

    Returns:
    - List of top important features
    """
    try:
        # Compute SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # SHAP Summary Plot
        st.subheader("🔍 SHAP Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=max_display)
        st.pyplot(fig)

        # Compute top features
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': shap_values.abs.mean(0).values
        }).sort_values(by="importance", ascending=False)

        top_features = importance_df['feature'].head(max_display).tolist()
        return top_features

    except Exception as e:
        st.warning(f"SHAP computation failed: {e}")
        return X_train.columns[:max_display].tolist()