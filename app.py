import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Classification")
st.write(
    "A Machine Learning project to classify Iris flowers "
    "using their sepal and petal measurements."
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():

    file_path = Path(__file__).parent / "Iris.csv"

    if file_path.exists():

        df = pd.read_csv(file_path)

        # Remove unnecessary ID column
        id_columns = [
            col for col in df.columns
            if col.lower() in ["id", "unnamed: 0"]
        ]

        df = df.drop(columns=id_columns, errors="ignore")

        # Identify target column
        target_candidates = [
            col for col in df.columns
            if col.lower() in ["species", "target", "class"]
        ]

        if not target_candidates:
            raise ValueError(
                "Target column not found. "
                "Expected a column named Species, Target, or Class."
            )

        target_column = target_candidates[0]

        # Select numeric feature columns
        feature_columns = [
            col for col in df.select_dtypes(include=np.number).columns
        ]

        if len(feature_columns) != 4:
            raise ValueError(
                "Expected exactly 4 numeric flower measurement columns."
            )

        df = df.dropna(subset=feature_columns + [target_column])

        return df, feature_columns, target_column

    else:

        iris = load_iris()

        df = pd.DataFrame(
            iris.data,
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        df["Species"] = [
            iris.target_names[i]
            for i in iris.target
        ]

        feature_columns = list(df.columns[:4])

        return df, feature_columns, "Species"


try:
    df, feature_columns, target_column = load_dataset()

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODELS
# --------------------------------------------------

@st.cache_resource
def train_models(X_train, y_train):

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ])
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


models = train_models(X_train, y_train)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

model_results = {}

for name, model in models.items():

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    model_results[name] = {
        "Model": name,
        "Accuracy": accuracy,
        "Predictions": predictions
    }


best_model_name = max(
    model_results,
    key=lambda name: model_results[name]["Accuracy"]
)

best_model = models[best_model_name]


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌸 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Dataset Analysis",
        "Model Performance",
        "Flower Prediction"
    ]
)


# ==================================================
# PAGE 1: HOME
# ==================================================

if page == "Home":

    st.header("Welcome to Iris Flower Classification")

    st.write("""
    This application uses Machine Learning algorithms
    to predict the species of Iris flowers.

    The prediction is based on four measurements:

    1. Sepal Length
    2. Sepal Width
    3. Petal Length
    4. Petal Width

    The application compares three Machine Learning models
    and uses the best-performing model for prediction.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric(
            "Flower Species",
            df[target_column].nunique()
        )

    with col3:
        st.metric(
            "Best Model",
            best_model_name
        )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Flower Species Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x=target_column,
        ax=ax
    )

    ax.set_title("Distribution of Iris Flower Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    plt.close(fig)


# ==================================================
# PAGE 2: DATASET ANALYSIS
# ==================================================

elif page == "Dataset Analysis":

    st.header("📊 Dataset Analysis")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Species Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x=target_column,
        ax=ax
    )

    ax.set_title("Flower Species Distribution")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Select Feature",
        feature_columns
    )

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x=selected_feature,
        hue=target_column,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {selected_feature}")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        df[feature_columns].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)


# ==================================================
# PAGE 3: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    st.subheader("Model Accuracy Comparison")

    results_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": result["Accuracy"]
        }
        for name, result in model_results.items()
    ])

    results_df["Accuracy (%)"] = (
        results_df["Accuracy"] * 100
    ).round(2)

    st.dataframe(
        results_df[["Model", "Accuracy (%)"]],
        use_container_width=True
    )

    fig, ax = plt.subplots()

    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    ax.set_title("Machine Learning Model Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)

    st.pyplot(fig)

    plt.close(fig)

    st.success(
        f"Best Performing Model: {best_model_name}"
    )

    st.subheader("Select Model for Evaluation")

    selected_model_name = st.selectbox(
        "Choose a Model",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    predictions = selected_model.predict(X_test)

    st.write(
        f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=selected_model.classes_
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=selected_model.classes_,
        yticklabels=selected_model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# ==================================================
# PAGE 4: FLOWER PREDICTION
# ==================================================

elif page == "Flower Prediction":

    st.header("🌺 Iris Flower Prediction")

    st.write(
        "Enter the flower measurements below "
        "to predict its species."
    )

    col1, col2 = st.columns(2)

    input_values = {}

    for index, feature in enumerate(feature_columns):

        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        mean_value = float(df[feature].mean())

        if index % 2 == 0:

            with col1:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=mean_value,
                    step=0.1
                )

        else:

            with col2:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=mean_value,
                    step=0.1
                )

    st.divider()

    prediction_model_name = st.selectbox(
        "Choose Prediction Model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )

    if st.button("🔍 Predict Flower Species"):

        input_data = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        prediction_model = models[prediction_model_name]

        prediction = prediction_model.predict(input_data)

        st.subheader("Prediction Results")

        st.success(
            f"Predicted Species: {prediction[0]}"
        )

        if hasattr(prediction_model, "predict_proba"):

            probabilities = prediction_model.predict_proba(
                input_data
            )[0]

            st.subheader("Prediction Confidence")

            probability_df = pd.DataFrame({
                "Species": prediction_model.classes_,
                "Probability (%)": probabilities * 100
            })

            probability_df["Probability (%)"] = (
                probability_df["Probability (%)"].round(2)
            )

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            fig, ax = plt.subplots()

            sns.barplot(
                data=probability_df,
                x="Species",
                y="Probability (%)",
                ax=ax
            )

            ax.set_title("Prediction Probability")

            st.pyplot(fig)

            plt.close(fig)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
    "Iris Flower Classification\n\n"
    "Machine Learning Project\n\n"
    "Python | Pandas | NumPy | Scikit-learn | Streamlit"
)

st.markdown("---")

st.caption(
    "Iris Flower Classification | Machine Learning Project"
)
