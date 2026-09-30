from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import io

app = Flask(__name__)

PCA_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score"
]

latest_result = None


@app.route("/", methods=["GET", "POST"])
def index():
    global latest_result

    if request.method == "GET":
        return render_template("index.html", analyzed=False)

    file = request.files.get("file")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            analyzed=False,
            error="Please upload a CSV file."
        )

    try:

        # Read CSV
        df = pd.read_csv(file)

        original_rows = len(df)
        original_columns = len(df.columns)

        # Remove duplicate rows
        duplicate_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

        # Missing values before cleaning
        missing_before = int(df.isnull().sum().sum())

        # Handle missing values
        for column in df.columns:

            if df[column].dtype == "object":

                mode_value = df[column].mode()

                if not mode_value.empty:
                    df[column] = df[column].fillna(
                        mode_value.iloc[0]
                    )
                else:
                    df[column] = df[column].fillna("Unknown")

            else:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                mean_value = df[column].mean()

                if pd.notna(mean_value):
                    df[column] = df[column].fillna(
                        mean_value
                    )
                else:
                    df[column] = df[column].fillna(0)

        # Missing values after cleaning
        missing_after = int(df.isnull().sum().sum())

        # Check required PCA features
        missing_features = [
            feature
            for feature in PCA_FEATURES
            if feature not in df.columns
        ]

        if missing_features:

            return render_template(
                "index.html",
                analyzed=False,
                error=(
                    "Required PCA columns are missing: "
                    + ", ".join(missing_features)
                )
            )

        # Select PCA features
        X = df[PCA_FEATURES].copy()

        # Convert to numeric
        for column in PCA_FEATURES:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        # Fill remaining missing values
        X = X.fillna(X.mean())
        X = X.fillna(0)

        # Standardization
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # Full PCA for explained variance
        pca_full = PCA()

        pca_full.fit(X_scaled)

        explained_variance = (
            pca_full.explained_variance_ratio_ * 100
        )

        cumulative_variance = (
            np.cumsum(
                pca_full.explained_variance_ratio_
            ) * 100
        )

        # Use 6 principal components
        n_components = min(
            6,
            X_scaled.shape[1],
            X_scaled.shape[0]
        )

        pca = PCA(
            n_components=n_components
        )

        principal_components = pca.fit_transform(
            X_scaled
        )

        retained_variance = round(
            pca.explained_variance_ratio_.sum() * 100,
            2
        )

        # PCA column names
        pca_columns = [
            f"PC{i}"
            for i in range(1, n_components + 1)
        ]

        # PCA result dataframe
        pca_result = pd.DataFrame(
            principal_components,
            columns=pca_columns
        )

        pca_result.insert(
            0,
            "Record",
            range(1, len(pca_result) + 1)
        )

        latest_result = pca_result.copy()

        # Explained variance table
        variance_table = []

        for i in range(n_components):

            variance_table.append({
                "component": f"PC{i + 1}",
                "variance": round(
                    explained_variance[i],
                    2
                ),
                "cumulative": round(
                    cumulative_variance[i],
                    2
                )
            })

        # PCA loadings
        loadings = pca.components_.T

        loading_table = []

        for i, feature in enumerate(PCA_FEATURES):

            row = {
                "feature": feature
            }

            for j in range(n_components):

                row[f"PC{j + 1}"] = round(
                    loadings[i, j],
                    4
                )

            loading_table.append(row)

        # PCA graph
        graph_df = pd.DataFrame({
            "PC1": principal_components[:, 0],
            "PC2": principal_components[:, 1]
        })

        fig = px.scatter(
            graph_df,
            x="PC1",
            y="PC2",
            title="PCA Visualization: PC1 vs PC2"
        )

        fig.update_layout(
            template="plotly_white",
            height=500
        )

        graph_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        )

        # Dataset preview
        preview = df.head(10).to_html(
            classes="data-table",
            index=False
        )

        # Feature counts
        numeric_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_count = len(
            df.select_dtypes(
                include="object"
            ).columns
        )

        return render_template(
            "index.html",
            analyzed=True,
            rows=len(df),
            columns=len(df.columns),
            original_rows=original_rows,
            original_columns=original_columns,
            duplicate_count=duplicate_count,
            missing_before=missing_before,
            missing_after=missing_after,
            numeric_count=numeric_count,
            categorical_count=categorical_count,
            preview=preview,
            variance_table=variance_table,
            retained_variance=retained_variance,
            n_components=n_components,
            loading_table=loading_table,
            graph_html=graph_html,
            pca_features=PCA_FEATURES,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            analyzed=False,
            error=(
                "Error while analyzing dataset: "
                + str(e)
            )
        )


@app.route("/download")
def download():

    global latest_result

    if latest_result is None:
        return "Please analyze a dataset first."

    output = io.BytesIO()

    latest_result.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="PCA_Analysis_Result.csv"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )