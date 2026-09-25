from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import io

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

app = Flask(__name__)

# Store latest PCA result
latest_result = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global latest_result

    file = request.files.get("dataset")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            error="No CSV file selected."
        )

    try:

        # ==========================================
        # 1. READ DATASET
        # ==========================================

        df = pd.read_csv(file)

        rows, columns = df.shape

        # ==========================================
        # 2. DATASET INFORMATION
        # ==========================================

        missing_values = int(
            df.isnull().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        # ==========================================
        # 3. IDENTIFY DATA TYPES
        # ==========================================

        numerical_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        # ==========================================
        # 4. REMOVE DUPLICATES
        # ==========================================

        df_clean = df.drop_duplicates().copy()

        # ==========================================
        # 5. HANDLE NUMERICAL MISSING VALUES
        # ==========================================

        for column in numerical_columns:

            if df_clean[column].isnull().any():

                mean_value = df_clean[column].mean()

                df_clean[column] = df_clean[column].fillna(
                    mean_value
                )

        # ==========================================
        # 6. HANDLE CATEGORICAL MISSING VALUES
        # ==========================================

        for column in categorical_columns:

            if df_clean[column].isnull().any():

                mode_value = df_clean[column].mode()

                if len(mode_value) > 0:

                    df_clean[column] = df_clean[column].fillna(
                        mode_value[0]
                    )

                else:

                    df_clean[column] = df_clean[column].fillna(
                        "Unknown"
                    )

        # ==========================================
        # 7. DATASET PREVIEW
        # ==========================================

        preview = df.head(10).to_html(
            classes="data-table",
            index=False
        )

        # ==========================================
        # 8. CHECK FEATURES
        # ==========================================

        if (
            len(numerical_columns) == 0
            and
            len(categorical_columns) == 0
        ):

            return render_template(
                "index.html",
                error="Dataset does not contain usable features."
            )

        # ==========================================
        # 9. CREATE PREPROCESSING PIPELINE
        # ==========================================

        transformers = []

        # Numerical features
        if len(numerical_columns) > 0:

            transformers.append(
                (
                    "numerical",
                    StandardScaler(),
                    numerical_columns
                )
            )

        # Categorical features
        if len(categorical_columns) > 0:

            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical_columns
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        # ==========================================
        # 10. APPLY PREPROCESSING
        # ==========================================

        processed_data = preprocessor.fit_transform(
            df_clean
        )

        processed_data = np.asarray(
            processed_data,
            dtype=float
        )

        # ==========================================
        # 11. CHECK PCA REQUIREMENTS
        # ==========================================

        if processed_data.shape[0] < 2:

            return render_template(
                "index.html",
                rows=rows,
                columns=columns,
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                preview=preview,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                error="At least two rows are required for PCA."
            )

        if processed_data.shape[1] < 2:

            return render_template(
                "index.html",
                rows=rows,
                columns=columns,
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                preview=preview,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                error="At least two usable features are required for PCA."
            )

        # ==========================================
        # 12. APPLY PCA
        # ==========================================

        pca = PCA(
            n_components=2
        )

        principal_components = pca.fit_transform(
            processed_data
        )

        # ==========================================
        # 13. EXPLAINED VARIANCE
        # ==========================================

        explained_variance = (
            pca.explained_variance_ratio_ * 100
        )

        pc1_variance = round(
            float(explained_variance[0]),
            2
        )

        pc2_variance = round(
            float(explained_variance[1]),
            2
        )

        total_variance = round(
            pc1_variance + pc2_variance,
            2
        )

        # ==========================================
        # 14. PCA RESULT DATAFRAME
        # ==========================================

        pca_df = pd.DataFrame(
            principal_components,
            columns=[
                "PC1",
                "PC2"
            ]
        )

        pca_df.insert(
            0,
            "Record",
            range(1, len(pca_df) + 1)
        )

        pca_preview = pca_df.head(10).to_html(
            classes="data-table",
            index=False
        )

        # ==========================================
        # 15. GRAPH DATA
        # ==========================================

        pc1_values = [
            round(float(value), 4)
            for value in principal_components[:, 0]
        ]

        pc2_values = [
            round(float(value), 4)
            for value in principal_components[:, 1]
        ]

        # ==========================================
        # 16. PCA FEATURE LOADINGS
        # ==========================================

        feature_names = []

        if len(numerical_columns) > 0:

            feature_names.extend(
                numerical_columns
            )

        if len(categorical_columns) > 0:

            encoder = preprocessor.named_transformers_[
                "categorical"
            ]

            encoded_names = encoder.get_feature_names_out(
                categorical_columns
            )

            feature_names.extend(
                encoded_names.tolist()
            )

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[
                "PC1",
                "PC2"
            ],
            index=feature_names
        )

        loadings = loadings.round(4)

        loadings_table = loadings.to_html(
            classes="data-table"
        )

        # ==========================================
        # 17. PREPROCESSING INFORMATION
        # ==========================================

        preprocessing_status = (
            "Missing values handled, "
            "duplicate rows removed, "
            "categorical data encoded, "
            "and numerical data standardized."
        )

        processed_feature_count = (
            processed_data.shape[1]
        )

        # ==========================================
        # 18. SAVE RESULT FOR DOWNLOAD
        # ==========================================

        latest_result = pca_df.copy()

        # ==========================================
        # 19. SEND RESULTS TO WEBSITE
        # ==========================================

        return render_template(
            "index.html",

            rows=rows,
            columns=columns,

            missing_values=missing_values,
            duplicate_rows=duplicate_rows,

            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,

            preview=preview,

            preprocessing_status=preprocessing_status,

            processed_feature_count=processed_feature_count,

            pc1_variance=pc1_variance,
            pc2_variance=pc2_variance,
            total_variance=total_variance,

            pca_preview=pca_preview,

            loadings_table=loadings_table,

            pc1_values=pc1_values,
            pc2_values=pc2_values
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error while processing dataset: {e}"
        )


# ==========================================
# DOWNLOAD PCA RESULT
# ==========================================

@app.route("/download")
def download():

    global latest_result

    if latest_result is None:

        return (
            "Please upload and analyze a dataset first."
        )

    output = io.StringIO()

    latest_result.to_csv(
        output,
        index=False
    )

    output.seek(0)

    file_data = io.BytesIO(
        output.getvalue().encode("utf-8")
    )

    file_data.seek(0)

    return send_file(
        file_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="PCA_Analysis_Result.csv"
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )