# Movie Recommendation System using Big Data Analytics

A Big Data Analytics project that analyzes movie ratings and generates personalized movie recommendations using **PySpark, Apache Spark MLlib, ALS collaborative filtering, and Streamlit**.

The project uses the real MovieLens `ml-latest-small` dataset from GroupLens. It implements a Bronze–Silver–Gold data pipeline, performs movie-rating analytics, trains a recommendation model, evaluates prediction accuracy, and displays the results through an interactive dashboard.

## 1. Project Objectives

- Process movie and user-rating data using PySpark.
- Implement a Bronze–Silver–Gold data architecture.
- Clean, validate, and transform raw movie-rating records.
- Analyze movie popularity, ratings, and user preferences.
- Build a recommendation model using Spark MLlib ALS.
- Evaluate model performance using Root Mean Squared Error (RMSE).
- Generate personalized movie recommendations.
- Visualize analytics and model outputs using Streamlit.

## 2. Features

- **Real dataset:** Uses MovieLens data instead of fabricated records.
- **Data ingestion:** Loads the dataset from CSV files.
- **Bronze layer:** Preserves raw source data and creates Parquet snapshots.
- **Silver layer:** Cleans records, validates data, removes duplicates, and prepares structured datasets.
- **Gold layer:** Produces analysis-ready datasets and aggregated movie statistics.
- **PySpark processing:** Performs distributed data transformations and aggregations.
- **Recommendation model:** Uses Spark MLlib ALS collaborative filtering.
- **Model evaluation:** Calculates RMSE using held-out test data.
- **Recommendation generation:** Produces recommendations for users and related item recommendations.
- **Interactive dashboard:** Displays movie information, analytics, personalized recommendations, and model metrics.
- **Parquet storage:** Stores processed data and model outputs in a columnar format.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PySpark | Big data processing |
| Apache Spark MLlib | ALS recommendation model |
| Pandas | Data handling and dashboard tables |
| NumPy | Numerical operations |
| PyArrow | Parquet file reading and writing where used |
| Streamlit | Interactive dashboard |
| Plotly / Streamlit charts | Data visualization, depending on the implementation |
| Java | Runtime required by Apache Spark |
| Git and GitHub | Version control and project hosting |

## 4. Dataset

The project uses the **MovieLens `ml-latest-small` dataset**, provided by GroupLens Research.

Dataset page: https://grouplens.org/datasets/movielens/

Dataset download: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

The dataset contains movie metadata and user ratings.

### Dataset files

| File | Description |
|---|---|
| `movies.csv` | Movie IDs, titles, and genres |
| `ratings.csv` | User IDs, movie IDs, ratings, and timestamps |
| `tags.csv` | User-generated tags for movies |
| `links.csv` | External movie identifiers |

The core recommendation pipeline uses `movies.csv` and `ratings.csv`. The other files are optional unless required by a particular implementation.

### Dataset statistics

The MovieLens `ml-latest-small` dataset contains approximately:

- **9,742 movies**
- **100,836 ratings**
- **610 users**

These are the standard dataset counts. The number of records retained after cleaning or used for model training may differ.

## 5. System Requirements

Recommended environment:

- Windows 10 or Windows 11
- Python 3.10–3.12
- A Java Development Kit compatible with the installed Apache Spark version
- 8 GB RAM recommended
- Visual Studio Code or another Python IDE
- Internet access for the initial dataset download and any dependencies not already installed

**Note:** The project has been run in a Windows environment with Python 3.12 and Java 22. Keep the currently working environment unless a compatibility issue requires a change.

## 6. Project Folder Structure

```text
movie_recommendation_big_data/
│
├── data/
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       └── links.csv
│
├── outputs/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── model/
│       ├── user_factors/
│       ├── item_factors/
│       ├── user_recommendations/
│       ├── item_recommendations/
│       └── metrics/
│
├── src/
│   ├── common.py
│   ├── pipeline.py
│   ├── local_writer.py
│   └── recommender.py
│
├── dashboard.py
├── run_pipeline.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── reports/
    ├── project_report.md
    └── presentation_outline.md
```

The exact files and output directories may vary slightly according to the current implementation. Only include files and folders that actually exist in your repository.

## 7. Installation and Setup

### Step 1: Extract the project

Extract the project ZIP file and open the project folder in Visual Studio Code.

Open the VS Code terminal using **Terminal → New Terminal**.

### Step 2: Create a virtual environment

```powershell
py -m venv .venv
```

### Step 3: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify Python and Java

```powershell
python --version
java -version
```

Make sure Python and Java are available in the terminal and that the Java version is compatible with your installed Spark release.

## 8. Running the Project

Run the following commands from the project root directory.

### Step 1: Download the dataset and build data layers

```powershell
python run_pipeline.py --download
```

This command downloads the MovieLens dataset if necessary and executes the data pipeline.

If the dataset is already present, you can use:

```powershell
python run_pipeline.py
```

### Step 2: Train the recommendation model

If your project contains `train_model.py`, run:

```powershell
python train_model.py
```

This step trains the ALS model, evaluates predictions, and writes recommendation outputs and metrics.

If your current implementation runs model training through another script or through the pipeline, use the command implemented in that file instead.

### Step 3: Launch the dashboard

```powershell
streamlit run dashboard.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the dashboard.

## 9. Data Pipeline Architecture

The project follows the **Bronze–Silver–Gold (Medallion) architecture**.

### Bronze Layer — Raw Data

The Bronze layer preserves the source data and creates raw Parquet snapshots.

Activities include:

- Reading the source CSV files.
- Retaining the original dataset files.
- Ingesting raw records into the Bronze layer.
- Preserving source information where supported by the implementation.

### Silver Layer — Cleaned Data

The Silver layer prepares validated and structured data for analysis.

Activities include:

- Validating required columns and data types.
- Removing duplicate records.
- Filtering invalid user IDs and movie IDs.
- Validating rating values.
- Joining ratings with movie metadata where required.
- Writing cleaned data in Parquet format.

### Gold Layer — Analysis-Ready Data

The Gold layer contains datasets prepared for analytical queries and recommendation modelling.

Activities include:

- Calculating movie-level statistics.
- Aggregating rating information.
- Preparing cleaned rating datasets.
- Producing datasets used by downstream model training and dashboard visualizations.

### Pipeline Flow

```text
MovieLens CSV Dataset
         |
         v
   Data Ingestion
         |
         v
    Bronze Layer
   Raw Parquet Data
         |
         v
    Silver Layer
 Cleaned and Validated Data
         |
         v
     Gold Layer
 Analysis-Ready Datasets
         |
         v
   ALS Recommendation Model
         |
         v
 Predictions and Evaluation
         |
         v
   Streamlit Dashboard
```

## 10. PySpark Processing

PySpark is used to process movie and rating data with Apache Spark.

Depending on the implemented pipeline, processing operations include:

- Reading structured datasets.
- Selecting and filtering records.
- Validating data.
- Joining ratings with movie metadata.
- Removing duplicate or invalid records.
- Aggregating ratings and movie-level statistics.
- Writing processed datasets to Parquet.

The distributed processing approach provides a scalable foundation that can be extended to larger datasets.

## 11. Recommendation Algorithm

The project uses **Alternating Least Squares (ALS)**, a collaborative filtering algorithm available in Apache Spark MLlib.

Collaborative filtering identifies patterns in user-rating behaviour to estimate how much a user may like a movie.

### How ALS Works

1. The dataset provides user IDs, movie IDs, and ratings.
2. The ratings are divided into training and test datasets.
3. ALS learns latent factors representing users and movies.
4. The model predicts ratings for user–movie combinations.
5. Predicted ratings are evaluated against held-out ratings.
6. The model generates recommendations for eligible users and items.

### Why ALS?

- Designed for collaborative filtering.
- Available directly in Spark MLlib.
- Works with explicit user-rating data.
- Can process rating datasets using Spark.
- Provides predicted ratings that can be evaluated quantitatively.

## 12. Model Evaluation

The model is evaluated using **Root Mean Squared Error (RMSE)**.

RMSE measures the difference between actual ratings and predicted ratings. Lower RMSE generally indicates smaller prediction errors on the evaluated dataset.

The formula is:

RMSE = √[Σ(actual rating − predicted rating)² / number of predictions]

### Recorded Project Results

| Metric | Result |
|---|---:|
| Model | Spark MLlib ALS |
| Training ratings | 80,916 |
| Test ratings | 19,920 |
| Test RMSE | 0.8687 |

These results are from the recorded model run. If you retrain the model with a different random split or configuration, update the figures using the new evaluation output.

The RMSE result represents rating-prediction error; it does not by itself establish the quality of the recommendation ranking for every user.

## 13. Streamlit Dashboard

The dashboard reads the project's generated datasets, recommendation outputs, and model metrics.

It contains four main pages.

### 1. Overview

Displays key dataset statistics, movie-rating analytics, and popular or highly rated movies.

### 2. Explore Movies

Allows users to browse and search movie information and filter available movies by genre.

### 3. Personalized Recommendations

Allows users to select a user ID and view movie recommendations generated by the model.

### 4. Model Performance

Displays the model evaluation metrics, training and testing statistics, and available output information.

The dashboard is intended to provide an accessible interface for exploring the results of the data pipeline and recommendation model.

## 14. Output Files

The project stores its outputs under the `outputs/` directory.

| Output | Purpose |
|---|---|
| `outputs/bronze/` | Raw data snapshots |
| `outputs/silver/` | Cleaned and validated data |
| `outputs/gold/` | Aggregated and analysis-ready datasets |
| `outputs/model/user_factors/` | Learned user latent factors |
| `outputs/model/item_factors/` | Learned movie latent factors |
| `outputs/model/user_recommendations/` | User-oriented recommendation results |
| `outputs/model/item_recommendations/` | Item recommendation results |
| `outputs/model/metrics/` | Model evaluation metrics |

The exact output names depend on the implementation and successful execution of the relevant pipeline and model scripts.

## 15. Troubleshooting

### Dataset not found

Confirm that the required files are present in:

```text
data/ml-latest-small/
```

At minimum, the main pipeline requires `movies.csv` and `ratings.csv`.

### Python package errors

Activate the virtual environment and reinstall the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Java or Spark errors

Check the installed Java version and ensure it is compatible with the Spark version in the project. Configure `JAVA_HOME` if necessary.

### Windows Hadoop warnings

Spark may display warnings about missing `winutils.exe` or native Hadoop libraries on Windows.

Some local operations can still work despite these warnings. Whether a warning is harmless depends on the operation and the resulting output. Check that the pipeline completes and that its expected files are generated.

### No recommendations appear

Check that:

- The model training step completed successfully.
- The recommendation output files exist.
- The selected user has recommendation results.
- The dashboard is reading the correct output directories.

### Dashboard cannot read Parquet files

Verify that the required Parquet outputs exist and that the installed Pandas/PyArrow dependencies are available.

## 16. Limitations

- MovieLens `ml-latest-small` is an educational dataset, not a production-scale big data source.
- Recommendations depend on the available user-rating history.
- ALS may not generate personalized results for users or movies absent from the training data.
- New users and movies may suffer from the cold-start problem.
- Model performance depends on the training/test split and configuration.
- This prototype does not provide production-grade real-time recommendations.
- Workflow orchestration and automated scheduling are not included unless separately configured.

## 17. Future Enhancements

Possible improvements include:

- Integrating additional movie metadata.
- Adding content-based recommendations.
- Combining collaborative and content-based filtering.
- Improving recommendation ranking evaluation.
- Adding automated model retraining.
- Implementing scheduled orchestration using Apache Airflow.
- Deploying the dashboard to a hosted environment.
- Extending the pipeline to larger datasets.

## 18. Academic Use

This project demonstrates an end-to-end educational application of Big Data Analytics, including data ingestion, data cleaning, layered storage, distributed processing, machine learning, model evaluation, and dashboard development.

It is intended for academic learning and demonstration rather than production deployment.

## 19. Author and Repository

**Project Title:** Movie Recommendation System using Big Data Analytics

**Developed using:** Python, PySpark, Apache Spark MLlib, ALS collaborative filtering, Parquet, and Streamlit.
