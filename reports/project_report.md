# Project Report: Movie Recommendation System using Big Data Analytics

## 1. Abstract
The project builds a movie recommendation and analytics platform from movie metadata and user ratings. Apache Spark and PySpark process the dataset through Bronze, Silver, and Gold layers. Spark MLlib's Alternating Least Squares (ALS) collaborative-filtering algorithm learns latent user and movie factors and generates personalized recommendations. A Streamlit dashboard exposes popular movies, rating statistics, search, and recommendations.

## 2. Introduction
Streaming platforms provide large catalogs. Users can find it difficult to select movies that match their interests. Recommendation systems use historical interactions to rank items likely to interest a user. This project demonstrates a reproducible, data-engineering-oriented implementation with data quality checks, analytics, model evaluation, and a dashboard.

## 3. Problem Statement
Given historical user ratings for movies, develop a data pipeline that cleans and aggregates the data and produces personalized movie recommendations, while exposing useful descriptive analytics through an interactive dashboard.

## 4. Objectives
- Ingest a documented, real-world dataset.
- preserve source data in a Bronze layer.
- Clean and validate data in Silver.
- Build analysis-ready Gold tables using PySpark.
- Train an ALS collaborative-filtering model with Spark MLlib.
- Evaluate predictions using test-set RMSE.
- Display actual pipeline outputs and recommendations in Streamlit.

## 5. Scope
The prototype uses MovieLens ml-latest-small for local development. It demonstrates distributed-data APIs and an end-to-end workflow, but the dataset is not large enough to claim production-scale performance. It does not use live streaming or user authentication.

## 6. Dataset
Source: GroupLens Research, MovieLens datasets, https://grouplens.org/datasets/movielens/
The `ml-latest-small` archive contains `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`. Movie metadata includes movie ID, title, and genres. Ratings include user ID, movie ID, rating, and timestamp. Cite GroupLens in the submitted report and include the retrieval date.

## 7. Technologies
- Python: project orchestration and dashboard.
- Apache Spark / PySpark: distributed data processing.
- Spark SQL and DataFrame API: cleaning and aggregation.
- Spark MLlib ALS: collaborative filtering.
- Parquet: columnar intermediate/output storage.
- Streamlit: interactive dashboard.
- Pandas/PyArrow: dashboard-side reading of outputs.

## 8. System Architecture
1. Source: MovieLens CSV files.
2. Bronze: raw Parquet snapshots; source CSV files remain in `data/ml-latest-small/`.
3. Silver: typed, deduplicated movie and rating tables, plus ratings joined to metadata.
4. Gold: movie rating counts, average ratings, Bayesian-smoothed weighted scores, and user analytics.
5. ML: ALS model fitted to training ratings, tested on held-out ratings, and evaluated with RMSE.
6. Serving: top recommendations saved to Parquet and displayed in Streamlit.

## 9. Methodology
### Data ingestion
The pipeline downloads the public MovieLens archive when requested and extracts the required CSVs. It then reads them with Spark.

### Data cleaning
Movie IDs and user IDs are cast to integers, ratings to numeric values, and timestamps to long integers. Invalid identifiers, empty titles, and ratings outside the expected 0.5–5.0 range are removed. Duplicate records are dropped. Ratings without matching movie metadata are excluded from the joined table.

### Gold analytics
For every movie, calculate rating count and average rating. The weighted score shrinks low-count averages toward the global mean to reduce the effect of a movie with only a few ratings. User-level rating counts and mean ratings are also computed.

### Recommendation model
ALS factorizes the user–movie rating matrix into lower-dimensional user and movie factors. The data is split into 80% training and 20% test records. The implementation evaluates predictions using root mean squared error (RMSE). The exact value must be taken from the actual run output; it is not pre-filled here.

### Dashboard
The dashboard displays dataset KPIs, top-rated movies subject to a minimum-rating filter, rating distribution, title search, and ALS-generated personalized recommendations for a selected user ID.

## 10. Evaluation
Record these values after running the project:
- Number of movies: shown in pipeline output.
- Number of valid ratings: shown in pipeline output.
- Number of users: shown in pipeline output.
- Training and test rows: shown in model output.
- Test RMSE: shown in model output.

RMSE is the square root of the average squared difference between predicted and actual ratings. Lower RMSE indicates lower prediction error on this split, but it does not alone measure recommendation ranking quality or user satisfaction.

## 11. Expected Outcomes
- Reproducible Bronze/Silver/Gold datasets.
- Movie analytics and user analytics stored in Parquet.
- A trained ALS model and saved recommendations.
- A functioning dashboard based on actual processed data.
- A test RMSE value calculated from the held-out data.

## 12. Limitations
- New users and movies have a cold-start problem.
- Ratings can be sparse and reflect only the users represented in MovieLens.
- RMSE measures rating prediction error, not ranking quality.
- The small dataset demonstrates the architecture but does not prove large-scale throughput.
- Dashboard recommendation reading depends on compatible Spark/Parquet/PyArrow versions.

## 13. Future Enhancements
- Add ranking metrics such as Precision@K and Recall@K.
- Include content-based recommendations from genres and tags.
- Add a user feedback form and refresh model workflow.
- Schedule the pipeline with Airflow.
- Deploy the dashboard and store artifacts in cloud object storage.
- Benchmark with a larger approved dataset.

## 14. Conclusion
This project demonstrates the end-to-end use of Big Data technologies for a practical recommendation task: data ingestion, layered processing, quality control, analytics, collaborative filtering, model evaluation, and interactive presentation. Final report metrics and screenshots should be captured from a successful local run.

## References
1. GroupLens Research, MovieLens Datasets: https://grouplens.org/datasets/movielens/
2. Apache Spark Documentation: https://spark.apache.org/docs/latest/
3. Spark MLlib ALS Recommendation: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
4. Streamlit Documentation: https://docs.streamlit.io/
