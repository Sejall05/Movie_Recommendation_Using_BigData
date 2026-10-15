
from pathlib import Path
import shutil

import pandas as pd
import pyarrow.parquet as pq

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from src.common import OUTPUT_DIR, make_spark, ensure_dirs
from src.local_writer import write_parquet_local


def train_and_recommend():
    ensure_dirs()

    gold_ratings_path = OUTPUT_DIR / "gold" / "ratings_training"
    parquet_file = gold_ratings_path / "part-00000.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(
            "Gold training data not found. "
            "Run `python run_pipeline.py` first."
        )

    # Read Parquet with PyArrow instead of Spark's Hadoop-based reader.
    pandas_data = pq.read_table(parquet_file).select(
        ["userId", "movieId", "rating"]
    ).to_pandas()

    pandas_data = pandas_data.dropna(
        subset=["userId", "movieId", "rating"]
    )

    pandas_data["userId"] = pandas_data["userId"].astype("int32")
    pandas_data["movieId"] = pandas_data["movieId"].astype("int32")
    pandas_data["rating"] = pandas_data["rating"].astype("float32")

    spark = make_spark("MovieLensALSRecommendation")

    try:
        # Create the Spark DataFrame from the locally loaded data.
        data = spark.createDataFrame(pandas_data)

        train, test = data.randomSplit([0.8, 0.2], seed=42)

        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=10,
            maxIter=10,
            regParam=0.1,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42,
        )

        print("Training ALS recommendation model...")
        model = als.fit(train)

        predictions = model.transform(test)

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )

        rmse = evaluator.evaluate(predictions)

        model_dir = OUTPUT_DIR / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model factors without Spark's Hadoop output writer.
        write_parquet_local(
            model.userFactors,
            model_dir / "user_factors",
        )

        write_parquet_local(
            model.itemFactors,
            model_dir / "item_factors",
        )

        # Save personalized recommendations for users and movies.
        user_recommendations = model.recommendForAllUsers(10)
        item_recommendations = model.recommendForAllItems(10)

        write_parquet_local(
            user_recommendations,
            model_dir / "user_recommendations",
        )

        write_parquet_local(
            item_recommendations,
            model_dir / "item_recommendations",
        )

        # Save evaluation metrics in a local JSON file.
        metrics_dir = model_dir / "metrics"

        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)

        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics = pd.DataFrame([{
            "rmse": float(rmse),
            "train_rows": int(train.count()),
            "test_rows": int(test.count()),
        }])

        metrics.to_json(
            metrics_dir / "part-00000.json",
            orient="records",
            lines=True,
        )

        print(f"\nALS model trained successfully.")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training rows: {train.count():,}")
        print(f"Test rows: {test.count():,}")
        print(f"Recommendations and metrics saved under: {model_dir}")

    finally:
        spark.stop()
