
from src.common import DATA_DIR, OUTPUT_DIR, make_spark, ensure_dirs
from src.local_writer import write_parquet_local


def build_layers():
    ensure_dirs()

    required = [
        DATA_DIR / "movies.csv",
        DATA_DIR / "ratings.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "MovieLens CSV files not found: "
            + ", ".join(missing)
            + ". Run `python run_pipeline.py --download` first."
        )

    spark = make_spark("MovieLensBronzeSilverGoldPipeline")

    try:
        # BRONZE: read the original CSV files and save raw snapshots.
        raw_movies = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(DATA_DIR / "movies.csv"))
        )

        raw_ratings = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(DATA_DIR / "ratings.csv"))
        )

        bronze = OUTPUT_DIR / "bronze"
        write_parquet_local(raw_movies, bronze / "movies_raw")
        write_parquet_local(raw_ratings, bronze / "ratings_raw")

        # SILVER: correct data types, remove invalid rows and duplicates.
        movies = (
            raw_movies.selectExpr(
                "CAST(movieId AS INT) movieId",
                "CAST(title AS STRING) title",
                "CAST(genres AS STRING) genres",
            )
            .filter(
                "movieId IS NOT NULL "
                "AND title IS NOT NULL "
                "AND length(trim(title)) > 0"
            )
            .dropDuplicates(["movieId"])
        )

        ratings = (
            raw_ratings.selectExpr(
                "CAST(userId AS INT) userId",
                "CAST(movieId AS INT) movieId",
                "CAST(rating AS DOUBLE) rating",
                "CAST(timestamp AS LONG) timestamp",
            )
            .filter(
                "userId IS NOT NULL "
                "AND movieId IS NOT NULL "
                "AND rating BETWEEN 0.5 AND 5.0"
            )
            .dropDuplicates(["userId", "movieId", "rating", "timestamp"])
        )

        ratings_with_movies = ratings.join(movies, "movieId", "inner")

        silver = OUTPUT_DIR / "silver"
        write_parquet_local(movies, silver / "movies")
        write_parquet_local(ratings, silver / "ratings")
        write_parquet_local(
            ratings_with_movies,
            silver / "ratings_with_movies",
        )

        # GOLD: calculate movie popularity and rating analytics.
        from pyspark.sql import functions as F

        stats = ratings.groupBy("movieId").agg(
            F.count("*").alias("rating_count"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
        )

        global_mean = (
            ratings.agg(F.avg("rating").alias("mu")).first()["mu"] or 0.0
        )

        # Bayesian shrinkage reduces the influence of movies with few ratings.
        gold_movies = (
            movies.join(stats, "movieId", "left")
            .fillna({"rating_count": 0, "avg_rating": 0.0})
            .withColumn("rating_count", F.col("rating_count").cast("long"))
            .withColumn("avg_rating", F.col("avg_rating").cast("double"))
            .withColumn(
                "weighted_score",
                F.when(
                    F.col("rating_count") > 0,
                    (
                        F.col("rating_count")
                        / (F.col("rating_count") + F.lit(10.0))
                    )
                    * F.col("avg_rating")
                    + (
                        F.lit(10.0)
                        / (F.col("rating_count") + F.lit(10.0))
                    )
                    * F.lit(float(global_mean)),
                ).otherwise(F.lit(float(global_mean))),
            )
            .orderBy(F.desc("weighted_score"), F.desc("rating_count"))
        )

        gold = OUTPUT_DIR / "gold"
        write_parquet_local(gold_movies, gold / "movie_analytics")
        write_parquet_local(
            ratings_with_movies,
            gold / "ratings_training",
        )

        # User-level analytics for dashboard KPIs.
        user_stats = ratings.groupBy("userId").agg(
            F.count("*").alias("ratings_made"),
            F.round(F.avg("rating"), 3).alias("mean_rating"),
        )

        write_parquet_local(user_stats, gold / "user_analytics")

        # Print summary information.
        print("\nPipeline completed successfully.")
        print(f"Movies: {movies.count():,}")
        print(f"Ratings: {ratings.count():,}")
        print(f"Users: {ratings.select('userId').distinct().count():,}")
        print(f"Output layers: {OUTPUT_DIR}")

    finally:
        spark.stop()
