
from pathlib import Path
import os
import sys

from pyspark.sql import SparkSession


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.getenv("MOVIELENS_DIR", str(ROOT / "data" / "ml-latest-small"))
)

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", str(ROOT / "outputs"))
)


def make_spark(app_name="MovieRecommendationBigData"):
    # Force Spark to use this project's virtual-environment Python.
    python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config(
            "spark.sql.warehouse.dir",
            (ROOT / "spark-warehouse").as_uri(),
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def ensure_dirs():
    for name in ("bronze", "silver", "gold", "model"):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
