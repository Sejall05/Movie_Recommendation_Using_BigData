
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_local(spark_df, output_path):
    output_path = Path(output_path)

    # Convert the Spark DataFrame to a local Pandas DataFrame.
    pandas_df = spark_df.toPandas()

    # Convert Pandas data to an Apache Arrow table.
    arrow_table = pa.Table.from_pandas(
        pandas_df,
        preserve_index=False,
    )

    # Replace existing output to support pipeline reruns.
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    # Write a single Parquet file inside the output directory.
    pq.write_table(
        arrow_table,
        output_path / "part-00000.parquet",
        compression="snappy",
    )

    print(f"Parquet written: {output_path}")


def write_csv_local(spark_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pandas_df = spark_df.toPandas()
    pandas_df.to_csv(output_path, index=False)

    print(f"CSV written: {output_path}")
