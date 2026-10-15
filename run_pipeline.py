
import argparse
import io
import urllib.request
import zipfile
import ssl
import certifi
from pathlib import Path

from src.common import DATA_DIR
from src.pipeline import build_layers


URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Skip downloading if the required dataset files already exist
    if (DATA_DIR / "movies.csv").exists() and (DATA_DIR / "ratings.csv").exists():
        print(f"Dataset already present at {DATA_DIR}")
        return

    print("Downloading MovieLens ml-latest-small from GroupLens...")

    try:
        # Use certifi's trusted CA certificates for SSL verification
        ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

        request = urllib.request.Request(
            URL,
            headers={"User-Agent": "MovieRecommendationBigData/1.0"}
        )

        with urllib.request.urlopen(
            request,
            timeout=60,
            context=ssl_context
        ) as response:
            archive_bytes = response.read()

        print("Download completed successfully.")
        
    except Exception as error:
        print("\nDataset download failed.")
        print(f"Error: {error}")
        print("\nPlease check your internet connection and SSL certificates.")
        print("If the problem continues, run the SSL diagnostic command.")
        raise SystemExit(1)

    # Extract the required MovieLens files
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            prefix = "ml-latest-small/"

            required_files = {
                "movies.csv",
                "ratings.csv",
                "tags.csv",
                "links.csv"
            }

            extracted_files = set()

            for name in zf.namelist():
                if (
                    name.startswith(prefix)
                    and not name.endswith("/")
                ):
                    filename = Path(name).name

                    if filename in required_files:
                        target = DATA_DIR / filename
                        target.write_bytes(zf.read(name))
                        extracted_files.add(filename)
                        print(f"Extracted: {filename}")

            # Confirm that the essential files were extracted
            essential_files = {"movies.csv", "ratings.csv"}

            if not essential_files.issubset(extracted_files):
                raise RuntimeError(
                    "The downloaded archive does not contain "
                    "the required movies.csv and ratings.csv files."
                )

        print(f"\nDataset extracted successfully to: {DATA_DIR}")

    except (zipfile.BadZipFile, RuntimeError) as error:
        print(f"\nDataset extraction failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Bronze, Silver and Gold data layers."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MovieLens dataset if missing."
    )

    args = parser.parse_args()

    if args.download:
        download_dataset()

    # Build the Bronze, Silver and Gold data layers
    build_layers()
