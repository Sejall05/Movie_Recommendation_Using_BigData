$ErrorActionPreference = "Stop"
python run_pipeline.py --download
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python train_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
streamlit run dashboard.py
