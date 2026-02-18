# Diamond Price & Market Segment Predictor

Streamlit app that predicts a diamond's price and its market segment (cluster) using pre-trained model artifacts.

## Summary

This repository contains a Streamlit app (`f_app.py`) that loads pre-trained ML artifacts (Random Forest regressor and KMeans + scaler) to predict a diamond's price and market segment from user inputs such as carat, dimensions, cut, color and clarity.

## Expected model artifacts

Place the following files in the project root (same folder as `f_app.py`):

- `RandomForestmodel.pkl` — Random Forest regressor used for price prediction
- `kmeans_best_model.pkl` — KMeans clustering model used for market segment prediction
- `scaler.pkl` — `sklearn` scaler used to scale features before clustering
- `diamonds.csv` — (optional) training/inspection dataset referenced by notebooks

Note: The app caption also references `ann_preprocessor.pkl` and `ann_state.pth` (for an ANN version). Those are optional and not required by the current `f_app.py` implementation unless you modify the app to use the ANN.

## Quickstart (Windows)

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run f_app.py
```

4. The app opens in your browser; enter diamond attributes and click the `Predict Price` or `cluster` buttons to see results.

## Files of interest

- `f_app.py` — main Streamlit application. See UI inputs and model loading there.
- `clustering.ipynb`, `prediction.ipynb` — notebooks included for EDA and model work.

## Notes

- The app uses an INR conversion constant (`INR_RATE = 82.0`) in `f_app.py`. Adjust as needed.
- Ensure the exact model filenames listed above match the files you place in the project root.
- If you get errors loading PyTorch/torch, ensure `torch` is installed for your platform (CPU vs GPU build).

## Contact

If you want, I can pin specific package versions in `requirements.txt`, add a sample models download script, or update `f_app.py` to handle missing artifacts more gracefully.
