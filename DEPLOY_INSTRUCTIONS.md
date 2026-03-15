# Deployment Instructions (Streamlit Cloud)

You can host this app for free on Streamlit Community Cloud.

## 1. Prepare Files
Ensure the following files are in your folder:
- `app_boatrace.py` (The App)
- `requirements.txt` (Dependencies)
- `lgb_ranker.txt` (The Trained Model)
- `app_data/` folder (containing the 4 CSV files exported by `export_app_data.py`)

## 2. Push to GitHub
1. Create a new repository on GitHub (e.g., `boatrace-predictor`).
2. Upload all the above files to the repository.
   - Note: If `app_data` files are large (>100MB), you might need Git LFS, but they should be small enough (<10MB usually, except maybe params).

## 3. Deploy
1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
2. Sign in with GitHub.
3. Click "New App".
4. Select your repository (`boatrace-predictor`).
5. Select Branch: `main`.
6. Main file path: `app_boatrace.py`.
7. Click **Deploy**.

## 4. Usage
- The app will spin up.
- Select a valid Date (today or future) and Venue.
- **Note**: The scraping logic works for `boatrace.jp`. If standard tables aren't found (e.g., race canceled or page layout change), it will show an error.
