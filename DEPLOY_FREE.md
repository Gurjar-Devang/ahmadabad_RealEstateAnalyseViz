# Free Deployment (Streamlit Community Cloud)

This project can be deployed for free using Streamlit Community Cloud.

## 1) Push project to GitHub
Make sure these files are in your repo root:
- `dashboard.py`
- `ahm_data.csv`
- `requirements.txt`

## 2) Create the app on Streamlit Cloud
1. Go to: https://share.streamlit.io/
2. Sign in with GitHub.
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to: `dashboard.py`
6. Click **Deploy**.

## 3) If deployment fails
- Confirm `ahm_data.csv` is committed in GitHub.
- Confirm `requirements.txt` exists in repo root.
- Reboot the app from Streamlit Cloud settings.

## Local run command
```bash
streamlit run dashboard.py
```
