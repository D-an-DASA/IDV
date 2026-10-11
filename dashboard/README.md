# CardioView Dash dashboard

The Dash version provides the population overview and individual risk prediction pages. It reads the existing BRFSS CSV and Logistic Regression artifact from `IDV/data` and `IDV/src/model`.

From the repository's `IDV` directory:

```bash
python -m pip install -r requirements.txt
python dashboard/dash_app.py
```

Then open `http://127.0.0.1:8050`. The previous Streamlit applications remain available as `dashboard/app.py` and `dashboard/dashboard.py`.
