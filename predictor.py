# predictor.py
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

from preprocessing import generate_all_features, prepare_features_for_model

class DemandPredictor:
    def __init__(self, model_path="artifacts/demand_forecast.pkl",
                 cols_path="artifacts/model_cols.pkl",
                 gm_path="artifacts/global_means.pkl"):
        self.model = joblib.load(model_path)
        self.model_cols = joblib.load(cols_path)
        self.global_means = joblib.load(gm_path)

    def _ensure_history(self, history_df):
        """
        history_df must have columns: ['date'(datetime), 'store', 'item', 'sales']
        sales must be in log1p scale when fed to generate_all_features,
        because the training used log1p(sales).
        """
        df = history_df.copy()
        # convert date
        df['date'] = pd.to_datetime(df['date'])
        return df

    def multi_step_forecast(self, history_df, store, item, steps=30):
        """
        history_df: dataframe containing past data for that (store,item) and optionally other pairs.
                    Must contain at least some prior rows; missing long lags will be filled by global_means.
                    sales column should be raw sales (we'll convert to log1p internally).
        store, item: ints
        steps: int days to forecast
        Returns: list of dicts [{'date':..., 'pred':... (raw scale)}, ...]
        """
        # filter for this store-item and sort by date
        df_hist = history_df.copy()
        df_hist = df_hist[(df_hist.store == store) & (df_hist.item == item)].sort_values('date').reset_index(drop=True)
        if df_hist.empty:
            raise ValueError("No history for this store/item. Add at least one historical row (date,store,item,sales).")

        # convert user's raw sales to log1p (model trained on log1p)
        df_hist = df_hist.copy()
        df_hist['sales'] = np.log1p(df_hist['sales'].astype(float).values)

        results = []
        df_work = df_hist.copy()

        for i in range(steps):
            # generate features for entire df_work
            df_features = generate_all_features(df_work)
            # prepare X (will fill missing features with global means)
            X = prepare_features_for_model(df_features, self.model_cols, self.global_means)

            # take last row as input
            X_last = X.iloc[[-1]]
            # model.predict returns log1p(sales)
            pred_log = self.model.predict(X_last)[0]
            pred_raw = float(np.expm1(pred_log))

            next_date = df_work['date'].iloc[-1] + pd.Timedelta(days=1)

            results.append({'date': next_date, 'pred_log': pred_log, 'pred': pred_raw})

            # append the predicted row into df_work in log space for next step
            df_work = pd.concat([
                df_work,
                pd.DataFrame({
                    'date': [next_date],
                    'store': [store],
                    'item': [item],
                    'sales': [pred_log]   # store in log domain to match pipeline
                })
            ], ignore_index=True)

        return results
