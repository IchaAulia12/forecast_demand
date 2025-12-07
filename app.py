# app.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from predictor import DemandPredictor
import matplotlib.pyplot as plt

# paths
HISTORY_PATH = "user_history.csv"  # lokal: simpan semua data user di sini (simple)
MODEL_PATH = "artifacts/demand_forecast.pkl"
COLS_PATH = "artifacts/model_cols.pkl"
GM_PATH = "artifacts/global_means.pkl"

st.set_page_config(page_title="Demand Forecast (Streamlit)", layout="wide")

st.title("📦 Demand Forecast — Streamlit Demo")
st.write("Input harian -> simpan -> setelah ada beberapa data, lakukan multi-step forecast menggunakan model LightGBM + global-fill.")

# init predictor
predictor = DemandPredictor(model_path=MODEL_PATH, cols_path=COLS_PATH, gm_path=GM_PATH)

# Sidebar: Upload or add data
with st.sidebar:
    st.header("Input / Upload Data")
    mode = st.radio("Pilih mode", ("Add single row", "Upload CSV (history)"))

    if mode == "Add single row":
        date = st.date_input("Date", value=datetime.today())
        store = st.number_input("Store ID", min_value=1, value=1, step=1)
        item = st.number_input("Item ID", min_value=1, value=1, step=1)
        sales = st.number_input("Sales (units)", min_value=0.0, value=0.0, step=1.0)

        if st.button("Simpan ke history"):
            # append to local CSV
            row = pd.DataFrame([{
                "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "store": int(store),
                "item": int(item),
                "sales": float(sales)
            }])
            if os.path.exists(HISTORY_PATH):
                row.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
            else:
                row.to_csv(HISTORY_PATH, index=False)
            st.success("Data tersimpan di history.")
    else:
        uploaded = st.file_uploader("Upload CSV (columns: date,store,item,sales)", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded, parse_dates=['date'])
            # save to history path (overwrite)
            df_up.to_csv(HISTORY_PATH, index=False)
            st.success(f"History tersimpan ke {HISTORY_PATH} (overwrite).")

# Main area: show history and forecasting UI
st.subheader("History (sample)")
if os.path.exists(HISTORY_PATH):
    df_hist = pd.read_csv(HISTORY_PATH, parse_dates=['date'])
    st.dataframe(df_hist.tail(20))
else:
    st.info("Belum ada history. Kamu bisa menambah data di sidebar (Add single row) atau upload CSV.")

st.subheader("Forecast")
col1, col2 = st.columns([1,1])

with col1:
    sel_store = st.number_input("Select Store ID for forecast", min_value=1, value=1, step=1)
    sel_item = st.number_input("Select Item ID for forecast", min_value=1, value=1, step=1)
    days = st.number_input("Forecast days (multi-step)", min_value=1, max_value=180, value=30, step=1)

with col2:
    if st.button("Run Forecast"):
        if not os.path.exists(HISTORY_PATH):
            st.error("History kosong. Tambahkan atau upload data sebelum prediksi.")
        else:
            hist = pd.read_csv(HISTORY_PATH, parse_dates=['date'])
            hist_pair = hist[(hist.store==sel_store) & (hist.item==sel_item)].sort_values('date').reset_index(drop=True)
            if hist_pair.shape[0] == 0:
                st.error("Tidak ada history untuk store/item tersebut. Tambahkan minimal 1 baris untuk pair ini.")
            else:
                with st.spinner("Mengenerate forecast..."):
                    preds = predictor.multi_step_forecast(hist, sel_store, sel_item, steps=int(days))
                    df_preds = pd.DataFrame(preds)
                    df_preds['date'] = pd.to_datetime(df_preds['date'])
                    df_preds_display = df_preds[['date','pred']].rename(columns={'pred':'forecast_sales'})

                    # plot: history + forecast
                    fig, ax = plt.subplots(figsize=(10,5))
                    # history raw scale
                    hist_plot = hist_pair.copy()
                    hist_plot['sales_raw'] = hist_plot['sales']
                    ax.plot(hist_plot['date'], hist_plot['sales_raw'], label='history', marker='o')
                    ax.plot(df_preds_display['date'], df_preds_display['forecast_sales'], label='forecast', marker='o')
                    ax.set_xlabel("date")
                    ax.set_ylabel("sales")
                    ax.legend()
                    st.pyplot(fig)

                    st.subheader("Forecast (table)")
                    st.dataframe(df_preds_display)

                    # allow download
                    csv = df_preds_display.to_csv(index=False)
                    st.download_button("Download forecast CSV", csv, file_name=f"forecast_store{sel_store}_item{sel_item}.csv")
