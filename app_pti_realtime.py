import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="PTI Lab Real-Time Intelligence", page_icon="🧪", layout="wide")

# ==============================================================================
# 1. KONFIGURASI KONEKSI GOOGLE SHEETS & MODEL TERLATIH (DARI COLAB)
# ==============================================================================
DEFAULT_GSHEET_ID = "1BRbB4ucQJffZbDmAevURXK9zRn8sNe43"
MODEL_PATH = "pti_models_20.joblib"

@st.cache_resource
def load_trained_models():
    # Langsung memuat otak model yang dilatih di Google Colab (murni inferensi)
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=10)
def fetch_realtime_gsheet(sheet_id):
    base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    df_cat = pd.read_csv(base_url + "Master_Katalog")
    df_hist = pd.read_csv(base_url + "Clean_History_Daily")
    
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist["usage"] = pd.to_numeric(df_hist["usage"], errors="coerce").fillna(0.0)
    df_cat["current_stock"] = pd.to_numeric(df_cat["current_stock"], errors="coerce").fillna(0.0)
    return df_hist, df_cat

def build_features(df_sub, horizon=14):
    d = df_sub.sort_values("date").copy()
    d.set_index("date", inplace=True)
    dow, doy = d.index.dayofweek, d.index.dayofyear
    d["sin_dow"], d["cos_dow"] = np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)
    d["sin_doy"], d["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["day_of_month"] = d.index.day
    d["is_sunday"] = (dow == 6).astype(int)
    d["is_mcu"] = ((d.index.day >= 20) & (d.index.day <= 26)).astype(int)
    
    sh = d["usage"].shift(horizon)
    d[f"lag_{horizon}"] = sh
    d[f"lag_{horizon+7}"] = d["usage"].shift(horizon + 7)
    d[f"ewma_{horizon}"] = sh.ewm(span=7).mean()
    d[f"std_{horizon}"] = sh.rolling(14).std()
    return d

# ==============================================================================
# 2. SIDEBAR & SINKRONISASI DATA REAL-TIME
# ==============================================================================
st.sidebar.title("🧪 PTI Lab Real-Time")
gsheet_id = st.sidebar.text_input("ID Database Google Sheets:", value=DEFAULT_GSHEET_ID)

if st.sidebar.button("🔄 Sinkronkan Data Terbaru (Apps Script)", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("📦 Muat Ulang File Model (.joblib)", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

try:
    artifact = load_trained_models()
    models = artifact["models"]
    f_cols = artifact["feature_cols"]
    horizon = artifact["horizon"]
except FileNotFoundError:
    st.error("File `pti_models_20.joblib` tidak ditemukan di repositori GitHub!")
    st.stop()
except Exception as e:
    st.error(
        f"Gagal memuat `pti_models_20.joblib` karena perbedaan versi Python/Scikit-Learn antara Colab dan Streamlit Cloud.\n\n"
        f"**Cara Mengatasi:** Di Streamlit Cloud klik **Manage app → Settings → ubah Python version ke 3.11**, lalu reboot app.\n\n"
        f"Detail teknis: `{e}`"
    )
    st.stop()

try:
    df_hist, df_cat = fetch_realtime_gsheet(gsheet_id)
except Exception as e:
    st.warning("Gagal membaca Google Sheets. Pastikan aksesnya sudah diatur ke 'Anyone with the link' (Viewer).")
    st.stop()

z_label = st.sidebar.selectbox("Target Service Level:", ["95% (Z=1.65)", "97.5% (Z=1.96)", "99% (Z=2.33)"], index=1)
z_score = float(z_label.split("Z=")[1].replace(")", ""))

reagent_map = {f"{r['reagent_id']} - {r['reagent_name']}": r['reagent_id'] for _, r in df_cat.iterrows()}
selected_id = reagent_map[st.sidebar.selectbox("Pilih Reagen Monitoring:", list(reagent_map.keys()), index=1)]

# ==============================================================================
# 3. INFERENSI REAL-TIME MENGGUNAKAN MODEL TERLATIH + DATA TERBARU GSHEETS
# ==============================================================================
proc_list = []
forecasts = {}

for _, item in df_cat.iterrows():
    rid = item["reagent_id"]
    unit, dec = item["unit"], (2 if item["unit"] == "Liter" else 1)
    
    sub = df_hist[df_hist["reagent_id"] == rid][["date", "usage"]].copy()
    sub = sub.groupby("date", as_index=False)["usage"].sum()
    
    full_idx = pd.date_range(start=sub["date"].min(), end=sub["date"].max(), freq="D")
    sub = sub.set_index("date").reindex(full_idx, fill_value=0.0).reset_index().rename(columns={"index": "date"})
    
    fut_dates = pd.date_range(start=sub["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    combined = pd.concat([sub, pd.DataFrame({"date": fut_dates, "usage": np.nan})], ignore_index=True)
    X_fut = build_features(combined, horizon).loc[fut_dates, f_cols]
    
    q10 = np.round(np.maximum(0, models[rid][0.10].predict(X_fut)), dec)
    q50 = np.round(np.maximum(0, models[rid][0.50].predict(X_fut)), dec)
    q90 = np.round(np.maximum(0, models[rid][0.90].predict(X_fut)), dec)
    
    st_val = float(item["current_stock"])
    lt_val, lt_std = int(item["lead_time"]), float(item["lt_std"])
    pack, cost, shelf = float(item["pack_size"]), float(item["unit_cost"]), int(item["shelf_life"])
    
    ss = z_score * np.sqrt((lt_val * (np.std(q50) ** 2)) + ((np.mean(q50) ** 2) * (lt_std ** 2)))
    rop = np.sum(q50[:lt_val]) + ss
    cum_q50 = st_val - np.cumsum(q50)
    so_days = np.where(cum_q50 <= 0)[0]
    
    max_ord = max(0.0, (np.mean(q50) * shelf) - st_val)
    shortage = min(max(0.0, (np.sum(q50) + ss) - st_val), max_ord)
    packs = int(np.ceil(shortage / pack))
    
    if len(so_days) > 0: status = f"KRITIS (Habis H+{so_days[0]+1})"
    elif st_val <= rop: status = "WASPADA (Bawah ROP)"
    else:
        status = "AMAN"
        if st_val > rop * 1.25: packs = 0
        
    proc_list.append({
        "ID": rid, "Nama Reagen": item["reagent_name"], "Kategori": item["category"], "Satuan": unit,
        "Stok Gudang Live": st_val, "Prediksi 14H": round(np.sum(q50), dec),
        "ROP": round(rop, dec), "Status": status, "Order (Pack)": packs,
        "Total Qty": round(packs * pack, dec), "Biaya (Rp)": packs * cost
    })
    forecasts[rid] = pd.DataFrame({
        "date": fut_dates, "q10": q10, "q50": q50, "q90": q90,
        "stok_q50": np.round(cum_q50, dec), "stok_q90": np.round(st_val - np.cumsum(q90), dec)
    })

df_proc = pd.DataFrame(proc_list)

# ==============================================================================
# 4. TAMPILAN DASHBOARD MONITORING REAL-TIME
# ==============================================================================
st.title("🧪 PTI Lab — Real-Time Reagent Monitoring & Forecasting")
st.caption(f"Terhubung langsung ke Google Sheets | Tanggal Data Terakhir: **{df_hist['date'].max().strftime('%Y-%m-%d')}** | Model Pre-Trained Colab (`pti_models_20.joblib`)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚨 Status KRITIS", f"{df_proc['Status'].str.contains('KRITIS').sum()} Item")
c2.metric("⚠️ Status WASPADA", f"{df_proc['Status'].str.contains('WASPADA').sum()} Item")
c3.metric("📦 Total Order", f"{df_proc['Order (Pack)'].sum()} Pack")
c4.metric("💰 Estimasi Anggaran", f"Rp {df_proc['Biaya (Rp)'].sum():,.0f}")

tab1, tab2, tab3 = st.tabs(["📈 Monitoring & Prediksi Live", "📋 Keputusan Order (20 Reagen)", "🔬 Laporan Akurasi Model (Colab)"])

with tab1:
    sel_info = df_proc[df_proc["ID"] == selected_id].iloc[0]
    hist_s = df_hist[df_hist["reagent_id"] == selected_id].tail(45)
    fore_s = forecasts[selected_id]

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=hist_s["date"], y=hist_s["usage"], name="Aktual (Google Sheets)", line=dict(color="#1d4ed8", width=2.5)))
        fig1.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q90"], line=dict(width=0), showlegend=False))
        fig1.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q10"], name="Interval 80% (Q10-Q90)", fill="tonexty", fillcolor="rgba(245,158,11,0.25)", line=dict(width=0)))
        fig1.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q50"], name="Prediksi Median (Q50)", line=dict(color="#d97706", width=3, dash="dash")))
        fig1.update_layout(title=f"Prediksi Konsumsi: {sel_info['Nama Reagen']} ({sel_info['Satuan']}/hari)", height=360, margin=dict(t=40, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["stok_q50"], name="Sisa Stok Normal (Q50)", line=dict(color="#059669", width=3)))
        fig2.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["stok_q90"], name="Sisa Stok Lonjakan (Q90)", line=dict(color="#dc2626", dash="dot")))
        fig2.add_hline(y=sel_info["ROP"], line_dash="dash", line_color="orange", annotation_text=f"ROP ({sel_info['ROP']})")
        fig2.add_hline(y=0, line_color="red")
        fig2.update_layout(title=f"Simulasi Stok Live ({sel_info['Stok Gudang Live']} {sel_info['Satuan']})", height=360, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.dataframe(df_proc.style.format({"Biaya (Rp)": "Rp {:,.0f}"}), use_container_width=True)

with tab3:
    if "evaluation" in artifact:
        st.dataframe(artifact["evaluation"], use_container_width=True)
    else:
        st.info("Tabel evaluasi tidak disertakan di dalam artefak model ini.")
