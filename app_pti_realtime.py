import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# KONFIGURASI HALAMAN WEB
# ==============================================================================
st.set_page_config(
    page_title="PTI Lab Intelligence - Monitoring & Prediksi Reagen",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_GSHEET_ID = "1BRbB4ucQJffZbDmAevURXK9zRn8sNe43"
MODEL_PATH = "pti_models_20.joblib"

# ==============================================================================
# FUNGSI PEMUATAN MODEL TERLATIH (COLAB) & DATA LIVE (GOOGLE SHEETS)
# ==============================================================================
@st.cache_resource
def load_trained_artifact():
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
# SIDEBAR: KONTROL SIMULASI & SUMBER DATABASE REAL-TIME
# ==============================================================================
st.sidebar.title("🧪 Kontrol Sistem PTI")
st.sidebar.markdown("Sistem *Predictive Time Intelligence* untuk Manajemen Inventaris Stokastik Reagen.")

# 1. Integrasi Google Sheets Live (Satu Sumber Data dengan Apps Script)
st.sidebar.subheader("1. Sinkronisasi Database Live")
gsheet_id = st.sidebar.text_input("ID Google Sheets:", value=DEFAULT_GSHEET_ID)

col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("🔄 Sync Data", type="primary", use_container_width=True, help="Tarik transaksi & stok terbaru dari Google Sheets"):
    st.cache_data.clear()
    st.rerun()

if col_s2.button("📦 Reload Model", use_container_width=True, help="Muat ulang file pti_models_20.joblib terbaru dari GitHub"):
    st.cache_resource.clear()
    st.rerun()

try:
    artifact = load_trained_artifact()
except FileNotFoundError:
    st.error("⚠️ File `pti_models_20.joblib` tidak ditemukan! Unggah file hasil pelatihan Google Colab ke repositori GitHub Anda.")
    st.stop()
except Exception as e:
    st.error(
        f"⚠️ Gagal memuat `pti_models_20.joblib` karena perbedaan versi Python/Scikit-Learn antara Colab dan Streamlit Cloud.\n\n"
        f"**Solusi:** Klik **Manage app → Settings → ubah Python version ke 3.11**, lalu Reboot app.\n\n"
        f"Detail: `{e}`"
    )
    st.stop()

try:
    df_hist, df_cat = fetch_realtime_gsheet(gsheet_id)
except Exception as e:
    st.error(f"⚠️ Gagal terhubung ke Google Sheets. Pastikan aksesnya 'Anyone with the link' (Viewer). Detail: {e}")
    st.stop()

models = artifact["models"]
f_cols = artifact["feature_cols"]
importances = artifact["importances"]
df_eval = artifact["evaluation"]
horizon = artifact["horizon"]

# 2. Parameter Risiko Stokastik (Service Level)
st.sidebar.subheader("2. Parameter Kebijakan Inventaris")
service_level_label = st.sidebar.selectbox(
    "Target Service Level (Alpha):",
    options=[
        "95.0% (Z = 1.65 - Standar)",
        "97.5% (Z = 1.96 - Ketat)",
        "99.0% (Z = 2.33 - Kritis/Klinis)"
    ],
    index=1
)
z_score = float(service_level_label.split("Z = ")[1].split(" ")[0])

# 3. Simulasi What-If pada Reagen Tertentu
st.sidebar.subheader("3. Simulasi What-If (Reagen Terpilih)")
reagent_options = {f"{r['reagent_id']} - {r['reagent_name']}": r['reagent_id'] for _, r in df_cat.iterrows()}
selected_label = st.sidebar.selectbox("Pilih Reagen untuk Analisis Detail:", list(reagent_options.keys()), index=1)
selected_id = reagent_options[selected_label]

sel_row = df_cat[df_cat["reagent_id"] == selected_id].iloc[0]
max_st = 5000.0 if sel_row["unit"] == "Gram" else 50.0
step_st = 10.0 if sel_row["unit"] == "Gram" else 0.5

sim_stock = st.sidebar.number_input(
    f"Simulasi Stok Fisik ({sel_row['unit']}):",
    min_value=0.0, max_value=max_st, value=float(sel_row["current_stock"]), step=step_st
)
sim_lt = st.sidebar.slider(
    "Simulasi Lead Time Supplier (Hari):",
    min_value=1, max_value=21, value=int(sel_row["lead_time"])
)

# ==============================================================================
# INFERENSI REAL-TIME & KALKULASI OPTIMASI PRESKRIPTIF (20 REAGEN)
# ==============================================================================
proc_records = []
daily_forecasts = {}

for _, item in df_cat.iterrows():
    rid = item["reagent_id"]
    unit = item["unit"]
    dec = 2 if unit == "Liter" else 1

    # Susun fitur lag dari histori terbaru di Google Sheets (murni inferensi)
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

    daily_forecasts[rid] = pd.DataFrame({
        "date": fut_dates,
        "q10": q10,
        "q50": q50,
        "q90": q90
    })

    st_val = sim_stock if rid == selected_id else float(item["current_stock"])
    lt_val = sim_lt if rid == selected_id else int(item["lead_time"])
    lt_std, pack, cost, shelf = float(item["lt_std"]), float(item["pack_size"]), float(item["unit_cost"]), int(item["shelf_life"])

    # Rumus Stokastik Safety Stock & Reorder Point
    ss = z_score * np.sqrt((lt_val * (np.std(q50) ** 2)) + ((np.mean(q50) ** 2) * (lt_std ** 2)))
    rop = np.sum(q50[:lt_val]) + ss
    cum_q50 = st_val - np.cumsum(q50)
    so_days = np.where(cum_q50 <= 0)[0]

    # Kendala Umur Simpan (Shelf-Life)
    max_order = max(0.0, (np.mean(q50) * shelf) - st_val)
    shortage = min(max(0.0, (np.sum(q50) + ss) - st_val), max_order)
    packs = int(np.ceil(shortage / pack))

    if len(so_days) > 0:
        status = f"KRITIS (Habis H+{so_days[0]+1})"
    elif st_val <= rop:
        status = "WASPADA (Bawah ROP)"
    else:
        status = "AMAN"
        if st_val > rop * 1.25:
            packs = 0

    proc_records.append({
        "ID": rid,
        "Nama Reagen": item["reagent_name"],
        "Kategori": item["category"],
        "Satuan": unit,
        "Stok Fisik": st_val,
        "Prediksi 14H": round(np.sum(q50), dec),
        "Safety Stock": round(ss, dec),
        "ROP": round(rop, dec),
        "Status": status,
        "Order (Pack)": packs,
        "Total Qty Order": round(packs * pack, dec),
        "Estimasi Biaya (Rp)": packs * cost
    })

df_proc = pd.DataFrame(proc_records)

# ==============================================================================
# TAMPILAN UTAMA DASHBOARD (KPI & TABS)
# ==============================================================================
st.title("📊 Predictive Time Intelligence (PTI) — Monitoring Reagen Laboratorium")
st.markdown(
    f"**Arsitektur Model:** *Direct Multi-Horizon Quantile Gradient Boosting ($Q_{{10}}, Q_{{50}}, Q_{{90}}$)* | "
    f"**Data Live s/d:** `{df_hist['date'].max().strftime('%Y-%m-%d')}` | "
    f"**Horizon:** `{horizon} Hari` | **Service Level:** `{service_level_label.split(' ')[0]}`"
)

# Baris KPI Utama
kritis_count = df_proc["Status"].str.contains("KRITIS").sum()
waspada_count = df_proc["Status"].str.contains("WASPADA").sum()
total_packs = df_proc["Order (Pack)"].sum()
total_budget = df_proc["Estimasi Biaya (Rp)"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🚨 Reagen Status KRITIS", f"{kritis_count} Item", delta="Segera Order" if kritis_count > 0 else "Aman", delta_color="inverse")
col2.metric("⚠️ Reagen Status WASPADA", f"{waspada_count} Item", delta="Mendekati ROP", delta_color="off")
col3.metric("📦 Total Kemasan Dipesan", f"{total_packs} Pack", delta=f"Untuk {horizon} hari ke depan")
col4.metric("💰 Total Estimasi Anggaran", f"Rp {total_budget:,.0f}", delta=f"Z-Score = {z_score}")

st.divider()

# Navigasi Tab
tab1, tab2, tab3 = st.tabs([
    "📈 Analisis Visual & Simulasi Reagen",
    "📋 Keputusan Pengadaan & Ekspor PO",
    "🔬 Validasi Model (Cross-Validation & MASE)"
])

# ------------------------------------------------------------------------------
# TAB 1: ANALISIS VISUAL & SIMULASI REAGEN
# ------------------------------------------------------------------------------
with tab1:
    sel_info = df_proc[df_proc["ID"] == selected_id].iloc[0]
    hist_s = df_hist[df_hist["reagent_id"] == selected_id].tail(45)
    fore_s = daily_forecasts[selected_id]
    imp_s = importances[selected_id]

    dec_sel = 2 if sel_info["Satuan"] == "Liter" else 1
    sim_stok_q50 = np.round(sel_info["Stok Fisik"] - np.cumsum(fore_s["q50"]), dec_sel)
    sim_stok_q90 = np.round(sel_info["Stok Fisik"] - np.cumsum(fore_s["q90"]), dec_sel)

    # Banner Status Reagen Terpilih
    if "KRITIS" in sel_info["Status"]:
        st.error(f"**STATUS {sel_info['ID']} ({sel_info['Nama Reagen']}): {sel_info['Status']}** — Rekomendasi beli segera: **{sel_info['Order (Pack)']} Pack ({sel_info['Total Qty Order']} {sel_info['Satuan']})**.")
    elif "WASPADA" in sel_info["Status"]:
        st.warning(f"**STATUS {sel_info['ID']} ({sel_info['Nama Reagen']}): {sel_info['Status']}** — Rekomendasi beli: **{sel_info['Order (Pack)']} Pack ({sel_info['Total Qty Order']} {sel_info['Satuan']})**.")
    else:
        st.success(f"**STATUS {sel_info['ID']} ({sel_info['Nama Reagen']}): {sel_info['Status']}** — Stok mencukupi kebutuhan {horizon} hari ke depan.")

    # Grafik 1: Probabilistic Forecast Fan Chart
    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=hist_s["date"], y=hist_s["usage"], name="Pemakaian Aktual (45 Hari Terakhir)", line=dict(color="#1d4ed8", width=2.5)))
    fig_fore.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q90"], line=dict(width=0), showlegend=False))
    fig_fore.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q10"], name="Interval Prediksi 80% (Q10 - Q90)", fill="tonexty", fillcolor="rgba(245, 158, 11, 0.25)", line=dict(width=0)))
    fig_fore.add_trace(go.Scatter(x=fore_s["date"], y=fore_s["q50"], name="Prediksi Median (Q50)", line=dict(color="#d97706", width=3, dash="dash")))
    fig_fore.update_layout(title=f"Prediksi Konsumsi Harian & Ketidakpastian: {sel_info['Nama Reagen']} ({sel_info['Satuan']}/hari)", height=380, margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_fore, use_container_width=True)

    # Grafik 2 & 3 Berdampingan
    c_left, c_right = st.columns(2)
    with c_left:
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(x=fore_s["date"], y=sim_stok_q50, name="Sisa Stok Skenario Normal (Q50)", line=dict(color="#059669", width=3)))
        fig_dep.add_trace(go.Scatter(x=fore_s["date"], y=sim_stok_q90, name="Sisa Stok Skenario Lonjakan (Q90)", line=dict(color="#dc2626", width=2.5, dash="dot")))
        fig_dep.add_hline(y=sel_info["ROP"], line_dash="dash", line_color="orange", annotation_text=f"ROP ({sel_info['ROP']})")
        fig_dep.add_hline(y=0, line_color="red", annotation_text="Batas Habis (0)")
        fig_dep.update_layout(title="Simulasi Deplesi Stok Fisik (Normal vs Lonjakan)", height=360, margin=dict(t=45, b=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_dep, use_container_width=True)

    with c_right:
        fig_imp = go.Figure(go.Bar(x=imp_s.values, y=imp_s.index, orientation="h", marker_color="#0ea5e9"))
        fig_imp.update_layout(title="Interpretabilitas: Permutation Feature Importance", xaxis_title="Skor Kontribusi Fitur", height=360, margin=dict(t=45, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: KEPUTUSAN PENGADAAN & EKSPOR PO
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Daftar Rekomendasi Pengadaan & Anggaran (20 Reagen)")
    
    kat_filter = st.multiselect("Filter Berdasarkan Kategori Kimia:", options=df_proc["Kategori"].unique(), default=df_proc["Kategori"].unique())
    df_filtered = df_proc[df_proc["Kategori"].isin(kat_filter)]

    def highlight_status(val):
        if "KRITIS" in str(val):
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
        elif "WASPADA" in str(val):
            return "background-color: #fef3c7; color: #92400e; font-weight: bold"
        return "background-color: #dcfce7; color: #166534"

    st.dataframe(
        df_filtered.style.map(highlight_status, subset=["Status"]).format({"Estimasi Biaya (Rp)": "Rp {:,.0f}"}),
        use_container_width=True,
        height=520
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        cat_sum = df_filtered.groupby("Kategori")["Estimasi Biaya (Rp)"].sum().sort_values(ascending=True)
        fig_cat = go.Figure(go.Bar(x=cat_sum.values, y=cat_sum.index, orientation="h", marker_color="#4f46e5"))
        fig_cat.update_layout(title="Alokasi Anggaran Belanja per Kategori Reagen (Rp)", height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b2:
        st.markdown("### 📥 Ekspor Dokumen Pengadaan")
        st.write("Unduh tabel rekomendasi order beserta hasil evaluasi model ke dalam format Excel siap cetak untuk bagian pengadaan.")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_filtered.to_excel(writer, sheet_name="Purchase_Requisition", index=False)
            df_eval.to_excel(writer, sheet_name="Evaluasi_Model", index=False)
        st.download_button(
            label="📄 Download Purchase Requisition (.xlsx)",
            data=buffer.getvalue(),
            file_name="Purchase_Requisition_PTI_Lab.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

# ------------------------------------------------------------------------------
# TAB 3: VALIDASI MODEL (CROSS-VALIDATION & MASE)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Audit Akademik: Evaluasi Expanding Window Time-Series Cross-Validation")
    st.markdown("""
- **CV_MAE & CV_RMSE:** Kesalahan rata-rata absolut dan akar kuadrat rata-rata pada lipatan uji kronologis (dalam satuan asli reagen).
- **CV_MASE (*Mean Absolute Scaled Error*):** Metrik standar bebas skala (*scale-independent*). Nilai **MASE < 1.0** membuktikan model *Quantile Gradient Boosting* mengungguli metode *Naïve Forecast*.
    """)
    st.dataframe(df_eval, use_container_width=True, height=550)
