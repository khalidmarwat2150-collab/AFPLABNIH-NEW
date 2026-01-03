import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="AFPLAB CSV Processor", layout="wide")
st.title("AFPLAB CSV Processing App")

# --------------------------------------------------
# 1. File uploader
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload AFPLAB CSV files (multiple allowed)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:

    # --------------------------------------------------
    # 2. Load & combine CSV files
    # --------------------------------------------------
    dfs = [pd.read_csv(io.BytesIO(f.read()), low_memory=False) for f in uploaded_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # --------------------------------------------------
    # 3. Required columns check
    # --------------------------------------------------
    required_cols = [
        "IDCODE","P11","P12","P21","P22","P31","P32","ENTERO1","ENTERO2",
        "PROVINCE","SPECFROM","STCOND1"
    ]
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # --------------------------------------------------
    # 4. Fix numeric data types
    # --------------------------------------------------
    num_cols = ["P11","P12","P21","P22","P31","P32","ENTERO1","ENTERO2"]
    combined_df[num_cols] = combined_df[num_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    # --------------------------------------------------
    # 5. IDCODE2 & Year
    # --------------------------------------------------
    combined_df["IDCODE2"] = combined_df["IDCODE"].astype(str).str.split("-", n=1).str[0]

    def extract_year(idcode):
        if pd.isna(idcode):
            return ""
        parts = str(idcode).split("/")
        return "20" + parts[2] if len(parts) >= 3 else ""
    combined_df["Year"] = combined_df["IDCODE"].apply(extract_year)

    # --------------------------------------------------
    # 6. Build Type columns
    # --------------------------------------------------
    def build_type(df, col1, col2, mapping):
        def _f(row):
            v1, v2 = row[col1], row[col2]
            if pd.notna(v1) and pd.notna(v2) and v1 == v2:
                return mapping.get(int(v1), "")
            vals = [mapping.get(int(v1), "") if pd.notna(v1) else "", mapping.get(int(v2), "") if pd.notna(v2) else ""]
            return " + ".join(dict.fromkeys(filter(None, vals)))
        return df.apply(_f, axis=1)

    mapping_type1 = {1:"WPV1",2:"SL1",3:"WPV1+SL1",4:"VDPV1",5:"DISCORDANT",
                     6:"ITD Pending",7:"Negative",8:"Under Process",9:"Not received in Lab",
                     11:"aVDPV1",12:"iVDPV1",13:"cVDPV1"}
    mapping_type2 = {1:"WPV2",2:"SL2",3:"WPV2+SL2",4:"VDPV2",5:"DISCORDANT",
                     6:"ITD Pending",7:"Negative",8:"Under Process",9:"Not received in Lab",
                     11:"aVDPV2",12:"iVDPV2",13:"cVDPV2"}
    mapping_type3 = {1:"WPV3",2:"SL3",3:"WPV3+SL3",4:"VDPV3",5:"DISCORDANT",
                     6:"ITD Pending",7:"Negative",8:"Under Process",9:"Not received in Lab",
                     11:"aVDPV3",12:"iVDPV3",13:"cVDPV3"}

    combined_df["type1"] = build_type(combined_df, "P11","P12", mapping_type1)
    combined_df["type2"] = build_type(combined_df, "P21","P22", mapping_type2)
    combined_df["type3"] = build_type(combined_df, "P31","P32", mapping_type3)

    # --------------------------------------------------
    # 7. ENTERO column
    # --------------------------------------------------
    entero_map = {1:"NPEV",7:"NVI",8:"Under Process"}
    def build_entero(row):
        e1, e2 = row["ENTERO1"], row["ENTERO2"]
        if pd.notna(e1) and pd.notna(e2) and e1 == e2:
            return entero_map.get(int(e1), "")
        if (pd.notna(e1) and e1==1) or (pd.notna(e2) and e2==1):
            return "NPEV"
        vals = [entero_map.get(int(e1), "") if pd.notna(e1) else "", entero_map.get(int(e2), "") if pd.notna(e2) else ""]
        return " + ".join(dict.fromkeys(filter(None, vals)))
    combined_df["ENTERO"] = combined_df.apply(build_entero, axis=1)

    # --------------------------------------------------
    # 8. RESULT column
    # --------------------------------------------------
    def contains(text, keyword):
        return keyword.lower() in str(text).lower()

    def build_result(row):
        t1, t2, t3 = row["type1"], row["type2"], row["type3"]
        entero = row["ENTERO"]
        if any(contains(x,"Not received in Lab") for x in [t1,t2,t3,entero]):
            return "Not received in Lab"
        parts=[]
        if contains(t1,"SL1"): parts.append("SL1")
        if contains(t1,"WPV1"): parts.append("WPV1")
        if contains(t2,"SL2"): parts.append("SL2")
        if contains(t2,"WPV2"): parts.append("WPV2")
        if contains(t3,"SL3"): parts.append("SL3")
        if contains(t3,"WPV3"): parts.append("WPV3")
        if contains(entero,"NPEV"): parts.append("NPEV")
        for txt,n in [(t1,1),(t2,2),(t3,3)]:
            if contains(txt,"iVDPV"): parts.append(f"iVDPV{n}")
            if contains(txt,"cVDPV"): parts.append(f"cVDPV{n}")
            if contains(txt,"aVDPV"): parts.append(f"aVDPV{n}")
            if contains(txt,"VDPV") and not any(contains(txt,x) for x in ["iVDPV","cVDPV","aVDPV"]):
                parts.append(f"VDPV{n}")
        parts = list(dict.fromkeys(parts))
        if not parts:
            if contains(entero,"NVI"): return "NVI"
            if contains(entero,"NPEV"): return "NPEV"
            return "Under Process"
        return " + ".join(parts)

    combined_df["RESULT"] = combined_df.apply(build_result, axis=1)

    # --------------------------------------------------
    # 9. R_ columns & R_Result2
    # --------------------------------------------------
    combined_df["R_Negative"] = combined_df["RESULT"].str.contains("NVI", case=False, na=False).astype(int)
    combined_df["R_NPEV"] = combined_df["RESULT"].str.contains("NPEV", case=False, na=False).astype(int)
    combined_df["R_WPV"] = combined_df["RESULT"].str.contains("WPV", case=False, na=False).astype(int)
    combined_df["R_SL"] = combined_df["RESULT"].str.contains("SL", case=False, na=False).astype(int)
    combined_df["R_VDPV1"] = (combined_df["RESULT"]=="VDPV1").astype(int)
    combined_df["R_cVDPV2"] = (combined_df["RESULT"]=="cVDPV2").astype(int)
    combined_df["R_aVDPV1"] = (combined_df["RESULT"]=="aVDPV1").astype(int)
    combined_df["R_iVDPV1"] = (combined_df["RESULT"]=="iVDPV1").astype(int)
    combined_df["R_iVDPV3"] = (combined_df["RESULT"]=="iVDPV3").astype(int)
    combined_df["R_aVDPV2"] = (combined_df["RESULT"].str.contains("aVDPV2", case=False, na=False)).astype(int)

    def build_result2(row):
        if row["R_WPV"]==1: return "WPV1"
        elif row["R_VDPV1"]==1: return "VDPV1"
        elif row["R_cVDPV2"]==1: return "cVDPV2"
        elif row["R_iVDPV1"]==1: return "iVDPV1"
        elif row["R_aVDPV1"]==1: return "aVDPV1"
        elif row["R_iVDPV3"]==1: return "iVDPV3"
        elif row["R_aVDPV2"]==1: return "aVDPV2"
        elif row["R_SL"]==1: return "SL"
        elif row["R_NPEV"]==1: return "NPEV"
        elif row["R_Negative"]==1: return "NVI"
        elif row["RESULT"]=="Not received in Lab": return "Not received in Lab"
        elif row["RESULT"]=="Under Process": return "Under Process"
        else: return None

    combined_df["R_Result2"] = combined_df.apply(build_result2, axis=1)

    # --------------------------------------------------
    # 10. Create Linelist & WPV1count columns
    # --------------------------------------------------
    combined_df["WPV1count"] = combined_df["RESULT"].str.contains("WPV1", na=False).astype(int)
    combined_df["Linelist"] = 0
    idx = combined_df[combined_df["WPV1count"] == 1].groupby("IDCODE2").head(1).index
    combined_df.loc[idx, "Linelist"] = 1

    # AFPCASES_UNIQ slicer (1=AFP, 0=Others)
    combined_df["AFPCASES_UNIQ"] = np.where(combined_df["STCOND1"]==1, 1, 0)

    # --------------------------------------------------
    # 11. Preview & download
    # --------------------------------------------------
    st.success("Processing completed successfully")
    st.dataframe(combined_df.head(25))
    csv = combined_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Processed CSV",
        csv,
        "AFPLAB_Combined.csv",
        "text/csv"
    )

    # --------------------------------------------------
    # 12. Dashboard Filters
    # --------------------------------------------------
    st.markdown("---")
    st.header("AFPLAB Dashboard")
    st.sidebar.header("Dashboard Filters")
    provinces = st.sidebar.multiselect(
        "Select Province(s)", options=combined_df["PROVINCE"].unique(),
        default=combined_df["PROVINCE"].unique()
    )
    years = st.sidebar.multiselect(
        "Select Year(s)", options=combined_df["Year"].unique(),
        default=combined_df["Year"].unique()
    )
    afp_cases = st.sidebar.selectbox(
        "AFP Cases Filter", options=["All", "AFP", "Others"], index=0
    )

    filtered_df = combined_df[
        (combined_df["PROVINCE"].isin(provinces)) &
        (combined_df["Year"].isin(years))
    ]
    if afp_cases=="AFP":
        filtered_df = filtered_df[filtered_df["AFPCASES_UNIQ"]==1]
    elif afp_cases=="Others":
        filtered_df = filtered_df[filtered_df["AFPCASES_UNIQ"]==0]

    # --------------------------------------------------
    # 14. KPI and Chart for Linelist
    # --------------------------------------------------
    st.subheader("Linelist KPI")
    st.metric("Total Linelist Cases", int(filtered_df["Linelist"].sum()))
    linelist_df = filtered_df.groupby(["PROVINCE"])["Linelist"].sum().reset_index()
    fig_linelist = px.bar(
        linelist_df,
        x="PROVINCE",
        y="Linelist",
        text="Linelist",
        title="Linelist Cases by Province",
        barmode="stack"
    )
    fig_linelist.update_traces(textposition="outside", textfont_size=16)
    st.plotly_chart(fig_linelist, use_container_width=True)

    # --------------------------------------------------
    # 15. KPI and Chart for WPV1count
    # --------------------------------------------------
    st.subheader("WPV1count KPI")
    st.metric("Total WPV1count Cases", int(filtered_df["WPV1count"].sum()))
    wpv1count_df = filtered_df.groupby(["PROVINCE"])["WPV1count"].sum().reset_index()
    fig_wpv1 = px.bar(
        wpv1count_df,
        x="PROVINCE",
        y="WPV1count",
        text="WPV1count",
        title="WPV1count Cases by Province",
        barmode="stack"
    )
    fig_wpv1.update_traces(textposition="outside", textfont_size=16)
    st.plotly_chart(fig_wpv1, use_container_width=True)

    # --------------------------------------------------
    # 13. KPIs based on R_Result2
    # --------------------------------------------------
    kpi_cols = ["WPV1","VDPV1","cVDPV2","iVDPV1","aVDPV1","iVDPV3","aVDPV2","SL","NPEV","NVI"]
    kpis = {col: int(filtered_df["R_Result2"].str.contains(col, na=False).sum()) for col in kpi_cols}

    st.subheader("Key Performance Indicators (KPIs)")
    kpi_cols_chunk = [kpi_cols[i:i+5] for i in range(0, len(kpi_cols), 5)]
    for chunk in kpi_cols_chunk:
        cols = st.columns(len(chunk))
        for i, col in enumerate(chunk):
            cols[i].metric(col, kpis[col])

    # --------------------------------------------------
    # 16. R_Result2 stacked bar chart
    # --------------------------------------------------
    chart_df = filtered_df.groupby(["PROVINCE","R_Result2"]).size().reset_index(name="count")
    fig_r_result2 = px.bar(
        chart_df,
        y="PROVINCE",
        x="count",
        color="R_Result2",
        text="count",
        title="Cases by Province and R_Result2 (Stacked Bar)",
        orientation="h",
        barmode="stack"
    )
    fig_r_result2.update_traces(textposition="outside", textfont_size=16)
    fig_r_result2.update_layout(
        xaxis_title="Number of Cases",
        yaxis_title="Province",
        uniformtext_minsize=14,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig_r_result2, use_container_width=True)

# --------------------------------------------------
# 17. PROVINCE-LEVEL ML RISK CLASSIFICATION
# --------------------------------------------------
st.markdown("---")
st.header("🧠 Province-level Risk Classification (ML)")

# Sidebar toggle to enable ML
enable_risk_ml = st.sidebar.checkbox("Enable Province Risk ML", value=False)

if enable_risk_ml and not filtered_df.empty:

    # --------------------------------------------------
    # Step 1: Province-level aggregation
    # --------------------------------------------------
    province_df = filtered_df.groupby("PROVINCE").agg(
        total_cases = ("IDCODE", "count"),
        afp_cases   = ("AFPCASES_UNIQ", "sum"),
        wpv_cases   = ("R_WPV", "sum"),
        npev_cases  = ("R_NPEV", "sum"),
        nvi_cases   = ("R_Negative", "sum"),
        vdpv1       = ("R_VDPV1", "sum"),
        cvdpv2      = ("R_cVDPV2", "sum"),
        avdpv1      = ("R_aVDPV1", "sum"),
        ivdpv1      = ("R_iVDPV1", "sum"),
        ivdpv3      = ("R_iVDPV3", "sum"),
        avdpv2      = ("R_aVDPV2", "sum"),
    ).reset_index()

    province_df["vdpv_cases"] = (
        province_df["vdpv1"] +
        province_df["cvdpv2"] +
        province_df["avdpv1"] +
        province_df["ivdpv1"] +
        province_df["ivdpv3"] +
        province_df["avdpv2"]
    )

    # --------------------------------------------------
    # Step 2: Define Risk Level (Target y)
    # --------------------------------------------------
    def assign_risk(row):
        if row["wpv_cases"] > 0 or row["vdpv_cases"] > 0:
            return "High"
        elif row["npev_cases"] > 0 or row["afp_cases"] > 0:
            return "Medium"
        else:
            return "Low"

    province_df["Risk_Level"] = province_df.apply(assign_risk, axis=1)

    # --------------------------------------------------
    # Step 3: Prepare ML data
    # --------------------------------------------------
    features = ["total_cases", "afp_cases", "wpv_cases", "vdpv_cases", "npev_cases", "nvi_cases"]
    X = province_df[features]
    y = province_df["Risk_Level"]

    # --------------------------------------------------
    # Step 4: Train Random Forest safely
    # --------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from collections import Counter

    @st.cache_resource(show_spinner="Training Province Risk ML model...")
    def train_risk_model(X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    # Safe train/test split
    class_counts = Counter(y)
    if min(class_counts.values()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    rf_model = train_risk_model(X_train, y_train)

    # --------------------------------------------------
    # Step 5: Model Evaluation
    # --------------------------------------------------
    if len(y_test) > 0 and len(set(y_test)) > 0:
        st.subheader("📊 Model Evaluation")
        st.text(classification_report(y_test, rf_model.predict(X_test)))

    # --------------------------------------------------
    # Step 6: Predict Province Risk
    # --------------------------------------------------
    province_df["Predicted_Risk"] = rf_model.predict(X)

    # --------------------------------------------------
    # Step 7: Visuals - Risk Classification
    # --------------------------------------------------
    st.subheader("🗺️ Province Risk Classification")
    st.dataframe(
        province_df[["PROVINCE", "Predicted_Risk", "wpv_cases", "vdpv_cases", "npev_cases", "total_cases"]]
        .sort_values("Predicted_Risk")
    )

    fig_risk = px.bar(
        province_df,
        x="PROVINCE",
        color="Predicted_Risk",
        title="Province-level Risk Classification (ML)",
        text="Predicted_Risk"
    )
    fig_risk.update_traces(textposition="outside", textfont_size=14)
    st.plotly_chart(fig_risk, use_container_width=True)

    # Feature importance
    st.subheader("🔍 Risk Drivers (Feature Importance)")
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.dataframe(imp_df)

    # --------------------------------------------------
    # Step 8: Manual retrain button
    # --------------------------------------------------
    if st.sidebar.button("🔁 Force Model Retrain"):
        train_risk_model.clear()
        st.experimental_rerun()

    # Info
    st.info(
        "⚠️ Province risk classification is for surveillance decision support only. "
        "It does NOT replace laboratory confirmation or epidemiological investigation."
    )


