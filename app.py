from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# ------------------------------------------------------------
# Page setup and project constants
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Learning Behaviour Clustering",
    page_icon="🎓",
    layout="wide",
)
RANDOM_STATE = 42
EXPECTED_COLUMNS = [
    "StudyHours", "Attendance", "Resources", "Extracurricular",
    "Motivation", "Internet", "Gender", "Age", "LearningStyle",
    "OnlineCourses", "Discussions", "AssignmentCompletion",
    "ExamScore", "EduTech", "StressLevel", "FinalGrade",
]
CLUSTER_FEATURES = [
    "StudyHours", "Attendance", "AssignmentCompletion",
    "OnlineCourses", "Discussions", "EduTech",
]
CONTEXT_COLUMNS = ["Motivation", "StressLevel", "Resources", "Internet", "Extracurricular"]
DATA_URL = "https://zenodo.org/records/16459132/files/merged_dataset.csv?download=1"


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    """Load a user-uploaded CSV, local project CSV, or the notebook's Zenodo source."""
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        source = "Uploaded CSV"
    else:
        candidates = [Path("student_performance.csv"), Path("merged_dataset.csv")]
        local = next((p for p in candidates if p.exists()), None)
        if local:
            data = pd.read_csv(local)
            source = str(local)
        else:
            try:
                data = pd.read_csv(DATA_URL)
                source = "Zenodo public dataset"
            except Exception as exc:
                raise RuntimeError(
                    "Could not load the public dataset. Upload the project CSV in the sidebar "
                    "or add student_performance.csv / merged_dataset.csv to the repository. "
                    f"Details: {exc}"
                ) from exc

    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "This CSV does not match the 16-column dataset used in the capstone. "
            f"Missing columns: {', '.join(missing)}"
        )
    data = data[EXPECTED_COLUMNS].copy()
    raw_rows = len(data)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    return data, source, raw_rows, duplicate_count


@st.cache_data(show_spinner="Preparing clustering analysis...")
def run_analysis(data):
    """Fit clustering on the six behavioural features; use other variables only for profiling."""
    df = data.copy()
    X = df[CLUSTER_FEATURES].copy()
    for col in CLUSTER_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    if X.isna().any().any():
        raise ValueError("One or more behavioural columns contain no usable numeric values.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Sample silhouette computations for responsiveness on the public dataset.
    k_values = list(range(2, 9))
    inertias, silhouettes = [], []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(
            X_scaled, labels, sample_size=min(5000, len(X_scaled)),
            random_state=RANDOM_STATE
        ))
    k_results = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes})
    best_k = int(k_results.loc[k_results["silhouette"].idxmax(), "k"])

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    k_sil = silhouette_score(
        X_scaled, df["Cluster"], sample_size=min(5000, len(X_scaled)),
        random_state=RANDOM_STATE
    )
    k_dbi = davies_bouldin_score(X_scaled, df["Cluster"])

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df.index)
    pca_df["Cluster"] = df["Cluster"].astype(str)

    # Use a reproducible sample for quadratic-cost comparison methods.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(3000, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), size=sample_n, replace=False)
    X_compare = X_scaled[sample_idx]

    agglo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels = agglo.fit_predict(X_compare)
    agg_sil = silhouette_score(
        X_compare, agg_labels, sample_size=min(2000, len(X_compare)),
        random_state=RANDOM_STATE
    )
    agg_dbi = davies_bouldin_score(X_compare, agg_labels)

    nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    kdist = np.sort(distances[:, -1])
    candidate_eps = np.unique(np.round(np.quantile(
        kdist, [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    ), 3))
    db_rows = []
    # Evaluate DBSCAN on a sample to limit deployment compute.
    for eps in candidate_eps:
        labels = DBSCAN(eps=float(eps), min_samples=5).fit_predict(X_compare)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = 100 * np.mean(labels == -1)
        valid = labels != -1
        if n_clusters >= 2 and valid.sum() > n_clusters:
            sil = silhouette_score(
                X_compare[valid], labels[valid],
                sample_size=min(2000, int(valid.sum())), random_state=RANDOM_STATE
            )
            dbi = davies_bouldin_score(X_compare[valid], labels[valid])
        else:
            sil, dbi = np.nan, np.nan
        db_rows.append({
            "eps": float(eps), "clusters": n_clusters,
            "noise_percent": noise_pct, "silhouette": sil, "davies_bouldin": dbi
        })
    db_results = pd.DataFrame(db_rows)
    valid_db = db_results.dropna(subset=["silhouette"])
    if not valid_db.empty:
        best_db = valid_db.loc[valid_db["silhouette"].idxmax()]
        db_sil = float(best_db["silhouette"])
        db_dbi = float(best_db["davies_bouldin"])
        db_labels = DBSCAN(eps=float(best_db["eps"]), min_samples=5).fit_predict(X_scaled)
        db_clusters = int(len(set(db_labels)) - (1 if -1 in db_labels else 0))
    else:
        db_sil, db_dbi, db_clusters = np.nan, np.nan, 0

    profiles = df.groupby("Cluster")[CLUSTER_FEATURES + [
        "Motivation", "StressLevel", "Resources", "Internet",
        "Extracurricular", "ExamScore", "FinalGrade"
    ]].mean(numeric_only=True).round(2)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=CLUSTER_FEATURES)
    centroids.index.name = "Cluster"
    global_means = df[CLUSTER_FEATURES].mean(numeric_only=True)

    recommendations = []
    for cid in sorted(df["Cluster"].unique()):
        row = profiles.loc[cid, CLUSTER_FEATURES]
        ordered = centroids.loc[cid].sort_values()
        high = ", ".join(ordered.tail(2).index)
        low = ", ".join(ordered.head(2).index)
        support = []
        if row["Attendance"] < global_means["Attendance"]:
            support.append("attendance support and reminders")
        if row["AssignmentCompletion"] < global_means["AssignmentCompletion"]:
            support.append("assignment planning and deadline support")
        if row["StudyHours"] < global_means["StudyHours"]:
            support.append("structured study scheduling")
        if row["OnlineCourses"] < global_means["OnlineCourses"]:
            support.append("guided supplementary online learning")
        if row["Discussions"] < global_means["Discussions"]:
            support.append("peer discussion or collaborative learning")
        if row["EduTech"] < global_means["EduTech"]:
            support.append("educational technology orientation")
        if not support:
            support = ["enrichment activities and advanced self-directed resources"]
        recommendations.append({
            "Cluster": int(cid),
            "Observed behaviour": f"Relatively high {high}; relatively low {low}.",
            "Recommended support": "; ".join(support[:3])
        })
    recommendation_table = pd.DataFrame(recommendations)

    learning_crosstab = pd.crosstab(
        df["Cluster"], df["LearningStyle"], normalize="index"
    ).mul(100).round(1)
    ari = adjusted_rand_score(df["LearningStyle"], df["Cluster"])
    nmi = normalized_mutual_info_score(df["LearningStyle"], df["Cluster"])

    outcome_profile = df.groupby("Cluster")[["ExamScore", "FinalGrade"]].agg(
        ["mean", "median", "std"]
    ).round(2)
    groups = [g["ExamScore"].dropna().values for _, g in df.groupby("Cluster")]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        anova = stats.f_oneway(*groups)
        grand_mean = df["ExamScore"].mean()
        valid_scores = df[["ExamScore", "Cluster"]].dropna()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
        )
        ss_total = ((valid_scores["ExamScore"] - grand_mean) ** 2).sum()
        eta_squared = float(ss_between / ss_total) if ss_total else np.nan
        anova_f, anova_p = float(anova.statistic), float(anova.pvalue)
    else:
        eta_squared, anova_f, anova_p = np.nan, np.nan, np.nan

    comparison = pd.DataFrame({
        "Model": ["K-Means", "Agglomerative (sample)", "DBSCAN (sample)"],
        "Silhouette": [k_sil, agg_sil, db_sil],
        "Davies-Bouldin": [k_dbi, agg_dbi, db_dbi],
    })

    return {
        "df": df, "X": X, "X_scaled": X_scaled, "X_pca": X_pca,
        "pca_variance": pca.explained_variance_ratio_,
        "k_results": k_results, "best_k": best_k, "k_sil": k_sil, "k_dbi": k_dbi,
        "agg_sil": agg_sil, "agg_dbi": agg_dbi, "db_sil": db_sil, "db_dbi": db_dbi,
        "db_results": db_results, "db_clusters": db_clusters,
        "profiles": profiles, "centroids": centroids, "recommendations": recommendation_table,
        "learning_crosstab": learning_crosstab, "ari": ari, "nmi": nmi,
        "outcome_profile": outcome_profile, "anova_f": anova_f, "anova_p": anova_p,
        "eta_squared": eta_squared, "comparison": comparison,
    }


# ------------------------------------------------------------
# Sidebar: data source and navigation
# ------------------------------------------------------------
st.sidebar.title("🎓 Capstone navigation")
st.sidebar.caption("TechCrush Cohort 8 · AI/ML Engineer Certification")
uploaded = st.sidebar.file_uploader("Optional: upload the project CSV", type=["csv"])
st.sidebar.info(
    "Expected columns include StudyHours, Attendance, AssignmentCompletion, "
    "OnlineCourses, Discussions, EduTech, LearningStyle, ExamScore and FinalGrade."
)

st.title("Student Learning-Style Clustering Based on Learning Behavior")
st.markdown(
    "An interactive unsupervised machine-learning capstone that identifies student "
    "behavioural segments and explores how they relate to reported learning styles "
    "and academic outcomes."
)

try:
    data, source, raw_rows, duplicate_count = load_data(uploaded)
    results = run_analysis(data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = results["df"]
st.caption(f"Data source: {source} · Exact duplicates removed: {duplicate_count:,}")

pages = [
    "Project overview", "Data explorer", "Exploratory analysis",
    "Cluster discovery", "Model comparison", "Cluster profiles",
    "Learning-style validation", "Academic outcomes", "Recommendations",
]
page = st.sidebar.radio("Choose a section", pages)

if page == "Project overview":
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", f"{raw_rows:,}")
    c2.metric("Analytical records", f"{len(df):,}")
    c3.metric("Behavioural features", len(CLUSTER_FEATURES))
    c4.metric("K-Means clusters", results["best_k"])
    st.markdown("### Research approach")
    st.write(
        "The app clusters students using six behavioural variables only. LearningStyle, "
        "ExamScore and FinalGrade are held out and analysed after cluster formation."
    )
    st.markdown("### Behavioural features used")
    st.write(", ".join(CLUSTER_FEATURES))
    st.markdown("### Methodology")
    st.markdown(
        "1. Validate the expected dataset columns and remove exact duplicate rows.\n"
        "2. Impute missing clustering-feature values with the column median.\n"
        "3. Standardise the six behavioural features.\n"
        "4. Select K-Means cluster count using silhouette scores for k = 2–8.\n"
        "5. Compare K-Means with Agglomerative clustering and DBSCAN.\n"
        "6. Profile clusters, then evaluate learning-style composition and academic outcomes."
    )
    st.warning(
        "Interpret clusters as patterns in this dataset, not permanent student identities. "
        "Associations do not establish causation."
    )

elif page == "Data explorer":
    st.subheader("Dataset preview")
    st.write(f"Rows after duplicate removal: **{len(df):,}** · Columns: **{df.shape[1]}**")
    st.dataframe(df.head(100), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values")
        st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    with right:
        st.markdown("#### Numeric summary")
        st.dataframe(df.describe().T, use_container_width=True)
    st.download_button(
        "Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_learning_behaviour_cleaned.csv",
        mime="text/csv",
    )

elif page == "Exploratory analysis":
    st.subheader("Exploratory data analysis")
    selected = st.multiselect("Select behavioural variables", CLUSTER_FEATURES, default=CLUSTER_FEATURES)
    if selected:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[selected], ax=ax)
        ax.set_title("Distribution of behavioural variables")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[selected].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Correlation matrix")
        st.pyplot(fig)
        plt.close(fig)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x="LearningStyle", ax=ax)
        ax.set_title("Reported learning-style categories")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x="ExamScore", kde=True, ax=ax)
        ax.set_title("Exam score distribution")
        st.pyplot(fig)
        plt.close(fig)

elif page == "Cluster discovery":
    st.subheader("K-Means cluster discovery")
    st.metric("Selected k", results["best_k"])
    st.write(
        f"Silhouette score: **{results['k_sil']:.3f}** · "
        f"Davies–Bouldin index: **{results['k_dbi']:.3f}**"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results["k_results"]["k"], results["k_results"]["inertia"], marker="o")
    axes[0].axvline(results["best_k"], linestyle="--")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(results["k_results"]["k"], results["k_results"]["silhouette"], marker="o")
    axes[1].axvline(results["best_k"], linestyle="--")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    st.pyplot(fig)
    plt.close(fig)

    pca_df = pd.DataFrame(results["X_pca"], columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Cluster"].astype(str).values
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", alpha=0.55, s=25, ax=ax)
    ax.set_title("PCA projection of K-Means behavioural clusters")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"PC1 explains {results['pca_variance'][0]:.1%} and PC2 explains "
        f"{results['pca_variance'][1]:.1%} of variance."
    )
    sizes = df["Cluster"].value_counts().sort_index().rename("Students").to_frame()
    sizes["Percentage"] = (sizes["Students"] / len(df) * 100).round(1)
    st.dataframe(sizes, use_container_width=True)

elif page == "Model comparison":
    st.subheader("Clustering algorithm comparison")
    st.dataframe(results["comparison"].round(3), use_container_width=True)
    st.caption(
        "For deployment responsiveness, Agglomerative clustering and DBSCAN comparisons "
        "are evaluated on a reproducible sample of up to 3,000 records; K-Means is fitted "
        "to all available analytical records. Compare metrics with this sampling caveat in mind."
    )
    st.markdown("#### DBSCAN parameter search")
    st.dataframe(results["db_results"].round(3), use_container_width=True)
    st.markdown("Higher silhouette is generally better; lower Davies–Bouldin is generally better. "
                "Neither metric alone determines educational usefulness.")

elif page == "Cluster profiles":
    st.subheader("Behavioural profiles")
    st.markdown("#### Mean values by K-Means cluster")
    st.dataframe(results["profiles"], use_container_width=True)
    st.markdown("#### Standardised cluster centroids")
    st.dataframe(results["centroids"].round(2), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(results["centroids"], annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Standardised behavioural cluster centroids")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Learning-style validation":
    st.subheader("Learning-style validation")
    st.write(
        "LearningStyle was not used to create the clusters. This section checks whether "
        "the behavioural clusters align with the dataset's reported learning-style categories."
    )
    c1, c2 = st.columns(2)
    c1.metric("Adjusted Rand Index (ARI)", f"{results['ari']:.3f}")
    c2.metric("Normalized Mutual Information (NMI)", f"{results['nmi']:.3f}")
    st.markdown("#### Learning-style composition within each cluster (%)")
    st.dataframe(results["learning_crosstab"], use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(results["learning_crosstab"], annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_title("Learning-style composition by behavioural cluster")
    ax.set_xlabel("Reported LearningStyle")
    ax.set_ylabel("K-Means cluster")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("A low ARI or NMI does not automatically mean clustering failed; behavioural groups may not map neatly to reported categories.")

elif page == "Academic outcomes":
    st.subheader("Academic outcome profiling")
    st.write("ExamScore and FinalGrade were not used as clustering inputs.")
    st.dataframe(results["outcome_profile"], use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Cluster", y="ExamScore", ax=ax)
    ax.set_title("Exam score by behavioural cluster")
    st.pyplot(fig)
    plt.close(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("ANOVA F-statistic", f"{results['anova_f']:.3f}" if pd.notna(results["anova_f"]) else "N/A")
    c2.metric("ANOVA p-value", f"{results['anova_p']:.4g}" if pd.notna(results["anova_p"]) else "N/A")
    c3.metric("Eta-squared", f"{results['eta_squared']:.3f}" if pd.notna(results["eta_squared"]) else "N/A")
    if pd.notna(results["anova_p"]):
        if results["anova_p"] < 0.05:
            st.info("The ANOVA detected a statistically significant difference in mean ExamScore across at least some clusters.")
        else:
            st.info("The ANOVA did not detect a statistically significant difference in mean ExamScore across clusters.")
    st.caption("ANOVA assumptions should be assessed before drawing strong conclusions; statistical association does not establish causation.")

elif page == "Recommendations":
    st.subheader("Suggested educational support")
    st.write(
        "Recommendations are rule-based and linked to behavioural features below the overall "
        "dataset mean. They are prompts for support planning, not diagnoses or fixed labels."
    )
    st.dataframe(results["recommendations"], use_container_width=True)
    st.download_button(
        "Download cluster assignments and recommendations",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_cluster_assignments.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "TechCrush Cohort 8 AI/ML Capstone · Student Learning-Style Clustering Based on Learning Behavior"
)
