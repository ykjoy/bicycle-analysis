"""
경영학부 데이터 분석 도구 (Streamlit App)
- t-test: 두 그룹 평균 차이
- Linear Regression: 숫자 Y 예측
- Logistic Regression: 0/1 Y 예측 (구매/이탈 등)
- Clustering: 유형 분류
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(page_title="데이터 분석 도구", page_icon="📊", layout="wide")
st.title("📊 데이터 분석 도구")
st.caption("CSV를 올리고 분석을 선택하면 결과와 해석을 자동으로 보여드립니다.")

# ============================================================
# 1단계: 데이터 업로드
# ============================================================
st.header("1️⃣ 데이터 업로드")
uploaded = st.file_uploader("CSV 파일을 선택하세요", type=["csv"])

if uploaded is None:
    st.info("👆 분석할 CSV 파일을 업로드해주세요.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"✅ 데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

with st.expander("📋 데이터 미리보기 (상위 5행)"):
    st.dataframe(df.head())

# 컬럼 타입 분리 (선택 UI를 돕기 위함)
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# ============================================================
# 2단계: 분석 방법 선택
# ============================================================
st.header("2️⃣ 분석 방법 선택")
method = st.selectbox(
    "어떤 분석을 하시겠습니까?",
    ["t-test", "Linear Regression", "Logistic Regression", "Clustering"],
    help="t-test: 두 그룹 비교 / Regression: 영향 요인 분석 / Clustering: 유형 분류",
)

# ============================================================
# 3단계: 분석 + 해석
# ============================================================
st.header("3️⃣ 분석 결과와 해석")

# ----------- t-test -----------
if method == "t-test":
    st.subheader("두 그룹 평균 비교 (독립표본 t-test)")
    st.write("**질문 예시:** 남성과 여성의 평균 만족도가 다른가?")

    group_col = st.selectbox("그룹 컬럼 (예: 성별, 지역)", df.columns)
    value_col = st.selectbox("비교할 값 컬럼 (수치형)", num_cols)

    if st.button("🚀 분석 실행", type="primary"):
        groups = df[group_col].dropna().unique()

        if len(groups) != 2:
            st.error(f"⚠️ 그룹 컬럼에 정확히 2개의 값이 있어야 합니다. (현재: {len(groups)}개)")
        else:
            g1 = df[df[group_col] == groups[0]][value_col].dropna()
            g2 = df[df[group_col] == groups[1]][value_col].dropna()
            t_stat, p_val = stats.ttest_ind(g1, g2)

            # 기술통계
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{groups[0]} 평균", f"{g1.mean():.2f}", f"n={len(g1)}")
            col2.metric(f"{groups[1]} 평균", f"{g2.mean():.2f}", f"n={len(g2)}")
            col3.metric("평균 차이", f"{(g1.mean() - g2.mean()):.2f}")

            # 통계량
            st.write(f"**t 통계량:** {t_stat:.3f}  |  **p값:** {p_val:.4f}")

            # 해석
            st.markdown("### 📝 해석")
            if p_val < 0.05:
                st.success(
                    f"✅ **p = {p_val:.4f} < 0.05** → 두 그룹의 평균 차이가 "
                    f"**통계적으로 유의**합니다.\n\n"
                    f"즉, '{groups[0]}'와 '{groups[1]}' 그룹의 '{value_col}' 값 "
                    f"차이는 우연이 아닐 가능성이 높습니다."
                )
            else:
                st.warning(
                    f"⚠️ **p = {p_val:.4f} ≥ 0.05** → 통계적으로 유의한 차이가 "
                    f"발견되지 않았습니다.\n\n"
                    f"현재 표본만으로는 두 그룹 간 차이가 우연이 아니라고 "
                    f"단정하기 어렵습니다."
                )

            # 시각화
            fig = px.box(df, x=group_col, y=value_col, points="all",
                         title=f"{group_col}별 {value_col} 분포")
            st.plotly_chart(fig, use_container_width=True)

            # 평가 기준
            st.info(
                "💡 **평가 기준:** p < 0.05이면 '유의함'으로 판단합니다. "
                "다만 표본이 너무 크면 사소한 차이도 유의하게 나올 수 있으므로, "
                "**평균 차이의 크기**도 함께 보세요."
            )

# ----------- Linear Regression -----------
elif method == "Linear Regression":
    st.subheader("선형 회귀 (Y가 숫자일 때)")
    st.write("**질문 예시:** 광고비가 매출에 얼마나 영향을 주는가?")

    y_col = st.selectbox("종속변수 Y (예: 매출, 만족도)", num_cols)
    x_cols = st.multiselect(
        "독립변수 X (관심 있는 요인)",
        [c for c in num_cols if c != y_col]
    )
    control_cols = st.multiselect(
        "통제변수 (영향을 빼고 싶은 변수)",
        [c for c in num_cols if c not in [y_col] + x_cols]
    )

    if st.button("🚀 분석 실행", type="primary"):
        if not x_cols:
            st.error("⚠️ 독립변수를 1개 이상 선택해주세요.")
        else:
            all_x = x_cols + control_cols
            data = df[[y_col] + all_x].dropna()
            X = sm.add_constant(data[all_x])
            model = sm.OLS(data[y_col], X).fit()

            # 핵심 지표
            col1, col2, col3 = st.columns(3)
            col1.metric("R² (설명력)", f"{model.rsquared:.3f}")
            col2.metric("표본 수", f"{len(data)}")
            col3.metric("F 검정 p값", f"{model.f_pvalue:.4f}")

            # 분석 결과 표
            st.markdown("### 📊 회귀분석 결과")
            st.text(model.summary().as_text())

            # 자동 해석
            st.markdown("### 📝 자동 해석")
            for x in x_cols:
                coef = model.params[x]
                pval = model.pvalues[x]
                sig = "✅ 유의함" if pval < 0.05 else "⚠️ 유의하지 않음"
                direction = "증가" if coef > 0 else "감소"
                st.write(
                    f"- **{x}**: 계수 = {coef:.3f} (p = {pval:.4f}) → {sig}\n"
                    f"  → {x}가 1단위 늘면 {y_col}이 {abs(coef):.3f}만큼 {direction}"
                )

            # 모델 평가
            st.markdown("### 📈 모델 평가")
            r2 = model.rsquared
            if r2 >= 0.5:
                grade = "✅ 우수 (변동의 절반 이상을 설명)"
            elif r2 >= 0.3:
                grade = "🟡 양호"
            elif r2 >= 0.1:
                grade = "🟠 약함"
            else:
                grade = "🔴 매우 약함 (변수 추가 필요)"

            st.info(
                f"**R² = {r2:.3f}** → {y_col} 변동의 {r2*100:.1f}%를 모델이 설명합니다.\n\n"
                f"{grade}"
            )

# ----------- Logistic Regression -----------
elif method == "Logistic Regression":
    st.subheader("로지스틱 회귀 (Y가 0/1일 때)")
    st.write("**질문 예시:** 어떤 요인이 고객 이탈(0/1)에 영향을 주는가?")

    y_col = st.selectbox("종속변수 Y (0 또는 1로 된 컬럼)", num_cols)
    x_cols = st.multiselect("독립변수 X", [c for c in num_cols if c != y_col])
    control_cols = st.multiselect(
        "통제변수",
        [c for c in num_cols if c not in [y_col] + x_cols]
    )

    if st.button("🚀 분석 실행", type="primary"):
        if not x_cols:
            st.error("⚠️ 독립변수를 1개 이상 선택해주세요.")
        else:
            y_vals = set(df[y_col].dropna().unique())
            if not y_vals.issubset({0, 1}):
                st.error(
                    f"⚠️ Y는 0과 1로만 구성되어야 합니다. (현재 값: {y_vals})\n"
                    f"필요시 데이터를 0/1로 변환 후 다시 업로드해주세요."
                )
            else:
                all_x = x_cols + control_cols
                data = df[[y_col] + all_x].dropna()
                X = sm.add_constant(data[all_x])

                try:
                    model = sm.Logit(data[y_col], X).fit(disp=0)
                except Exception as e:
                    st.error(f"⚠️ 모델 학습 실패: {e}")
                    st.stop()

                # 핵심 지표
                col1, col2 = st.columns(2)
                col1.metric("Pseudo R²", f"{model.prsquared:.3f}")
                col2.metric("표본 수", f"{len(data)}")

                # 결과 표
                st.markdown("### 📊 로지스틱 회귀 결과")
                st.text(model.summary().as_text())

                # 자동 해석 (Odds Ratio)
                st.markdown("### 📝 자동 해석 (Odds Ratio)")
                for x in x_cols:
                    coef = model.params[x]
                    pval = model.pvalues[x]
                    odds = np.exp(coef)
                    sig = "✅ 유의함" if pval < 0.05 else "⚠️ 유의하지 않음"
                    st.write(
                        f"- **{x}**: Odds Ratio = {odds:.3f} (p = {pval:.4f}) → {sig}\n"
                        f"  → {x}가 1단위 늘면 '{y_col}=1'일 확률(승산)이 "
                        f"**{odds:.2f}배**가 됩니다."
                    )

                st.info(
                    "💡 **Odds Ratio 읽는 법**\n"
                    "- OR > 1: Y=1 가능성 ↑\n"
                    "- OR < 1: Y=1 가능성 ↓\n"
                    "- OR = 1: 영향 없음"
                )

# ----------- Clustering -----------
elif method == "Clustering":
    st.subheader("군집 분석 (유형 찾기)")
    st.write("**질문 예시:** 우리 고객은 몇 가지 유형으로 나뉘는가?")

    x_cols = st.multiselect("클러스터링에 사용할 변수 (수치형, 2개 이상)", num_cols)
    k = st.slider("클러스터 개수", 2, 10, 3)
    algo = st.selectbox(
        "알고리즘 (유사도 모델)",
        ["KMeans (거리 기반)", "Hierarchical (계층 병합)"]
    )

    if st.button("🚀 분석 실행", type="primary"):
        if len(x_cols) < 2:
            st.error("⚠️ 변수를 2개 이상 선택해주세요.")
        else:
            data = df[x_cols].dropna()
            scaled = StandardScaler().fit_transform(data)  # 표준화

            if algo.startswith("KMeans"):
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(scaled)

            # 평가 지표
            sil = silhouette_score(scaled, labels)
            st.metric(
                "실루엣 점수",
                f"{sil:.3f}",
                help="1에 가까울수록 군집이 잘 나뉨. 0.5↑ 양호, 0.25↑ 보통"
            )

            # 클러스터별 평균
            result = data.copy()
            result["cluster"] = labels
            st.markdown("### 📊 클러스터별 변수 평균")
            st.dataframe(result.groupby("cluster").mean().round(2))

            # 클러스터 크기
            st.markdown("### 👥 클러스터별 크기")
            st.bar_chart(result["cluster"].value_counts().sort_index())

            # PCA로 2D 시각화
            pca = PCA(n_components=2)
            xy = pca.fit_transform(scaled)
            viz = pd.DataFrame(xy, columns=["PC1", "PC2"])
            viz["cluster"] = labels.astype(str)
            fig = px.scatter(
                viz, x="PC1", y="PC2", color="cluster",
                title=f"{k}개 군집 시각화 (PCA 2D 투영)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 해석
            st.markdown("### 📝 해석")
            if sil >= 0.5:
                st.success(
                    f"✅ 실루엣 = {sil:.3f} → 군집이 잘 나뉘었습니다.\n\n"
                    "위 표에서 클러스터별 특징을 비교해 각 그룹에 이름을 붙여보세요."
                )
            elif sil >= 0.25:
                st.warning(
                    f"🟡 실루엣 = {sil:.3f} → 군집 구분이 보통입니다.\n\n"
                    "k값을 바꿔보거나 변수 조합을 다르게 시도해보세요."
                )
            else:
                st.error(
                    f"🔴 실루엣 = {sil:.3f} → 군집이 잘 안 나뉩니다.\n\n"
                    "변수를 줄이거나 다른 변수로 시도해보세요."
                )
