import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="Multi-Test Statistical Analysis Suite", page_icon="📊", layout="wide")

# Dark theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stMarkdown, .stText, .stDataFrame { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multi-Test Statistical Analysis Suite")
st.markdown("### *ANOVA (RBD) | t-test | Z-test | Tukey | Factorial ANOVA with Blocking*")
st.markdown("---")

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
for key in ['rbd_df', 'twog_df', 'tukey_df', 'factorial_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    st.markdown("**👩‍🎓 Mariam Muhsen Hussein**")
    st.markdown("*Master student – Advanced Manufacturing System Engineering*")
    st.markdown("🏛️ University of Baghdad – Al Khwarizmi Engineering College")
    st.markdown("👨‍🏫 Supervisor: Dr. Osamah Fadhil Abdulateef")
    st.markdown("---")
    
    test_type = st.selectbox(
        "Select Statistical Test:",
        ["Two-Way Factorial ANOVA with Blocking",
         "ANOVA (F-test) - RBD",
         "Two-Sample t-test",
         "Two-Sample Z-test",
         "Tukey HSD (Post-hoc)"]
    )
    data_source = st.radio("Data input method:", ["📂 Upload Excel/CSV", "✏️ Manual entry"])
    alpha = st.number_input("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    if st.button("🗑️ Clear Data"):
        for key in ['rbd_df', 'twog_df', 'tukey_df', 'factorial_df']:
            st.session_state[key] = None
        st.session_state.data_loaded = False
        st.rerun()

# ------------------------------------------------------------------
# Data loading functions (generic)
# ------------------------------------------------------------------
def load_factorial():
    if st.session_state.factorial_df is not None and st.session_state.data_loaded:
        st.success("✅ Data already loaded. You can now run the analysis.")
        return st.session_state.factorial_df

    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Factorial Data")
        st.info("File must contain columns related to: Block, FactorA, FactorB, Response (e.g., 'Block', 'Cutting Speed', 'Coolant', 'Surface Roughness')")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx', 'csv'], key="fact_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None

            st.write("Data preview:")
            st.dataframe(df.head())

            # Automatic column mapping
            rename = {}
            for col in df.columns:
                col_low = col.lower()
                if 'block' in col_low:
                    rename[col] = 'Block'
                elif 'cutting' in col_low or col_low == 'speed' or 'factor a' in col_low:
                    rename[col] = 'FactorA'
                elif 'coolant' in col_low or 'factor b' in col_low:
                    rename[col] = 'FactorB'
                elif 'roughness' in col_low or 'surface' in col_low or 'response' in col_low:
                    rename[col] = 'Response'
            df = df.rename(columns=rename)
            required = ['Block', 'FactorA', 'FactorB', 'Response']
            if all(c in df.columns for c in required):
                st.success("Columns successfully mapped!")
                st.session_state.factorial_df = df[required].dropna()
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.error(f"Missing required columns. Found: {list(df.columns)}. Expected: Block, FactorA, FactorB, Response (or similar names).")
        return None
    else:
        st.subheader("Manual Entry for Factorial ANOVA")
        with st.form(key="fact_manual"):
            col1, col2, col3 = st.columns(3)
            with col1:
                levels_a = st.number_input("Factor A levels", min_value=2, max_value=3, value=2)
            with col2:
                levels_b = st.number_input("Factor B levels", min_value=2, max_value=3, value=2)
            with col3:
                blocks = st.number_input("Number of blocks", min_value=2, max_value=3, value=2)
            names_a = [st.text_input(f"A{i+1}", value=f"A{i+1}", key=f"fa_{i}") for i in range(int(levels_a))]
            names_b = [st.text_input(f"B{j+1}", value=f"B{j+1}", key=f"fb_{j}") for j in range(int(levels_b))]
            names_block = [st.text_input(f"Block{k+1}", value=f"Block{k+1}", key=f"fk_{k}") for k in range(int(blocks))]
            rows = []
            for k in range(int(blocks)):
                st.write(f"**{names_block[k]}**")
                for i in range(int(levels_a)):
                    for j in range(int(levels_b)):
                        val = st.number_input(f"{names_a[i]} × {names_b[j]}", value=10.0, key=f"fact_man_{k}_{i}_{j}", step=0.1)
                        rows.append({'Block': names_block[k], 'FactorA': names_a[i], 'FactorB': names_b[j], 'Response': val})
            if st.form_submit_button("✅ Save Data"):
                st.session_state.factorial_df = pd.DataFrame(rows)
                st.session_state.data_loaded = True
                st.rerun()
        return st.session_state.factorial_df

def load_rbd():
    if st.session_state.rbd_df is not None and st.session_state.data_loaded:
        st.success("✅ Data already loaded.")
        return st.session_state.rbd_df
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload RBD Data")
        st.info("File must have columns: Treatment, Block, Response")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="rbd_upload")
        if uploaded:
            df = pd.read_excel(uploaded, engine='openpyxl') if not uploaded.name.endswith('.csv') else pd.read_csv(uploaded)
            if all(c in df.columns for c in ['Treatment', 'Block', 'Response']):
                st.dataframe(df)
                if st.button("✅ Save Data"):
                    st.session_state.rbd_df = df.dropna()
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.error("Missing columns: Treatment, Block, Response")
        return None
    else:
        st.subheader("Manual RBD Entry")
        with st.form(key="rbd_manual"):
            t = st.number_input("Treatments", min_value=2, max_value=10, value=3)
            b = st.number_input("Blocks", min_value=2, max_value=10, value=4)
            treat_names = [st.text_input(f"T{i+1}", value=f"T{i+1}", key=f"rt_{i}") for i in range(int(t))]
            block_names = [st.text_input(f"B{j+1}", value=f"B{j+1}", key=f"rb_{j}") for j in range(int(b))]
            data_input = []
            for i in range(int(t)):
                cols = st.columns(int(b))
                row = []
                for j in range(int(b)):
                    with cols[j]:
                        val = st.number_input(f"{treat_names[i]},{block_names[j]}", value=10.0, key=f"rbd_man_{i}_{j}", step=0.1)
                        row.append(val)
                data_input.append(row)
            if st.form_submit_button("✅ Save Data"):
                Y = np.array(data_input)
                rows = []
                for i, tr in enumerate(treat_names):
                    for j, bl in enumerate(block_names):
                        rows.append({'Treatment': tr, 'Block': bl, 'Response': Y[i,j]})
                st.session_state.rbd_df = pd.DataFrame(rows)
                st.session_state.data_loaded = True
                st.rerun()
        return st.session_state.rbd_df

def load_two_groups():
    if st.session_state.twog_df is not None and st.session_state.data_loaded:
        st.success("✅ Data already loaded.")
        return st.session_state.twog_df
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Two Groups")
        uploaded = st.file_uploader("CSV/Excel (first two columns = groups)", type=['xlsx','csv'], key="twog_upload")
        if uploaded:
            df = pd.read_excel(uploaded, engine='openpyxl') if not uploaded.name.endswith('.csv') else pd.read_csv(uploaded)
            if df.shape[1] >= 2:
                g1 = df.iloc[:,0].dropna().values
                g2 = df.iloc[:,1].dropna().values
                long_df = pd.DataFrame({'Group': ['G1']*len(g1)+['G2']*len(g2), 'Value': np.concatenate([g1,g2])})
                st.dataframe(long_df)
                if st.button("✅ Save Data"):
                    st.session_state.twog_df = long_df
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.error("Need at least two columns")
        return None
    else:
        st.subheader("Manual Two Groups")
        with st.form(key="twog_manual"):
            g1_str = st.text_area("Group1 values (comma separated)", "23,25,28,22,26")
            g2_str = st.text_area("Group2 values (comma separated)", "19,21,24,20,22")
            if st.form_submit_button("✅ Save Data"):
                try:
                    g1 = np.array([float(x.strip()) for x in g1_str.split(',')])
                    g2 = np.array([float(x.strip()) for x in g2_str.split(',')])
                    st.session_state.twog_df = pd.DataFrame({'Group': ['G1']*len(g1)+['G2']*len(g2), 'Value': np.concatenate([g1,g2])})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Invalid numbers")
        return st.session_state.twog_df

def load_tukey():
    if st.session_state.tukey_df is not None and st.session_state.data_loaded:
        st.success("✅ Data already loaded.")
        return st.session_state.tukey_df
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Tukey Data (each column = group)")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="tukey_upload")
        if uploaded:
            df = pd.read_excel(uploaded, engine='openpyxl') if not uploaded.name.endswith('.csv') else pd.read_csv(uploaded)
            st.dataframe(df)
            if st.button("✅ Save Data"):
                values, labels = [], []
                for col in df.columns:
                    vals = df[col].dropna().values
                    values.extend(vals)
                    labels.extend([col]*len(vals))
                st.session_state.tukey_df = pd.DataFrame({'Group': labels, 'Value': values})
                st.session_state.data_loaded = True
                st.rerun()
        return None
    else:
        st.subheader("Manual Tukey")
        with st.form(key="tukey_manual"):
            text_input = st.text_area("Groups (one per line: Group: val1,val2)", "Group1: 23,25,28\nGroup2: 19,21,24")
            if st.form_submit_button("✅ Save Data"):
                values, labels = [], []
                try:
                    for line in text_input.strip().split('\n'):
                        if ':' in line:
                            label, vals_str = line.split(':')
                            vals = [float(x.strip()) for x in vals_str.split(',')]
                            values.extend(vals)
                            labels.extend([label.strip()]*len(vals))
                    st.session_state.tukey_df = pd.DataFrame({'Group': labels, 'Value': values})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Parse error")
        return st.session_state.tukey_df

# ------------------------------------------------------------------
# Load data based on test type
# ------------------------------------------------------------------
if test_type == "Two-Way Factorial ANOVA with Blocking":
    load_factorial()
elif test_type == "ANOVA (F-test) - RBD":
    load_rbd()
elif test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
    load_two_groups()
elif test_type == "Tukey HSD (Post-hoc)":
    load_tukey()

# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
if st.button("🔬 Run Analysis", type="primary"):
    if not st.session_state.data_loaded:
        st.error("Please load data first using the 'Save Data' button.")
        st.stop()

    # ---------- Factorial ANOVA with Blocking ----------
    if test_type == "Two-Way Factorial ANOVA with Blocking":
        df = st.session_state.factorial_df

        # Type II ANOVA
        model = ols('Response ~ C(FactorB) + C(FactorA) + C(Block) + C(FactorA):C(FactorB)', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        grand_mean = df['Response'].mean()
        ss_total = np.sum((df['Response'] - grand_mean)**2)

        # Build results table
        results = []
        for name, key in [('Coolant', 'C(FactorB)'),
                          ('Cutting Speed', 'C(FactorA)'),
                          ('Block', 'C(Block)'),
                          ('Coolant × Cutting Speed', 'C(FactorA):C(FactorB)')]:
            ss = anova.loc[key, 'sum_sq']
            df_f = anova.loc[key, 'df']
            ms = ss / df_f
            f_val = anova.loc[key, 'F']
            p_val = anova.loc[key, 'PR(>F)']
            eta = ss / ss_total
            results.append([name, ss, df_f, ms, f_val, p_val, eta])
        # Residuals
        ss_res = anova.loc['Residual', 'sum_sq']
        df_res = anova.loc['Residual', 'df']
        ms_res = ss_res / df_res
        results.append(['Residuals', ss_res, df_res, ms_res, '', '', ''])

        anova_df = pd.DataFrame(results, columns=['Factor', 'Sum of Squares', 'df', 'Mean Square', 'F', 'p-value', 'η²'])
        for col in ['Sum of Squares', 'Mean Square', 'F', 'η²']:
            anova_df[col] = anova_df[col].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)
        anova_df['p-value'] = anova_df['p-value'].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)

        st.subheader("ANOVA Table (Type II)")
        st.dataframe(anova_df, use_container_width=True)

        # -------------------------------
        # Plots with 95% Confidence Intervals
        # -------------------------------
        st.subheader("Visualization")

        def mean_ci(vals):
            m = np.mean(vals)
            n = len(vals)
            if n > 1:
                sem = stats.sem(vals)
                ci_half = sem * stats.t.ppf(0.975, n-1)
            else:
                ci_half = 0
            return m, ci_half

        # Factor A (Cutting Speed)
        levels_a = sorted(df['FactorA'].unique())
        means_a, err_a = [], []
        for lev in levels_a:
            m, ci = mean_ci(df[df['FactorA']==lev]['Response'].values)
            means_a.append(m); err_a.append(ci)

        # Factor B (Coolant)
        levels_b = sorted(df['FactorB'].unique())
        means_b, err_b = [], []
        for lev in levels_b:
            m, ci = mean_ci(df[df['FactorB']==lev]['Response'].values)
            means_b.append(m); err_b.append(ci)

        # Interaction
        inter = df.groupby(['FactorA', 'FactorB'])['Response'].agg(list).reset_index()
        inter_means, inter_err = [], []
        for _, row in inter.iterrows():
            m, ci = mean_ci(row['Response'])
            inter_means.append(m); inter_err.append(ci)
        inter['mean'] = inter_means
        inter['err'] = inter_err

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Main effect of Factor A
        axes[0].bar(levels_a, means_a, yerr=err_a, capsize=5, color=['skyblue','steelblue'], edgecolor='black')
        axes[0].set_ylabel('Mean Response')
        axes[0].set_title('Effect of Factor A (95% CI)')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Main effect of Factor B
        axes[1].bar(levels_b, means_b, yerr=err_b, capsize=5, color=['lightcoral','darkred'], edgecolor='black')
        axes[1].set_ylabel('Mean Response')
        axes[1].set_title('Effect of Factor B (95% CI)')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Interaction plot
        for b in levels_b:
            sub = inter[inter['FactorB']==b]
            axes[2].errorbar(sub['FactorA'], sub['mean'], yerr=sub['err'], marker='o', label=b, capsize=5, linewidth=2)
        axes[2].set_xlabel('Factor A')
        axes[2].set_ylabel('Mean Response')
        axes[2].set_title('Interaction Plot (95% CI)')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

        # Block means (optional)
        block_means = df.groupby('Block')['Response'].mean().round(3)
        st.subheader("Block Means")
        st.dataframe(pd.DataFrame(block_means))
        fig2, ax = plt.subplots(figsize=(6,4))
        block_means.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
        ax.set_title('Mean Response by Block')
        ax.set_ylabel('Mean Response')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        # Statistical conclusion
        p_a = anova.loc['C(FactorA)', 'PR(>F)']
        p_b = anova.loc['C(FactorB)', 'PR(>F)']
        p_block = anova.loc['C(Block)', 'PR(>F)']
        p_int = anova.loc['C(FactorA):C(FactorB)', 'PR(>F)']
        st.subheader("Conclusion")
        if p_b < alpha:
            st.write(f"✅ **Factor B** has a significant effect (p = {p_b:.4f})")
        if p_a < alpha:
            st.write(f"✅ **Factor A** has a significant effect (p = {p_a:.4f})")
        if p_block < alpha:
            st.write(f"⚠️ **Block** has a significant effect (p = {p_block:.4f})")
        if p_int >= alpha:
            st.write(f"❌ **Interaction** is not significant (p = {p_int:.4f}) – factors act independently.")
        st.success("Analysis complete!")

    # ---------- RBD ANOVA ----------
    elif test_type == "ANOVA (F-test) - RBD":
        df = st.session_state.rbd_df
        model = ols('Response ~ C(Treatment) + C(Block)', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        st.subheader("ANOVA Results (RBD)")
        st.dataframe(anova.drop(index='Intercept').round(4), use_container_width=True)
        treat_means = df.groupby('Treatment')['Response'].mean()
        block_means = df.groupby('Block')['Response'].mean()
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Treatment Means**")
            st.dataframe(pd.DataFrame(treat_means))
        with col2:
            st.write("**Block Means**")
            st.dataframe(pd.DataFrame(block_means))
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        treat_means.plot(kind='bar', ax=ax[0], color='skyblue')
        block_means.plot(kind='bar', ax=ax[1], color='lightcoral')
        st.pyplot(fig)

    # ---------- Two-Sample t-test ----------
    elif test_type == "Two-Sample t-test":
        df = st.session_state.twog_df
        g1 = df[df['Group']=='G1']['Value'].values
        g2 = df[df['Group']=='G2']['Value'].values
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        st.subheader("Two-Sample t-test Results (Welch)")
        st.write(f"Group1: n={len(g1)}, mean={np.mean(g1):.3f}")
        st.write(f"Group2: n={len(g2)}, mean={np.mean(g2):.3f}")
        st.write(f"t = {t_stat:.4f}, p = {p_val:.4f}")
        if p_val < alpha:
            st.error("Reject H0: Significant difference.")
        else:
            st.success("Fail to reject H0: No significant difference.")

    # ---------- Two-Sample Z-test ----------
    elif test_type == "Two-Sample Z-test":
        df = st.session_state.twog_df
        g1 = df[df['Group']=='G1']['Value'].values
        g2 = df[df['Group']=='G2']['Value'].values
        se = np.sqrt(np.var(g1,ddof=1)/len(g1) + np.var(g2,ddof=1)/len(g2))
        z_stat = (np.mean(g1)-np.mean(g2))/se
        p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
        st.subheader("Two-Sample Z-test Results")
        st.write(f"z = {z_stat:.4f}, p = {p_val:.4f}")
        if p_val < alpha:
            st.error("Reject H0: Significant difference.")
        else:
            st.success("Fail to reject H0: No significant difference.")

    # ---------- Tukey HSD ----------
    elif test_type == "Tukey HSD (Post-hoc)":
        df = st.session_state.tukey_df
        tukey = pairwise_tukeyhsd(df['Value'], df['Group'], alpha=alpha)
        st.subheader("Tukey HSD Results")
        st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
        fig, ax = plt.subplots()
        tukey.plot_simultaneous(ax=ax)
        st.pyplot(fig)

    st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("Multi-Test Statistical Analysis Suite | Accurate Type II ANOVA | Generic for any dataset")
