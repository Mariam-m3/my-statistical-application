import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="Multi-Test Statistical Analysis Suite", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stMarkdown, .stText, .stDataFrame { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multi-Test Statistical Analysis Suite")
st.markdown("### *ANOVA (RBD) | t-test | Z-test | Tukey | Factorial ANOVA with Blocking*")
st.markdown("---")

# ==================================================================
# Session state initialization
# ==================================================================
if 'test_type' not in st.session_state:
    st.session_state.test_type = "Two-Way Factorial ANOVA with Blocking"
if 'factorial_df' not in st.session_state:
    st.session_state.factorial_df = None
if 'rbd_df' not in st.session_state:
    st.session_state.rbd_df = None
if 'twog_df' not in st.session_state:
    st.session_state.twog_df = None
if 'tukey_df' not in st.session_state:
    st.session_state.tukey_df = None
if 'factorial_loaded' not in st.session_state:
    st.session_state.factorial_loaded = False
if 'rbd_loaded' not in st.session_state:
    st.session_state.rbd_loaded = False
if 'twog_loaded' not in st.session_state:
    st.session_state.twog_loaded = False
if 'tukey_loaded' not in st.session_state:
    st.session_state.tukey_loaded = False
if 'fact_names' not in st.session_state:
    st.session_state.fact_names = None
if 'fact_needs_mapping' not in st.session_state:
    st.session_state.fact_needs_mapping = False

# ==================================================================
# Helper functions
# ==================================================================
def mean_ci(vals):
    m = np.mean(vals)
    n = len(vals)
    if n > 1:
        ci = stats.sem(vals) * stats.t.ppf(0.975, n-1)
    else:
        ci = 0
    return m, ci

def ensure_numeric(df, col='Response'):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=[col])

# ==================================================================
# Factorial ANOVA analysis function (must be defined before use)
# ==================================================================
def analyze_factorial(df, alpha, name_a, name_b):
    if 'Block' not in df.columns:
        df['Block'] = 'All'
    df['FactorA'] = df['FactorA'].astype(str)
    df['FactorB'] = df['FactorB'].astype(str)
    df['Block'] = df['Block'].astype(str)
    df = ensure_numeric(df, 'Response')
    if df.empty:
        st.error("No valid numeric data in Response column.")
        return

    if df['FactorA'].nunique() < 2:
        st.error(f"Factor A ('{name_a}') has only {df['FactorA'].nunique()} level(s). Need at least 2.")
        return
    if df['FactorB'].nunique() < 2:
        st.error(f"Factor B ('{name_b}') has only {df['FactorB'].nunique()} level(s). Need at least 2.")
        return
    if len(df) < 4:
        st.error("Not enough data points (need at least 4).")
        return

    try:
        formula = 'Response ~ C(FactorB) + C(FactorA) + C(Block) + C(FactorA):C(FactorB)'
        model = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        st.error(f"ANOVA failed: {e}")
        st.info("Possible reasons: insufficient data, singular design, or constant response values.")
        return

    grand_mean = df['Response'].mean()
    ss_total = np.sum((df['Response'] - grand_mean)**2)

    results = []
    for factor_key, factor_name in [('C(FactorB)', name_b), ('C(FactorA)', name_a),
                                    ('C(Block)', 'Block'), ('C(FactorA):C(FactorB)', f'{name_a} × {name_b}')]:
        if factor_key not in anova.index:
            continue
        ss = anova.loc[factor_key, 'sum_sq']
        df_f = anova.loc[factor_key, 'df']
        ms = ss / df_f
        f_val = anova.loc[factor_key, 'F']
        p_val = anova.loc[factor_key, 'PR(>F)']
        eta = ss / ss_total
        results.append([factor_name, ss, df_f, ms, f_val, p_val, eta])
    ss_res = anova.loc['Residual', 'sum_sq']
    df_res = anova.loc['Residual', 'df']
    ms_res = ss_res / df_res
    results.append(['Residuals', ss_res, df_res, ms_res, '', '', ''])

    anova_df = pd.DataFrame(results, columns=['Factor', 'SS', 'df', 'MS', 'F', 'p-value', 'η²'])
    for col in ['SS', 'MS', 'F', 'η²']:
        anova_df[col] = anova_df[col].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)
    anova_df['p-value'] = anova_df['p-value'].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)

    st.subheader("ANOVA Table (Type II)")
    st.dataframe(anova_df, use_container_width=True)

    st.subheader("Visualization")
    levels_a = sorted(df['FactorA'].unique())
    means_a, err_a = [], []
    for lev in levels_a:
        m, ci = mean_ci(df[df['FactorA']==lev]['Response'].values)
        means_a.append(m); err_a.append(ci)
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.bar(levels_a, means_a, yerr=err_a, capsize=5, color=['skyblue','steelblue'], edgecolor='black')
    ax1.set_ylabel('Mean Response')
    ax1.set_title(f'Effect of {name_a} (95% CI)')
    ax1.set_xlabel(name_a)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    levels_b = sorted(df['FactorB'].unique())
    means_b, err_b = [], []
    for lev in levels_b:
        m, ci = mean_ci(df[df['FactorB']==lev]['Response'].values)
        means_b.append(m); err_b.append(ci)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.bar(levels_b, means_b, yerr=err_b, capsize=5, color=['lightcoral','darkred'], edgecolor='black')
    ax2.set_ylabel('Mean Response')
    ax2.set_title(f'Effect of {name_b} (95% CI)')
    ax2.set_xlabel(name_b)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    inter = df.groupby(['FactorA', 'FactorB'])['Response'].agg(list).reset_index()
    inter_means, inter_err = [], []
    for _, row in inter.iterrows():
        m, ci = mean_ci(row['Response'])
        inter_means.append(m); inter_err.append(ci)
    inter['mean'] = inter_means
    inter['err'] = inter_err
    fig3, ax3 = plt.subplots(figsize=(6,4))
    for b in levels_b:
        sub = inter[inter['FactorB'] == b]
        ax3.errorbar(sub['FactorA'], sub['mean'], yerr=sub['err'], marker='o', label=b, capsize=5, linewidth=2)
    ax3.set_xlabel(name_a)
    ax3.set_ylabel('Mean Response')
    ax3.set_title('Interaction Plot (95% CI)')
    ax3.legend(title=name_b)
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)

    if len(df['Block'].unique()) > 1:
        block_means = df.groupby('Block')['Response'].mean().round(3)
        st.subheader("Block Means")
        st.dataframe(pd.DataFrame(block_means))
        fig4, ax4 = plt.subplots(figsize=(6,4))
        block_means.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax4)
        ax4.set_title('Mean Response by Block')
        ax4.set_ylabel('Mean Response')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig4)

    if 'C(FactorA)' in anova.index:
        p_a = anova.loc['C(FactorA)', 'PR(>F)']
        if p_a < alpha:
            st.write(f"✅ **{name_a}** has a significant effect (p = {p_a:.4f})")
        else:
            st.write(f"❌ **{name_a}** effect not significant (p = {p_a:.4f})")
    if 'C(FactorB)' in anova.index:
        p_b = anova.loc['C(FactorB)', 'PR(>F)']
        if p_b < alpha:
            st.write(f"✅ **{name_b}** has a significant effect (p = {p_b:.4f})")
        else:
            st.write(f"❌ **{name_b}** effect not significant (p = {p_b:.4f})")
    if 'C(Block)' in anova.index:
        p_block = anova.loc['C(Block)', 'PR(>F)']
        if p_block < alpha:
            st.write(f"⚠️ **Block** has a significant effect (p = {p_block:.4f})")
    if 'C(FactorA):C(FactorB)' in anova.index:
        p_int = anova.loc['C(FactorA):C(FactorB)', 'PR(>F)']
        if p_int >= alpha:
            st.write(f"❌ **Interaction** not significant (p = {p_int:.4f}) – factors act independently.")
    st.success("Analysis complete!")

# ==================================================================
# Data loading functions
# ==================================================================
def manual_column_mapping_factorial():
    if 'fact_raw_df' not in st.session_state:
        return
    df_raw = st.session_state.fact_raw_df
    st.subheader("Manual Column Mapping")
    st.write("Please select which columns correspond to:")
    cols = df_raw.columns.tolist()
    col_block = st.selectbox("Block column (optional, select 'None' if not present)", ['None'] + cols)
    col_factorA = st.selectbox("Factor A column (e.g., Cutting Speed levels)", cols)
    col_factorB = st.selectbox("Factor B column (e.g., Coolant levels)", [c for c in cols if c != col_factorA])
    col_response = st.selectbox("Response column (numerical values)", [c for c in cols if c not in [col_factorA, col_factorB]])
    
    if st.button("✅ Apply Mapping and Save Data"):
        if col_factorA == col_factorB or col_factorA == col_response or col_factorB == col_response:
            st.error("Please select distinct columns for Factor A, Factor B, and Response.")
            return
        # Build long format dataframe
        df_long = pd.DataFrame({
            'FactorA': df_raw[col_factorA].astype(str),
            'FactorB': df_raw[col_factorB].astype(str),
            'Response': pd.to_numeric(df_raw[col_response], errors='coerce')
        })
        if col_block != 'None':
            df_long['Block'] = df_raw[col_block].astype(str)
        else:
            df_long['Block'] = 'All'
        df_long = df_long.dropna(subset=['Response'])
        if df_long.empty:
            st.error("No valid numeric data in Response column.")
            return
        if df_long['FactorA'].nunique() < 2:
            st.error(f"Factor A column '{col_factorA}' has only {df_long['FactorA'].nunique()} unique value(s). Need at least 2.")
            return
        if df_long['FactorB'].nunique() < 2:
            st.error(f"Factor B column '{col_factorB}' has only {df_long['FactorB'].nunique()} unique value(s). Need at least 2.")
            return
        st.session_state.factorial_df = df_long
        st.session_state.fact_names = {'FactorA': col_factorA, 'FactorB': col_factorB}
        st.session_state.factorial_loaded = True
        st.session_state.fact_needs_mapping = False
        del st.session_state.fact_raw_df
        st.rerun()

def load_factorial():
    if st.session_state.factorial_loaded and st.session_state.factorial_df is not None:
        st.success("✅ Factorial data already loaded. You can run analysis.")
        return

    if data_source == "✏️ Manual entry":
        st.subheader("Manual Entry for Factorial ANOVA")
        with st.form(key="fact_manual"):
            col1, col2 = st.columns(2)
            with col1:
                name_a = st.text_input("Name of Factor A (e.g., 'Cutting Speed'):", value="Factor A")
                levels_a = st.number_input("Levels of Factor A", min_value=2, max_value=5, value=2)
            with col2:
                name_b = st.text_input("Name of Factor B (e.g., 'Coolant'):", value="Factor B")
                levels_b = st.number_input("Levels of Factor B", min_value=2, max_value=5, value=2)
            has_block = st.checkbox("Include Block factor?", value=True)
            if has_block:
                blocks = st.number_input("Number of blocks", min_value=2, max_value=5, value=2)
            else:
                blocks = 1
            levels_a_vals = [st.text_input(f"{name_a} level {i+1}", value=f"L{i+1}", key=f"fa_{i}") for i in range(int(levels_a))]
            levels_b_vals = [st.text_input(f"{name_b} level {j+1}", value=f"L{j+1}", key=f"fb_{j}") for j in range(int(levels_b))]
            if has_block:
                block_names = [st.text_input(f"Block {k+1}", value=f"B{k+1}", key=f"fk_{k}") for k in range(int(blocks))]
            else:
                block_names = ['All']
                blocks = 1
            rows = []
            for k in range(int(blocks)):
                if has_block:
                    st.write(f"**{block_names[k]}**")
                for i in range(int(levels_a)):
                    for j in range(int(levels_b)):
                        val = st.number_input(f"{levels_a_vals[i]} × {levels_b_vals[j]}", value=10.0, key=f"fact_man_{k}_{i}_{j}", step=0.1)
                        row = {'FactorA': levels_a_vals[i], 'FactorB': levels_b_vals[j], 'Response': val}
                        if has_block:
                            row['Block'] = block_names[k]
                        else:
                            row['Block'] = 'All'
                        rows.append(row)
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                df = pd.DataFrame(rows)
                df = ensure_numeric(df, 'Response')
                if df.empty:
                    st.error("No valid numeric data. Please enter numbers.")
                else:
                    st.session_state.factorial_df = df
                    st.session_state.fact_names = {'FactorA': name_a, 'FactorB': name_b}
                    st.session_state.factorial_loaded = True
                    st.rerun()
        return
    else:  # Upload file
        st.subheader("Upload Factorial Data")
        st.info("If automatic detection fails, you will be asked to manually select columns.")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="fact_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            st.write("Preview of uploaded file:")
            st.dataframe(df_raw.head())
            st.session_state.fact_raw_df = df_raw
            st.session_state.fact_needs_mapping = True
            st.rerun()
        return

def load_rbd():
    if st.session_state.rbd_loaded and st.session_state.rbd_df is not None:
        st.success("✅ RBD data already loaded.")
        return

    if data_source == "✏️ Manual entry":
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
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                Y = np.array(data_input)
                rows = []
                for i, tr in enumerate(treat_names):
                    for j, bl in enumerate(block_names):
                        rows.append({'Treatment': tr, 'Block': bl, 'Response': Y[i,j]})
                df = pd.DataFrame(rows)
                df = ensure_numeric(df, 'Response')
                if df.empty:
                    st.error("No valid numeric data.")
                else:
                    st.session_state.rbd_df = df
                    st.session_state.rbd_loaded = True
                    st.rerun()
        return
    else:
        st.subheader("Upload RBD Data")
        st.info("File must have columns: Treatment, Block, Response (in that order).")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="rbd_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            st.write("Preview:")
            st.dataframe(df_raw.head())
            if df_raw.shape[1] >= 3:
                df = pd.DataFrame({
                    'Treatment': df_raw.iloc[:,0].astype(str),
                    'Block': df_raw.iloc[:,1].astype(str),
                    'Response': pd.to_numeric(df_raw.iloc[:,2], errors='coerce')
                })
                df = df.dropna(subset=['Response'])
                if df.empty:
                    st.error("No valid numeric data in Response column.")
                    return
                st.session_state.rbd_df = df
                st.session_state.rbd_loaded = True
                st.rerun()
            else:
                st.error("Need at least 3 columns: Treatment, Block, Response")
        return

def analyze_rbd(df, alpha):
    model = ols('Response ~ C(Treatment) + C(Block)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    st.subheader("ANOVA Table (RBD)")
    if 'Intercept' in anova.index:
        anova_display = anova.drop(index='Intercept').round(4)
    else:
        anova_display = anova.round(4)
    st.dataframe(anova_display, use_container_width=True)
    treat_means = df.groupby('Treatment')['Response'].mean()
    block_means = df.groupby('Block')['Response'].mean()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Treatment Means**")
        st.dataframe(pd.DataFrame(treat_means))
    with col2:
        st.write("**Block Means**")
        st.dataframe(pd.DataFrame(block_means))
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    treat_means.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Treatment Means')
    block_means.plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Block Means')
    st.pyplot(fig)
    p_treat = anova.loc['C(Treatment)', 'PR(>F)']
    p_block = anova.loc['C(Block)', 'PR(>F)']
    if p_treat < alpha:
        st.write("✅ Treatment has significant effect.")
    else:
        st.write("❌ Treatment effect not significant.")
    if p_block < alpha:
        st.write("⚠️ Block has significant effect.")
    st.success("Analysis complete!")

def load_two_groups():
    if st.session_state.twog_loaded and st.session_state.twog_df is not None:
        st.success("✅ Two-group data already loaded.")
        return

    if data_source == "✏️ Manual entry":
        st.subheader("Manual Two Groups")
        with st.form(key="twog_manual"):
            g1_str = st.text_area("Group1 values (comma separated)", "23,25,28,22,26")
            g2_str = st.text_area("Group2 values (comma separated)", "19,21,24,20,22")
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                try:
                    g1 = np.array([float(x.strip()) for x in g1_str.split(',')])
                    g2 = np.array([float(x.strip()) for x in g2_str.split(',')])
                    df = pd.DataFrame({'Group': ['G1']*len(g1)+['G2']*len(g2), 'Value': np.concatenate([g1,g2])})
                    st.session_state.twog_df = df
                    st.session_state.twog_loaded = True
                    st.rerun()
                except:
                    st.error("Invalid numbers")
        return
    else:
        st.subheader("Upload Two Groups")
        uploaded = st.file_uploader("CSV/Excel (first two columns = groups)", type=['xlsx','csv'], key="twog_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            if df_raw.shape[1] >= 2:
                g1 = df_raw.iloc[:,0].dropna().values
                g2 = df_raw.iloc[:,1].dropna().values
                long_df = pd.DataFrame({'Group': ['G1']*len(g1)+['G2']*len(g2), 'Value': np.concatenate([g1,g2])})
                st.dataframe(long_df)
                if st.button("✅ Save Data"):
                    st.session_state.twog_df = long_df
                    st.session_state.twog_loaded = True
                    st.rerun()
            else:
                st.error("Need at least two columns")
        return

def analyze_two_groups(df, test_type, alpha):
    g1 = df[df['Group']=='G1']['Value'].values
    g2 = df[df['Group']=='G2']['Value'].values
    if test_type == "Two-Sample t-test":
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        st.subheader("Two-Sample t-test Results (Welch)")
        st.write(f"Group1: n={len(g1)}, mean={np.mean(g1):.3f}, std={np.std(g1, ddof=1):.3f}")
        st.write(f"Group2: n={len(g2)}, mean={np.mean(g2):.3f}, std={np.std(g2, ddof=1):.3f}")
        st.write(f"t = {t_stat:.4f}, p = {p_val:.4f}")
    else:
        se = np.sqrt(np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))
        z_stat = (np.mean(g1)-np.mean(g2))/se
        p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
        st.subheader("Two-Sample Z-test Results")
        st.write(f"Group1: n={len(g1)}, mean={np.mean(g1):.3f}")
        st.write(f"Group2: n={len(g2)}, mean={np.mean(g2):.3f}")
        st.write(f"z = {z_stat:.4f}, p = {p_val:.4f}")
    if p_val < alpha:
        st.error("Reject H0: Significant difference.")
    else:
        st.success("Fail to reject H0: No significant difference.")
    fig, ax = plt.subplots()
    ax.boxplot([g1, g2], labels=['Group 1', 'Group 2'])
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Two Groups')
    st.pyplot(fig)
    st.success("Analysis complete!")

def load_tukey():
    if st.session_state.tukey_loaded and st.session_state.tukey_df is not None:
        st.success("✅ Tukey data already loaded.")
        return

    if data_source == "✏️ Manual entry":
        st.subheader("Manual Tukey")
        with st.form(key="tukey_manual"):
            text_input = st.text_area("Groups (one per line: Group: val1,val2)", "Group1: 23,25,28\nGroup2: 19,21,24")
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                values, labels = [], []
                try:
                    for line in text_input.strip().split('\n'):
                        if ':' in line:
                            label, vals_str = line.split(':')
                            vals = [float(x.strip()) for x in vals_str.split(',')]
                            values.extend(vals)
                            labels.extend([label.strip()]*len(vals))
                    df = pd.DataFrame({'Group': labels, 'Value': values})
                    st.session_state.tukey_df = df
                    st.session_state.tukey_loaded = True
                    st.rerun()
                except:
                    st.error("Parse error")
        return
    else:
        st.subheader("Upload Tukey Data (each column = group)")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="tukey_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            st.dataframe(df_raw)
            if st.button("✅ Save Data"):
                values, labels = [], []
                for col in df_raw.columns:
                    vals = df_raw[col].dropna().values
                    values.extend(vals)
                    labels.extend([col]*len(vals))
                df = pd.DataFrame({'Group': labels, 'Value': values})
                st.session_state.tukey_df = df
                st.session_state.tukey_loaded = True
                st.rerun()
        return

def analyze_tukey(df, alpha):
    tukey = pairwise_tukeyhsd(df['Value'], df['Group'], alpha=alpha)
    st.subheader("Tukey HSD Results")
    st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
    fig, ax = plt.subplots()
    tukey.plot_simultaneous(ax=ax)
    st.pyplot(fig)
    st.success("Analysis complete!")

# ==================================================================
# Sidebar
# ==================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    st.markdown("**👩‍🎓 Mariam Muhsen Hussein**")
    st.markdown("*Master student – Advanced Manufacturing System Engineering*")
    st.markdown("🏛️ University of Baghdad – Al Khwarizmi Engineering College")
    st.markdown("👨‍🏫 Supervisor: Dr. Osamah Fadhil Abdulateef")
    st.markdown("---")
    
    new_test_type = st.selectbox(
        "Select Statistical Test:",
        ["Two-Way Factorial ANOVA with Blocking",
         "ANOVA (F-test) - RBD",
         "Two-Sample t-test",
         "Two-Sample Z-test",
         "Tukey HSD (Post-hoc)"]
    )
    if new_test_type != st.session_state.test_type:
        st.session_state.test_type = new_test_type
        # Reset loaded flag for new test
        if new_test_type == "Two-Way Factorial ANOVA with Blocking":
            st.session_state.factorial_loaded = False
        elif new_test_type == "ANOVA (F-test) - RBD":
            st.session_state.rbd_loaded = False
        elif new_test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
            st.session_state.twog_loaded = False
        elif new_test_type == "Tukey HSD (Post-hoc)":
            st.session_state.tukey_loaded = False
        st.rerun()
    
    data_source = st.radio("Data input method:", ["✏️ Manual entry", "📂 Upload Excel/CSV"])
    alpha = st.number_input("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    
    if st.button("🗑️ Clear Current Data"):
        if st.session_state.test_type == "Two-Way Factorial ANOVA with Blocking":
            st.session_state.factorial_df = None
            st.session_state.factorial_loaded = False
        elif st.session_state.test_type == "ANOVA (F-test) - RBD":
            st.session_state.rbd_df = None
            st.session_state.rbd_loaded = False
        elif st.session_state.test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
            st.session_state.twog_df = None
            st.session_state.twog_loaded = False
        elif st.session_state.test_type == "Tukey HSD (Post-hoc)":
            st.session_state.tukey_df = None
            st.session_state.tukey_loaded = False
        st.rerun()

# ==================================================================
# Main logic: load data and run analysis
# ==================================================================
# First, handle manual mapping if needed
if st.session_state.test_type == "Two-Way Factorial ANOVA with Blocking" and st.session_state.get('fact_needs_mapping', False):
    manual_column_mapping_factorial()
    st.stop()

# Then load data based on test type
if st.session_state.test_type == "Two-Way Factorial ANOVA with Blocking":
    load_factorial()
elif st.session_state.test_type == "ANOVA (F-test) - RBD":
    load_rbd()
elif st.session_state.test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
    load_two_groups()
elif st.session_state.test_type == "Tukey HSD (Post-hoc)":
    load_tukey()

# Run analysis button
if st.button("🔬 Run Analysis", type="primary"):
    if st.session_state.test_type == "Two-Way Factorial ANOVA with Blocking":
        if not st.session_state.factorial_loaded or st.session_state.factorial_df is None:
            st.error("Please load factorial data first.")
        else:
            df = st.session_state.factorial_df
            if st.session_state.fact_names:
                name_a = st.session_state.fact_names.get('FactorA', 'Factor A')
                name_b = st.session_state.fact_names.get('FactorB', 'Factor B')
            else:
                name_a, name_b = 'Factor A', 'Factor B'
            analyze_factorial(df, alpha, name_a, name_b)
    elif st.session_state.test_type == "ANOVA (F-test) - RBD":
        if not st.session_state.rbd_loaded or st.session_state.rbd_df is None:
            st.error("Please load RBD data first.")
        else:
            analyze_rbd(st.session_state.rbd_df, alpha)
    elif st.session_state.test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
        if not st.session_state.twog_loaded or st.session_state.twog_df is None:
            st.error("Please load two-group data first.")
        else:
            analyze_two_groups(st.session_state.twog_df, st.session_state.test_type, alpha)
    elif st.session_state.test_type == "Tukey HSD (Post-hoc)":
        if not st.session_state.tukey_loaded or st.session_state.tukey_df is None:
            st.error("Please load Tukey data first.")
        else:
            analyze_tukey(st.session_state.tukey_df, alpha)
