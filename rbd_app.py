import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

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
        ["ANOVA (F-test) - RBD",
         "Two-Sample t-test",
         "Two-Sample Z-test",
         "Tukey HSD (Post-hoc)",
         "Two-Way Factorial ANOVA with Blocking"]
    )
    data_source = st.radio("Data input method:", ["✏️ Manual entry", "📂 Upload Excel/CSV"])
    alpha = st.number_input("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    if st.button("🗑️ Clear Data"):
        for key in list(st.session_state.keys()):
            if key not in ['data_loaded', 'analysis_done']:
                del st.session_state[key]
        st.session_state.data_loaded = False
        st.rerun()

# --------------------------------------------------------------
# Load RBD data
# --------------------------------------------------------------
def load_rbd():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for RBD")
        with st.form(key="rbd_form"):
            col1, col2 = st.columns(2)
            with col1:
                t = st.number_input("Number of treatments", min_value=2, max_value=10, value=3)
            with col2:
                b = st.number_input("Number of blocks", min_value=2, max_value=10, value=4)
            treat_names = []
            block_names = []
            st.write("**Treatment names:**")
            cols = st.columns(int(t))
            for i in range(int(t)):
                with cols[i]:
                    name = st.text_input(f"Treatment {i+1}", value=f"T{i+1}")
                    treat_names.append(name)
            st.write("**Block names:**")
            cols = st.columns(int(b))
            for j in range(int(b)):
                with cols[j]:
                    name = st.text_input(f"Block {j+1}", value=f"B{j+1}")
                    block_names.append(name)
            st.write("**Response values:**")
            data_input = []
            for i in range(int(t)):
                cols = st.columns(int(b))
                row = []
                for j in range(int(b)):
                    with cols[j]:
                        val = st.number_input(f"{treat_names[i]}, {block_names[j]}", value=10.0, key=f"rbd_{i}_{j}", step=0.1)
                        row.append(val)
                data_input.append(row)
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                Y = np.array(data_input, dtype=float)
                rows = []
                for i, tr in enumerate(treat_names):
                    for j, bl in enumerate(block_names):
                        rows.append({'Treatment': tr, 'Block': bl, 'Response': Y[i, j]})
                df = pd.DataFrame(rows)
                st.session_state.rbd_df = df
                st.session_state.data_loaded = True
                st.success("Data saved!")
                st.rerun()
        return st.session_state.get('rbd_df', None)
    else:
        st.subheader("Upload RBD Data")
        st.info("File must have columns: Treatment, Block, Response")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="rbd_upload")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')
            if all(col in df.columns for col in ['Treatment', 'Block', 'Response']):
                st.dataframe(df)
                if st.button("✅ Save Uploaded Data"):
                    st.session_state.rbd_df = df.dropna()
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.error("Missing columns: Treatment, Block, Response")
        return None

# --------------------------------------------------------------
# Load two groups data (t-test / Z-test)
# --------------------------------------------------------------
def load_two_groups():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for Two Groups")
        with st.form(key="twog_form"):
            g1_str = st.text_area("Group 1 values (comma separated)", "23,25,28,22,26")
            g2_str = st.text_area("Group 2 values (comma separated)", "19,21,24,20,22")
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                try:
                    g1 = np.array([float(x.strip()) for x in g1_str.split(',')])
                    g2 = np.array([float(x.strip()) for x in g2_str.split(',')])
                    st.session_state.twog_df = pd.DataFrame({'Group': ['G1']*len(g1) + ['G2']*len(g2),
                                                              'Value': np.concatenate([g1, g2])})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Invalid numbers.")
        return st.session_state.get('twog_df', None)
    else:
        st.subheader("Upload Two Groups Data")
        uploaded = st.file_uploader("CSV/Excel (first two columns = groups)", type=['xlsx','csv'], key="twog_upload")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')
            if df.shape[1] >= 2:
                g1 = df.iloc[:,0].dropna().values
                g2 = df.iloc[:,1].dropna().values
                long_df = pd.DataFrame({'Group': ['G1']*len(g1) + ['G2']*len(g2),
                                        'Value': np.concatenate([g1, g2])})
                st.dataframe(long_df)
                if st.button("✅ Save Uploaded Data"):
                    st.session_state.twog_df = long_df
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.error("Need at least two columns.")
        return None

# --------------------------------------------------------------
# Load Tukey data
# --------------------------------------------------------------
def load_tukey():
    if data_source == "✏️ Manual entry":
        st.subheader("Manual Data Entry for Tukey HSD")
        st.write("One group per line: GroupName: val1,val2,...")
        with st.form(key="tukey_form"):
            text_input = st.text_area("Groups data", "Group1: 23,25,28\nGroup2: 19,21,24\nGroup3: 15,18,20")
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                values, labels = [], []
                try:
                    for line in text_input.strip().split('\n'):
                        if ':' in line:
                            label, vals_str = line.split(':')
                            vals = [float(x.strip()) for x in vals_str.split(',')]
                            values.extend(vals)
                            labels.extend([label.strip()] * len(vals))
                    st.session_state.tukey_df = pd.DataFrame({'Group': labels, 'Value': values})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Parse error.")
        return st.session_state.get('tukey_df', None)
    else:
        st.subheader("Upload Tukey Data (each column = group)")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="tukey_upload")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')
            st.dataframe(df)
            if st.button("✅ Save Uploaded Data"):
                values, labels = [], []
                for col in df.columns:
                    vals = df[col].dropna().values
                    values.extend(vals)
                    labels.extend([col] * len(vals))
                st.session_state.tukey_df = pd.DataFrame({'Group': labels, 'Value': values})
                st.session_state.data_loaded = True
                st.rerun()
        return None

# --------------------------------------------------------------
# Load Factorial data (Two-Way ANOVA with Blocking) - CORRECTED
# --------------------------------------------------------------
def load_factorial():
    if data_source == "✏️ Manual entry":
        st.subheader("Two-Way Factorial ANOVA with Blocking")
        with st.form(key="factorial_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                levels_a = st.number_input("Factor A levels", min_value=2, max_value=5, value=2)
            with col2:
                levels_b = st.number_input("Factor B levels", min_value=2, max_value=5, value=2)
            with col3:
                blocks = st.number_input("Number of blocks", min_value=2, max_value=5, value=2)

            names_a = []
            names_b = []
            names_block = []

            st.write("**Factor A (e.g., Cutting Speed) levels:**")
            cols_a = st.columns(int(levels_a))
            for i in range(int(levels_a)):
                with cols_a[i]:
                    name = st.text_input(f"A{i+1}", value=f"A{i+1}")
                    names_a.append(name)

            st.write("**Factor B (e.g., Coolant) levels:**")
            cols_b = st.columns(int(levels_b))
            for j in range(int(levels_b)):
                with cols_b[j]:
                    name = st.text_input(f"B{j+1}", value=f"B{j+1}")
                    names_b.append(name)

            st.write("**Block names:**")
            cols_block = st.columns(int(blocks))
            for k in range(int(blocks)):
                with cols_block[k]:
                    name = st.text_input(f"Block{k+1}", value=f"Block{k+1}")
                    names_block.append(name)

            st.write("**Response values for each block:**")
            rows = []
            for k in range(int(blocks)):
                st.write(f"**{names_block[k]}**")
                for i in range(int(levels_a)):
                    for j in range(int(levels_b)):
                        val = st.number_input(f"{names_a[i]} × {names_b[j]}", value=10.0, key=f"fact_{k}_{i}_{j}", step=0.1)
                        rows.append({
                            'Block': names_block[k],
                            'FactorA': names_a[i],
                            'FactorB': names_b[j],
                            'Response': val
                        })
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                df = pd.DataFrame(rows)
                st.session_state.factorial_df = df
                st.session_state.data_loaded = True
                st.rerun()
        return st.session_state.get('factorial_df', None)
    else:
        st.subheader("Upload Factorial Data")
        st.info("File should contain columns: Block, FactorA, FactorB, Response (names can be flexible, e.g. 'Cutting Speed', 'Coolant', 'Surface Roughness (µm)')")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="fact_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded, engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None

            st.write("Data preview (original column names):")
            st.dataframe(df.head())

            # تعيين الأعمدة الأصلية إلى الأسماء القياسية (original -> new)
            # نبني قاموساً يكون مفتاحه الاسم الأصلي وقيمته الاسم الجديد
            rename_dict = {}
            # Block
            if 'Block' in df.columns:
                rename_dict['Block'] = 'Block'
            elif 'block' in df.columns:
                rename_dict['block'] = 'Block'
            # Factor A (Cutting Speed)
            if 'Cutting Speed' in df.columns:
                rename_dict['Cutting Speed'] = 'FactorA'
            elif 'FactorA' in df.columns:
                rename_dict['FactorA'] = 'FactorA'
            elif 'Speed' in df.columns:
                rename_dict['Speed'] = 'FactorA'
            elif 'factorA' in df.columns:
                rename_dict['factorA'] = 'FactorA'
            # Factor B (Coolant)
            if 'Coolant' in df.columns:
                rename_dict['Coolant'] = 'FactorB'
            elif 'FactorB' in df.columns:
                rename_dict['FactorB'] = 'FactorB'
            elif 'factorB' in df.columns:
                rename_dict['factorB'] = 'FactorB'
            # Response
            if 'Surface Roughness (µm)' in df.columns:
                rename_dict['Surface Roughness (µm)'] = 'Response'
            elif 'Response' in df.columns:
                rename_dict['Response'] = 'Response'
            elif 'Roughness' in df.columns:
                rename_dict['Roughness'] = 'Response'
            elif 'response' in df.columns:
                rename_dict['response'] = 'Response'

            # التحقق من وجود الأعمدة الأربعة المطلوبة بعد إعادة التسمية
            df_renamed = df.rename(columns=rename_dict)
            required = ['Block', 'FactorA', 'FactorB', 'Response']
            missing = [col for col in required if col not in df_renamed.columns]
            if not missing:
                st.success("Columns automatically mapped!")
                st.write("Renamed data preview (first 5 rows):")
                st.dataframe(df_renamed[required].head())
                if st.button("✅ Save Uploaded Data"):
                    st.session_state.factorial_df = df_renamed[required].dropna()
                    st.session_state.data_loaded = True
                    st.rerun()
            else:
                st.error(f"Could not find required columns after mapping. Missing: {missing}. Found columns: {list(df_renamed.columns)}. Please ensure your file contains appropriate column names.")
        return None

# --------------------------------------------------------------
# Load data based on test type
# --------------------------------------------------------------
if test_type == "ANOVA (F-test) - RBD":
    load_rbd()
elif test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
    load_two_groups()
elif test_type == "Tukey HSD (Post-hoc)":
    load_tukey()
elif test_type == "Two-Way Factorial ANOVA with Blocking":
    load_factorial()

# --------------------------------------------------------------
# Analysis button
# --------------------------------------------------------------
if st.button("🔬 Run Analysis", type="primary"):
    if not st.session_state.data_loaded:
        st.error("Please load data first using the 'Save Data' button.")
        st.stop()

    if test_type == "ANOVA (F-test) - RBD":
        df = st.session_state.rbd_df
        model = ols('Response ~ C(Treatment) + C(Block)', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=3)
        st.markdown("---")
        st.header("📈 ANOVA Results (RBD)")
        st.dataframe(anova.round(4), use_container_width=True)
        treat_means = df.groupby('Treatment')['Response'].mean().round(3)
        block_means = df.groupby('Block')['Response'].mean().round(3)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Treatment Means**")
            st.dataframe(pd.DataFrame(treat_means))
        with col2:
            st.write("**Block Means**")
            st.dataframe(pd.DataFrame(block_means))
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        treat_means.plot(kind='bar', ax=ax[0], color='skyblue')
        ax[0].set_title('Treatment Means')
        block_means.plot(kind='bar', ax=ax[1], color='lightcoral')
        ax[1].set_title('Block Means')
        st.pyplot(fig)

    elif test_type == "Two-Sample t-test":
        df = st.session_state.twog_df
        groups = df['Group'].unique()
        g1 = df[df['Group']==groups[0]]['Value'].values
        g2 = df[df['Group']==groups[1]]['Value'].values
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        st.markdown("---")
        st.header("Two-Sample t-test Results (Welch)")
        st.write(f"Group1: n={len(g1)}, mean={np.mean(g1):.3f}")
        st.write(f"Group2: n={len(g2)}, mean={np.mean(g2):.3f}")
        st.write(f"t = {t_stat:.4f}, p = {p_val:.4f}")
        if p_val < alpha:
            st.error("Reject H0: Significant difference.")
        else:
            st.success("Fail to reject H0: No significant difference.")

    elif test_type == "Two-Sample Z-test":
        df = st.session_state.twog_df
        groups = df['Group'].unique()
        g1 = df[df['Group']==groups[0]]['Value'].values
        g2 = df[df['Group']==groups[1]]['Value'].values
        se = np.sqrt(np.var(g1,ddof=1)/len(g1) + np.var(g2,ddof=1)/len(g2))
        z_stat = (np.mean(g1)-np.mean(g2))/se
        p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
        st.markdown("---")
        st.header("Two-Sample Z-test Results")
        st.write(f"z = {z_stat:.4f}, p = {p_val:.4f}")
        if p_val < alpha:
            st.error("Reject H0: Significant difference.")
        else:
            st.success("Fail to reject H0: No significant difference.")

    elif test_type == "Tukey HSD (Post-hoc)":
        df = st.session_state.tukey_df
        tukey = pairwise_tukeyhsd(df['Value'], df['Group'], alpha=alpha)
        st.markdown("---")
        st.header("Tukey HSD Results")
        st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
        fig, ax = plt.subplots()
        tukey.plot_simultaneous(ax=ax)
        st.pyplot(fig)

    elif test_type == "Two-Way Factorial ANOVA with Blocking":
        df = st.session_state.factorial_df
        model = ols('Response ~ C(FactorA) + C(FactorB) + C(Block) + C(FactorA):C(FactorB)', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=3)
        st.markdown("---")
        st.header("📈 Two-Way Factorial ANOVA with Blocking")
        st.subheader("ANOVA Table (Type III Sum of Squares)")
        st.dataframe(anova.round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Factor A Means**")
            means_a = df.groupby('FactorA')['Response'].mean().round(3)
            st.dataframe(pd.DataFrame(means_a))
        with col2:
            st.write("**Factor B Means**")
            means_b = df.groupby('FactorB')['Response'].mean().round(3)
            st.dataframe(pd.DataFrame(means_b))

        st.subheader("Interaction Plot")
        fig, ax = plt.subplots()
        sns.pointplot(data=df, x='FactorA', y='Response', hue='FactorB', dodge=True,
                      markers=['o','s'], linestyle='-', errorbar='se', ax=ax)
        ax.set_title('Interaction between Factor A and Factor B')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        if len(df['FactorA'].unique()) > 2:
            with st.expander("Tukey HSD for Factor A"):
                tukey_a = pairwise_tukeyhsd(df['Response'], df['FactorA'], alpha=alpha)
                st.dataframe(pd.DataFrame(data=tukey_a.summary().data[1:], columns=tukey_a.summary().data[0]))
        if len(df['FactorB'].unique()) > 2:
            with st.expander("Tukey HSD for Factor B"):
                tukey_b = pairwise_tukeyhsd(df['Response'], df['FactorB'], alpha=alpha)
                st.dataframe(pd.DataFrame(data=tukey_b.summary().data[1:], columns=tukey_b.summary().data[0]))

    st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("Multi-Test Statistical Analysis Suite | Accurate Type III ANOVA | Factorial with Blocking | Flexible column mapping")
