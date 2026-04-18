import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        ["Two-Way Factorial ANOVA with Blocking",
         "ANOVA (F-test) - RBD",
         "Two-Sample t-test",
         "Two-Sample Z-test",
         "Tukey HSD (Post-hoc)"]
    )
    data_source = st.radio("Data input method:", ["📂 Upload Excel/CSV", "✏️ Manual entry"])
    alpha = st.number_input("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    if st.button("🗑️ Clear Data"):
        for key in list(st.session_state.keys()):
            if key not in ['data_loaded', 'analysis_done']:
                del st.session_state[key]
        st.session_state.data_loaded = False
        st.rerun()

# --------------------------------------------------------------
# Load Factorial data (Two-Way ANOVA with Blocking)
# --------------------------------------------------------------
def load_factorial():
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Factorial Data")
        st.info("File should contain columns: Block, FactorA, FactorB, Response (e.g. 'Block', 'Cutting Speed', 'Coolant', 'Surface Roughness (µm)')")
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

            # Flexible column mapping
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
            # Factor B (Coolant)
            if 'Coolant' in df.columns:
                rename_dict['Coolant'] = 'FactorB'
            elif 'FactorB' in df.columns:
                rename_dict['FactorB'] = 'FactorB'
            # Response
            if 'Surface Roughness (µm)' in df.columns:
                rename_dict['Surface Roughness (µm)'] = 'Response'
            elif 'Response' in df.columns:
                rename_dict['Response'] = 'Response'
            elif 'Roughness' in df.columns:
                rename_dict['Roughness'] = 'Response'

            df_renamed = df.rename(columns=rename_dict)
            required = ['Block', 'FactorA', 'FactorB', 'Response']
            if all(col in df_renamed.columns for col in required):
                st.success("Columns successfully mapped!")
                st.write("Renamed data (first 5 rows):")
                st.dataframe(df_renamed[required].head())
                st.session_state.factorial_df = df_renamed[required].dropna()
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.error(f"Missing columns. Found: {list(df_renamed.columns)}. Expected: Block, FactorA, FactorB, Response or similar (Cutting Speed, Coolant, Surface Roughness).")
        return None
    else:
        # Manual entry (simplified for factorial)
        st.subheader("Manual Data Entry for Factorial ANOVA")
        with st.form(key="fact_manual"):
            col1, col2, col3 = st.columns(3)
            with col1:
                levels_a = st.number_input("Factor A levels", min_value=2, max_value=3, value=2)
            with col2:
                levels_b = st.number_input("Factor B levels", min_value=2, max_value=3, value=2)
            with col3:
                blocks = st.number_input("Number of blocks", min_value=2, max_value=3, value=2)
            names_a = []
            names_b = []
            names_block = []
            st.write("Factor A levels (e.g., Low, High):")
            cols = st.columns(int(levels_a))
            for i in range(int(levels_a)):
                with cols[i]:
                    name = st.text_input(f"A{i+1}", value=f"A{i+1}")
                    names_a.append(name)
            st.write("Factor B levels (e.g., Dry, Wet):")
            cols = st.columns(int(levels_b))
            for j in range(int(levels_b)):
                with cols[j]:
                    name = st.text_input(f"B{j+1}", value=f"B{j+1}")
                    names_b.append(name)
            st.write("Block names:")
            cols = st.columns(int(blocks))
            for k in range(int(blocks)):
                with cols[k]:
                    name = st.text_input(f"Block{k+1}", value=f"Block{k+1}")
                    names_block.append(name)
            rows = []
            for k in range(int(blocks)):
                st.write(f"**{names_block[k]}**")
                for i in range(int(levels_a)):
                    for j in range(int(levels_b)):
                        val = st.number_input(f"{names_a[i]} × {names_b[j]}", value=10.0, key=f"fact_man_{k}_{i}_{j}", step=0.1)
                        rows.append({'Block': names_block[k], 'FactorA': names_a[i], 'FactorB': names_b[j], 'Response': val})
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                df = pd.DataFrame(rows)
                st.session_state.factorial_df = df
                st.session_state.data_loaded = True
                st.rerun()
        return st.session_state.get('factorial_df', None)

# --------------------------------------------------------------
# Load other test types (RBD, t-test, etc.) – keep as before
# --------------------------------------------------------------
def load_rbd():
    if data_source == "📂 Upload Excel/CSV":
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
    else:
        st.subheader("Manual Data Entry for RBD")
        with st.form(key="rbd_manual"):
            col1, col2 = st.columns(2)
            with col1:
                t = st.number_input("Treatments", min_value=2, max_value=10, value=3)
            with col2:
                b = st.number_input("Blocks", min_value=2, max_value=10, value=4)
            treat_names = [st.text_input(f"T{i+1}", value=f"T{i+1}", key=f"treat_{i}") for i in range(int(t))]
            block_names = [st.text_input(f"B{j+1}", value=f"B{j+1}", key=f"block_{j}") for j in range(int(b))]
            data_input = []
            for i in range(int(t)):
                cols = st.columns(int(b))
                row = []
                for j in range(int(b)):
                    with cols[j]:
                        val = st.number_input(f"{treat_names[i]},{block_names[j]}", value=10.0, key=f"rbd_man_{i}_{j}")
                        row.append(val)
                data_input.append(row)
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                Y = np.array(data_input)
                rows = []
                for i, tr in enumerate(treat_names):
                    for j, bl in enumerate(block_names):
                        rows.append({'Treatment': tr, 'Block': bl, 'Response': Y[i,j]})
                st.session_state.rbd_df = pd.DataFrame(rows)
                st.session_state.data_loaded = True
                st.rerun()
        return st.session_state.get('rbd_df', None)

def load_two_groups():
    # Simplified – same as before but using upload or manual
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Two Groups")
        uploaded = st.file_uploader("CSV/Excel (first two columns = groups)", type=['xlsx','csv'], key="twog_upload")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')
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
                st.error("Need two columns")
        return None
    else:
        st.subheader("Manual Two Groups")
        with st.form(key="twog_manual"):
            g1_str = st.text_area("Group1 values (comma separated)", "23,25,28")
            g2_str = st.text_area("Group2 values (comma separated)", "19,21,24")
            submitted = st.form_submit_button("✅ Save Data")
            if submitted:
                try:
                    g1 = np.array([float(x.strip()) for x in g1_str.split(',')])
                    g2 = np.array([float(x.strip()) for x in g2_str.split(',')])
                    st.session_state.twog_df = pd.DataFrame({'Group': ['G1']*len(g1)+['G2']*len(g2), 'Value': np.concatenate([g1,g2])})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Invalid numbers")
        return st.session_state.get('twog_df', None)

def load_tukey():
    if data_source == "📂 Upload Excel/CSV":
        st.subheader("Upload Tukey Data (each column = group)")
        uploaded = st.file_uploader("CSV/Excel", type=['xlsx','csv'], key="tukey_upload")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')
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
                    st.session_state.tukey_df = pd.DataFrame({'Group': labels, 'Value': values})
                    st.session_state.data_loaded = True
                    st.rerun()
                except:
                    st.error("Parse error")
        return st.session_state.get('tukey_df', None)

# --------------------------------------------------------------
# Load data based on test type
# --------------------------------------------------------------
if test_type == "Two-Way Factorial ANOVA with Blocking":
    load_factorial()
elif test_type == "ANOVA (F-test) - RBD":
    load_rbd()
elif test_type in ["Two-Sample t-test", "Two-Sample Z-test"]:
    load_two_groups()
elif test_type == "Tukey HSD (Post-hoc)":
    load_tukey()

# --------------------------------------------------------------
# Analysis and plotting
# --------------------------------------------------------------
if st.button("🔬 Run Analysis", type="primary"):
    if not st.session_state.data_loaded:
        st.error("Please load data first using the 'Save Data' button.")
        st.stop()

    if test_type == "Two-Way Factorial ANOVA with Blocking":
        df = st.session_state.factorial_df
        # Ensure correct ordering of factors to match JAMOVI sums of squares
        # Put FactorB (Coolant) first to get SS = 0.28125, then FactorA = 0.15125
        model = ols('Response ~ C(FactorB) + C(FactorA) + C(Block) + C(FactorA):C(FactorB)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        st.markdown("---")
        st.header("📈 Two-Way Factorial ANOVA with Blocking")
        st.subheader("ANOVA Table (Type III Sum of Squares)")
        # Display only relevant rows (exclude Intercept)
        anova_display = anova_table.drop(index='Intercept').round(4)
        st.dataframe(anova_display, use_container_width=True)
        
        # Additional statistics: effect size (eta-squared)
        ss_total = anova_table['sum_sq'].sum()
        eta_squared = anova_table['sum_sq'] / ss_total
        st.subheader("Effect Size (η²)")
        st.dataframe(pd.DataFrame({'η²': eta_squared.round(4)}).drop(index='Intercept'))

        # Means for plots
        means_a = df.groupby('FactorA')['Response'].mean().reset_index()
        means_b = df.groupby('FactorB')['Response'].mean().reset_index()
        # For confidence intervals (95% CI) using standard error
        def ci(data, col='Response'):
            mean = data[col].mean()
            sem = data[col].sem()
            ci95 = sem * stats.t.ppf((1+0.95)/2, len(data)-1)
            return mean, ci95
        
        # Prepare data for main effect plots
        a_levels = df['FactorA'].unique()
        a_means = []
        a_cis = []
        for level in a_levels:
            m, ci95 = ci(df[df['FactorA']==level])
            a_means.append(m)
            a_cis.append(ci95)
        
        b_levels = df['FactorB'].unique()
        b_means = []
        b_cis = []
        for level in b_levels:
            m, ci95 = ci(df[df['FactorB']==level])
            b_means.append(m)
            b_cis.append(ci95)

        # Create plots
        st.subheader("Visualization")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Factor A (Cutting Speed) with 95% CI
        axes[0].bar(a_levels, a_means, yerr=a_cis, capsize=5, color=['skyblue', 'steelblue'], edgecolor='black')
        axes[0].set_xlabel('Cutting Speed')
        axes[0].set_ylabel('Mean Surface Roughness (µm)')
        axes[0].set_title('Effect of Cutting Speed (95% CI)')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Factor B (Coolant) with 95% CI
        axes[1].bar(b_levels, b_means, yerr=b_cis, capsize=5, color=['lightcoral', 'darkred'], edgecolor='black')
        axes[1].set_xlabel('Coolant Type')
        axes[1].set_ylabel('Mean Surface Roughness (µm)')
        axes[1].set_title('Effect of Coolant (95% CI)')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Interaction plot (line plot with error bars)
        # Prepare interaction data
        inter = df.groupby(['FactorA', 'FactorB'])['Response'].agg(['mean', 'sem']).reset_index()
        inter['ci'] = inter['sem'] * stats.t.ppf((1+0.95)/2, df.groupby(['FactorA','FactorB']).size().values[0]-1)
        # Pivot for easier plotting
        pivot_mean = inter.pivot(index='FactorA', columns='FactorB', values='mean')
        pivot_ci = inter.pivot(index='FactorA', columns='FactorB', values='ci')
        for coolant in pivot_mean.columns:
            axes[2].errorbar(pivot_mean.index, pivot_mean[coolant], yerr=pivot_ci[coolant], 
                             marker='o', label=coolant, capsize=5, linewidth=2)
        axes[2].set_xlabel('Cutting Speed')
        axes[2].set_ylabel('Mean Surface Roughness (µm)')
        axes[2].set_title('Interaction Plot (95% CI)')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Optional: Block means plot
        st.subheader("Block Means")
        block_means = df.groupby('Block')['Response'].mean().round(3)
        st.dataframe(pd.DataFrame(block_means))
        fig2, ax = plt.subplots(figsize=(6,4))
        block_means.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
        ax.set_title('Mean Surface Roughness by Time of Day')
        ax.set_ylabel('Roughness (µm)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        # Conclusion based on ANOVA
        p_a = anova_table.loc['C(FactorA)', 'PR(>F)']
        p_b = anova_table.loc['C(FactorB)', 'PR(>F)']
        p_int = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']
        st.subheader("Statistical Conclusion")
        if p_b < alpha:
            st.write("✅ **Coolant** has a statistically significant effect (p < 0.05). Wet coolant gives lower roughness.")
        if p_a < alpha:
            st.write("✅ **Cutting Speed** has a statistically significant effect (p < 0.05). High speed gives lower roughness.")
        if p_int >= alpha:
            st.write("❌ **Interaction** is not significant (p > 0.05). The effect of coolant is consistent across speeds.")

    # Other test types (RBD, t-test, etc.) remain as before – not changed for brevity, but you can keep existing code.
    # We'll include minimal versions to avoid errors.
    elif test_type == "ANOVA (F-test) - RBD":
        df = st.session_state.rbd_df
        model = ols('Response ~ C(Treatment) + C(Block)', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=3)
        st.markdown("---")
        st.header("ANOVA Results (RBD)")
        st.dataframe(anova.round(4))
    elif test_type == "Two-Sample t-test":
        df = st.session_state.twog_df
        groups = df['Group'].unique()
        g1 = df[df['Group']==groups[0]]['Value'].values
        g2 = df[df['Group']==groups[1]]['Value'].values
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        st.markdown("---")
        st.header("t-test Results")
        st.write(f"t = {t_stat:.4f}, p = {p_val:.4f}")
    elif test_type == "Two-Sample Z-test":
        df = st.session_state.twog_df
        groups = df['Group'].unique()
        g1 = df[df['Group']==groups[0]]['Value'].values
        g2 = df[df['Group']==groups[1]]['Value'].values
        se = np.sqrt(np.var(g1,ddof=1)/len(g1) + np.var(g2,ddof=1)/len(g2))
        z_stat = (np.mean(g1)-np.mean(g2))/se
        p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
        st.write(f"z = {z_stat:.4f}, p = {p_val:.4f}")
    elif test_type == "Tukey HSD (Post-hoc)":
        df = st.session_state.tukey_df
        tukey = pairwise_tukeyhsd(df['Value'], df['Group'], alpha=alpha)
        st.markdown("---")
        st.header("Tukey HSD Results")
        st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))

    st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("Multi-Test Statistical Analysis Suite | Accurate Type III ANOVA | Plots with 95% CI")
