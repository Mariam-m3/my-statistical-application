import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.set_page_config(page_title="Factorial ANOVA with Blocking", layout="wide")
st.title("📊 Two-Way Factorial ANOVA with Blocking")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx'])
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, step=0.01)

if uploaded_file is not None:
    # Read the first sheet
    df_raw = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    # Flexible column renaming
    rename_dict = {}
    for col in df_raw.columns:
        col_lower = col.lower()
        if 'block' in col_lower:
            rename_dict[col] = 'Block'
        elif 'cutting' in col_lower or (col_lower == 'speed') or 'factor a' in col_lower:
            rename_dict[col] = 'FactorA'
        elif 'coolant' in col_lower or 'factor b' in col_lower:
            rename_dict[col] = 'FactorB'
        elif 'roughness' in col_lower or 'surface' in col_lower or 'response' in col_lower:
            rename_dict[col] = 'Response'
    df = df_raw.rename(columns=rename_dict)

    required = ['Block', 'FactorA', 'FactorB', 'Response']
    if not all(col in df.columns for col in required):
        st.error(f"Missing columns. Found: {list(df.columns)}. Required: Block, FactorA, FactorB, Response (or similar names).")
        st.stop()

    df = df[required].dropna()

    # ANOVA Type II (same as JAMOVI for balanced design)
    formula = 'Response ~ C(FactorB) + C(FactorA) + C(Block) + C(FactorA):C(FactorB)'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Compute total sum of squares for eta squared
    grand_mean = df['Response'].mean()
    ss_total = np.sum((df['Response'] - grand_mean)**2)

    # Build results table
    results = []
    factors = ['C(FactorB)', 'C(FactorA)', 'C(Block)', 'C(FactorA):C(FactorB)']
    names = ['Coolant', 'Cutting Speed', 'Block', 'Coolant × Cutting Speed']
    for fact, name in zip(factors, names):
        ss = anova_table.loc[fact, 'sum_sq']
        df_f = anova_table.loc[fact, 'df']
        ms = ss / df_f
        f_val = anova_table.loc[fact, 'F']
        p_val = anova_table.loc[fact, 'PR(>F)']
        eta = ss / ss_total
        results.append([name, ss, df_f, ms, f_val, p_val, eta])
    # Residuals
    ss_res = anova_table.loc['Residual', 'sum_sq']
    df_res = anova_table.loc['Residual', 'df']
    ms_res = ss_res / df_res
    results.append(['Residuals', ss_res, df_res, ms_res, '', '', ''])

    anova_df = pd.DataFrame(results, columns=['Factor', 'SS', 'df', 'MS', 'F', 'p-value', 'η²'])
    # Round numeric columns
    for col in ['SS', 'MS', 'F', 'η²']:
        anova_df[col] = anova_df[col].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)
    anova_df['p-value'] = anova_df['p-value'].apply(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)

    st.subheader("ANOVA Table (Type II)")
    st.dataframe(anova_df, use_container_width=True)

    # -------------------------------
    # Plots with 95% Confidence Interval
    # -------------------------------
    st.subheader("Visualization")

    def mean_ci(data):
        """Return mean and half-width of 95% confidence interval."""
        mean = np.mean(data)
        n = len(data)
        sem = stats.sem(data) if n > 1 else 0
        # Use t-distribution for small samples
        if n > 1:
            t_crit = stats.t.ppf(0.975, n-1)
            ci_half = sem * t_crit
        else:
            ci_half = 0
        return mean, ci_half

    # Factor A (Cutting Speed)
    levels_a = sorted(df['FactorA'].unique())
    means_a, err_a = [], []
    for level in levels_a:
        vals = df[df['FactorA']==level]['Response'].values
        m, ci = mean_ci(vals)
        means_a.append(m)
        err_a.append(ci)

    # Factor B (Coolant)
    levels_b = sorted(df['FactorB'].unique())
    means_b, err_b = [], []
    for level in levels_b:
        vals = df[df['FactorB']==level]['Response'].values
        m, ci = mean_ci(vals)
        means_b.append(m)
        err_b.append(ci)

    # Interaction data
    inter = df.groupby(['FactorA', 'FactorB'])['Response'].agg(list).reset_index()
    inter_means, inter_err = [], []
    for _, row in inter.iterrows():
        m, ci = mean_ci(row['Response'])
        inter_means.append(m)
        inter_err.append(ci)
    inter['mean'] = inter_means
    inter['err'] = inter_err

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Cutting Speed
    axes[0].bar(levels_a, means_a, yerr=err_a, capsize=5, color=['skyblue', 'steelblue'], edgecolor='black')
    axes[0].set_ylabel('Mean Surface Roughness (µm)')
    axes[0].set_title('Effect of Cutting Speed (95% CI)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Coolant
    axes[1].bar(levels_b, means_b, yerr=err_b, capsize=5, color=['lightcoral', 'darkred'], edgecolor='black')
    axes[1].set_ylabel('Mean Surface Roughness (µm)')
    axes[1].set_title('Effect of Coolant (95% CI)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 3: Interaction
    for coolant in levels_b:
        sub = inter[inter['FactorB']==coolant]
        axes[2].errorbar(sub['FactorA'], sub['mean'], yerr=sub['err'],
                         marker='o', label=coolant, capsize=5, linewidth=2)
    axes[2].set_xlabel('Cutting Speed')
    axes[2].set_ylabel('Mean Surface Roughness (µm)')
    axes[2].set_title('Interaction Plot (95% CI)')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

    # Block means
    block_means = df.groupby('Block')['Response'].mean().round(3)
    st.subheader("Block Means")
    st.dataframe(pd.DataFrame(block_means))

    fig2, ax = plt.subplots(figsize=(6,4))
    block_means.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
    ax.set_title('Mean Surface Roughness by Time of Day')
    ax.set_ylabel('Roughness (µm)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # Interpretation
    p_coolant = anova_table.loc['C(FactorB)', 'PR(>F)']
    p_speed = anova_table.loc['C(FactorA)', 'PR(>F)']
    p_block = anova_table.loc['C(Block)', 'PR(>F)']
    p_int = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']

    st.subheader("Statistical Conclusion")
    if p_coolant < alpha:
        st.write(f"✅ **Coolant** has a significant effect (p = {p_coolant:.4f}) – Wet coolant gives lower roughness.")
    if p_speed < alpha:
        st.write(f"✅ **Cutting Speed** has a significant effect (p = {p_speed:.4f}) – High speed gives lower roughness.")
    if p_block < alpha:
        st.write(f"⚠️ **Time of Day (Block)** has a significant effect (p = {p_block:.4f}) – Afternoon measurements are slightly higher.")
    if p_int >= alpha:
        st.write(f"❌ **Interaction** is not significant (p = {p_int:.4f}) – The effect of coolant is consistent across speeds.")

    st.success("✅ Analysis complete!")

else:
    st.info("Please upload an Excel file containing columns like: Block, Cutting Speed, Coolant, Surface Roughness (µm)")
