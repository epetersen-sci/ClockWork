"""
HMM Model Selection Page — Cross-validate state count and emission model choice.

Run this page BEFORE 6_HMM_Analysis to decide how many states to use and
which emission model best fits your data.
"""

import os
import sys

import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add core/ (analysis modules) and app/ to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hmm_models import HMMConfig, compare_n_states

st.header("HMM Model Selection")
st.markdown(
    "Use k-fold cross-validation to select the optimal **number of states** and "
    "**emission model** before committing to a full HMM run. "
    "Results guide parameter choices on the **HMM Analysis** page."
)

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset

# Prefer LD dataset for HMM model selection
if st.session_state.get("dataset_LD") is not None:
    ds = st.session_state.dataset_LD
    st.caption("Using **LD dataset** for HMM model selection.")

n_flies = len(ds["id"].values)

# ============================================================
# Configuration
# ============================================================
st.subheader("Cross-Validation Setup")

col1, col2 = st.columns(2)

with col1:
    n_folds = st.slider(
        "Number of folds",
        min_value=2,
        max_value=min(10, n_flies),
        value=min(5, n_flies),
        help=f"Dataset has {n_flies} flies. Each fold holds out ~{n_flies // min(5, n_flies)} flies.",
        key="cv_n_folds",
    )

    states_min = st.number_input(
        "Min states", min_value=2, max_value=8, value=2, key="cv_states_min"
    )
    states_max = st.number_input(
        "Max states", min_value=2, max_value=10, value=5, key="cv_states_max"
    )

with col2:
    emission_options = st.multiselect(
        "Emission models to test",
        options=["zip", "gaussian", "binary"],
        default=["zip"],
        help=(
            "**zip**: Zero-Inflated Poisson — recommended for raw DAM count data.\n\n"
            "**gaussian**: Normalized activity (Harbison style).\n\n"
            "**binary**: Moving/not-moving (Wiggin style).\n\n"
            "Log-likelihoods across emission models are NOT directly comparable "
            "(different observation spaces). Use **Agreement %** for cross-model comparison."
        ),
        key="cv_emissions",
    )

    cv_restarts = st.slider(
        "Restarts per fold fit",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Fewer restarts than production is fine for model selection.",
        key="cv_restarts",
    )

with st.expander("Advanced options"):
    cv_n_iter = st.slider("Max EM iterations per fold", 50, 200, 100, step=25, key="cv_n_iter")
    transition_constraints = st.selectbox(
        "Transition constraints",
        ["soft", "hard", "none"],
        key="cv_trans",
        help="Applied to all models tested.",
    )

if states_max < states_min:
    st.error("Max states must be >= min states.")
    st.stop()

if not emission_options:
    st.warning("Select at least one emission model.")
    st.stop()

n_states_range = list(range(int(states_min), int(states_max) + 1))
n_fits = len(emission_options) * len(n_states_range) * n_folds
st.info(
    f"**{n_fits} total fits** "
    f"({len(emission_options)} model(s) × {len(n_states_range)} state count(s) × {n_folds} folds) "
    f"+ {len(emission_options) * len(n_states_range)} full-data fits for AIC/BIC."
)

# ============================================================
# Run CV
# ============================================================
if st.button("Run Model Selection", key="run_cv"):
    base_config = HMMConfig.improved()
    base_config.transition_constraints = transition_constraints
    base_config.n_restarts = cv_restarts

    cv_progress = st.progress(0, text="Fitting models...")

    def _cv_progress(completed, total):
        cv_progress.progress(
            completed / total,
            text=f"Cross-validation: {completed}/{total} fold fits",
        )

    with st.spinner("Running cross-validation..."):
        try:
            fold_df, summary_df = compare_n_states(
                ds,
                n_states_range=n_states_range,
                emission_models=emission_options,
                n_folds=n_folds,
                base_config=base_config,
                cv_restarts=cv_restarts,
                cv_n_iter=cv_n_iter,
                progress_callback=_cv_progress,
            )
            st.session_state["cv_fold_df"] = fold_df
            st.session_state["cv_summary_df"] = summary_df
            st.success("Model selection complete.")
        except Exception as e:
            st.error(f"Model selection error: {e}")
            st.exception(e)
        finally:
            cv_progress.empty()

# ============================================================
# Results
# ============================================================
fold_df = st.session_state.get("cv_fold_df")
summary_df = st.session_state.get("cv_summary_df")

if summary_df is None or len(summary_df) == 0:
    st.stop()

st.divider()
st.subheader("Results")

has_agreement = "mean_agreement_pct" in summary_df.columns

# ---- Summary table ----
st.markdown("#### Summary table")

display_cols = ["emission_model", "n_states", "mean_ll", "sem_ll", "aic", "bic"]
if has_agreement:
    display_cols.append("mean_agreement_pct")

display_df = summary_df[display_cols].copy()
display_df.columns = ["Emission", "N States", "Mean CV LL/obs", "SEM", "AIC", "BIC"] + (
    ["Mean Agree %"] if has_agreement else []
)

st.dataframe(
    display_df.style.format(
        {
            "Mean CV LL/obs": "{:.4f}",
            "SEM": "{:.4f}",
            "AIC": "{:.1f}",
            "BIC": "{:.1f}",
            **({"Mean Agree %": "{:.1f}%"} if has_agreement else {}),
        }
    ),
    width="stretch",
)

csv_summary = summary_df.to_csv(index=False)
st.download_button(
    "Download Summary CSV",
    csv_summary,
    "hmm_model_selection_summary.csv",
    "text/csv",
    key="dl_cv_summary",
)

# ---- Held-out LL plot ----
st.markdown("#### Held-out log-likelihood per observation")
st.caption(
    "Higher = better generalization. Compare within an emission model only. "
    "The elbow or plateau indicates the optimal state count."
)

emission_list = summary_df["emission_model"].unique().tolist()
colors = plt.cm.tab10(np.linspace(0, 0.5, len(emission_list)))

fig, ax = plt.subplots(figsize=(7, 4))

for em, color in zip(emission_list, colors):
    sub = summary_df[summary_df["emission_model"] == em].sort_values("n_states")
    ax.errorbar(
        sub["n_states"],
        sub["mean_ll"],
        yerr=sub["sem_ll"],
        marker="o",
        label=em,
        color=color,
        capsize=4,
        linewidth=2,
    )

ax.set_xlabel("Number of states")
ax.set_ylabel("Mean held-out LL / observation")
ax.legend(title="Emission model")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- AIC / BIC plots (within-model) ----
if len(emission_list) > 0:
    st.markdown("#### AIC and BIC (full-data fit, within emission model)")
    st.caption(
        "Lower AIC/BIC = better model. "
        "AIC/BIC penalize complexity so they can differ from the CV result when "
        "the training set is small or the model is near-flat."
    )

    n_em = len(emission_list)
    fig2, axes = plt.subplots(1, n_em, figsize=(5 * n_em, 4), squeeze=False)

    for ax2, em in zip(axes[0], emission_list):
        sub = summary_df[summary_df["emission_model"] == em].sort_values("n_states")
        ax2.plot(sub["n_states"], sub["aic"], marker="s", label="AIC", color="tab:blue")
        ax2.plot(sub["n_states"], sub["bic"], marker="^", label="BIC", color="tab:orange")
        best_aic = sub.loc[sub["aic"].idxmin(), "n_states"]
        best_bic = sub.loc[sub["bic"].idxmin(), "n_states"]
        ax2.axvline(
            best_aic, color="tab:blue", linestyle="--", alpha=0.4, label=f"Best AIC: {best_aic}"
        )
        ax2.axvline(
            best_bic, color="tab:orange", linestyle="--", alpha=0.4, label=f"Best BIC: {best_bic}"
        )
        ax2.set_title(f"{em}")
        ax2.set_xlabel("Number of states")
        ax2.set_ylabel("AIC / BIC")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ---- Agreement % plot (cross-model) ----
if has_agreement:
    st.markdown("#### Agreement with threshold sleep (cross-model comparison)")
    st.caption(
        "Agreement % measures how often the HMM's sleep/wake assignment matches "
        "the traditional 5-minute immobility threshold. "
        "This is the only metric that is directly comparable across emission models."
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    for em, color in zip(emission_list, colors):
        sub = summary_df[summary_df["emission_model"] == em].sort_values("n_states")
        if "mean_agreement_pct" in sub.columns and sub["mean_agreement_pct"].notna().any():
            ax3.plot(
                sub["n_states"],
                sub["mean_agreement_pct"],
                marker="o",
                label=em,
                color=color,
                linewidth=2,
            )

    ax3.set_xlabel("Number of states")
    ax3.set_ylabel("Mean agreement with threshold sleep (%)")
    ax3.legend(title="Emission model")
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ---- Per-fold detail ----
if fold_df is not None and len(fold_df) > 0:
    with st.expander("Per-fold detail"):
        st.dataframe(fold_df, width="stretch")
        csv_folds = fold_df.to_csv(index=False)
        st.download_button(
            "Download Per-fold CSV",
            csv_folds,
            "hmm_cv_folds.csv",
            "text/csv",
            key="dl_cv_folds",
        )

# ---- Interpretation guide ----
st.divider()
with st.expander("Interpretation guide"):
    st.markdown("""
**Choosing state count within an emission model:**
- Plot the held-out LL curve — look for an elbow or plateau.
- Confirm with AIC/BIC (lower = better). BIC penalizes complexity more heavily.
- A model with 1–2 more states than the elbow rarely adds interpretable biology.

**Comparing emission models:**
- You cannot directly compare held-out LL across ZIP, Gaussian, and binary models —
  the observation spaces are fundamentally different.
- Use **Agreement %** as the common benchmark: which emission model best recovers
  the traditional threshold-based sleep classification?
- Then inspect state parameters on the **HMM Analysis** page to check biological
  interpretability (do the states map sensibly to deep sleep, light sleep, quiet wake,
  active wake?).

**Typical findings in Drosophila DAM data:**
- ZIP emissions with 3–5 states often outperform binary (more granular) but the
  optimal count is dataset-dependent.
- BIC tends to prefer fewer states than AIC. The CV held-out LL is the most
  principled estimate of generalization.
    """)
