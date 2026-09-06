"""
HMM Model Selection Page — Cross-validate state count and emission model choice.

Run this page BEFORE 6_HMM_Analysis to decide how many states to use and
which emission model best fits your data.
"""


import matplotlib
import numpy as np
import streamlit as st

from ui.guards import require_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dam_utilities
from dataset_meta import dataset_fingerprint
from hmm_models import HMMConfig, compare_n_states

st.markdown(
    "Use k-fold cross-validation to select the optimal **number of states** and "
    "**emission model** before committing to a full HMM run. "
    "Results guide parameter choices on the **HMM Analysis** page."
)

master = require_dataset()

# --- Phase selection: which lighting paradigm to CROSS-VALIDATE on. ---
# This page used to read st.session_state.dataset_LD if it happened to exist and
# otherwise fall through to the full LD+DD master, with no phase control at all.
# That let cross-validation select a model on one data source while HMM Analysis
# fitted the production model on another — and with no split applied it silently
# mixed paradigms. The phase is now explicit and sourced from a select_phase()
# view of the master, mirroring hmm_analysis.py.
#
# Only LD and DD are offered here. Analysis also has "Both (together)" and
# "Both (separate)"; cross-validating those would mean picking one k for a model
# that is fitted per paradigm, which is not a question this page can answer.
_has_transition = ("first_DD_day" in master.coords) or ("split_minute" in master.coords)
if _has_transition:
    cv_phase = st.radio(
        "Data phase for cross-validation",
        ["LD", "DD"],
        index=0,
        horizontal=True,
        key="cv_phase_choice",
        help="LD (entrained; the standard sleep reference) or DD (constant "
        "darkness). Pick the same phase you intend to fit on the HMM Analysis "
        "page — see the note below.",
    )
    ds, _phase_used = dam_utilities.select_phase(master, cv_phase)
else:
    # Mirror hmm_analysis.py's else branch: no boundary to split on, so say so
    # rather than silently cross-validating on a mix the user did not choose.
    cv_phase = "Full"
    ds = master
    st.caption("No LD/DD transition in this dataset — cross-validating on the full recording.")

# The two pages keep separate widget keys for states/emissions/transitions, so the
# winning configuration is retyped by hand — and nothing makes the phases agree
# either. A model selected on LD does not describe a DD fit.
st.info(
    f"Cross-validating on **{cv_phase}**. Select the **same phase** on the "
    "**HMM Analysis** page, or the model chosen here will not describe the data "
    "actually being fitted. Analysis additionally offers *Both (together)* and "
    "*Both (separate)*, which this page does not — match the LD or DD case."
)


@st.cache_data(show_spinner=False)
def _usable_fly_count(fp, _ds):
    """Flies with at least one finite observation minute, and the mean number of
    them per fly.

    NOT ``len(ds['id'])``. ``select_phase`` returns a NaN-masked view over the
    FULL time axis rather than a physical slice, so the id dimension is identical
    for LD and DD — but ``extract_observations`` keeps only finite minutes
    (``hmm_models._transform_observation``), so a fly whose record ends before the
    LD/DD boundary contributes nothing to one phase while still being counted in
    ``sizes['id']``. On example_data that is 189 flies in LD and 178 in DD.
    Reporting the id count would tell the user the phase picker did nothing.

    Counted on ``activity`` (or ``moving``); the two masks agree, since
    select_phase masks every ``(id, time)`` var on the same boundary.
    """
    var = "activity" if "activity" in _ds.data_vars else "moving"
    da = _ds[var]
    finite_per_fly = np.isfinite(np.asarray(da.values)).sum(axis=da.dims.index("time"))
    usable = int((finite_per_fly > 0).sum())
    mean_minutes = float(finite_per_fly[finite_per_fly > 0].mean()) if usable else 0.0
    return usable, mean_minutes


n_flies, _mean_minutes = _usable_fly_count(dataset_fingerprint(ds), ds)

if n_flies < 2:
    st.error(
        f"Only {n_flies} fly/flies have usable {cv_phase} data — cross-validation "
        "needs at least 2. Pick the other phase."
    )
    st.stop()

_n_ids = len(ds["id"].values)
if n_flies < _n_ids:
    st.caption(
        f"**{n_flies}** of {_n_ids} flies have usable **{cv_phase}** data "
        f"(~{_mean_minutes / 1440:.1f} days each). The rest have no minutes in "
        "this phase and are dropped by the observation extractor."
    )

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
        key="cv_n_folds",
    )
    # The hold-out size depends on the fold count the user actually picked, so it
    # cannot be computed inside help= — that string is built before the widget
    # returns and would be stuck describing the default (min(5, n_flies)) forever.
    # Rendering it as a caption after the widget lets it track the slider.
    st.caption(
        f"{n_flies} usable flies. Each fold holds out ~{n_flies // n_folds} of them."
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
