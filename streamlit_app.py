import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="Linear vs Exponential Growth Explorer",
    layout="wide"
)

st.title("📈 Linear vs Exponential Growth Explorer")

st.markdown(
    """
This app lets you compare a **linear function** and an **exponential function**  where we interpret the growth/decay rate per period.

Use the sliders on the left to change the parameters and see how the curves behave over time.
"""
)

# ------------------------
# SIDEBAR CONTROLS
# ------------------------
st.sidebar.header("🔧 Parameters")

# X-range settings
time_max = st.sidebar.slider(
    "Time range (max x)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)

#num_points = st.sidebar.slider(
#    "Number of points (smoothness)",
#    min_value=50,
#    max_value=1000,
#    value=400,
#    step=50
#)

st.sidebar.markdown("---")

same_a = st.sidebar.checkbox(
    "Use same initial value a for both functions",
    value=True
)
same_rate = st.sidebar.checkbox(
    "Use same rate (linear increment vs exponential % growth) for both",
    value=True
)

st.sidebar.markdown("### Initial value a (at x = 0)")
a_min, a_max = -10_000.0, 10_000.0

if same_a:
    a_shared = st.sidebar.slider(
        "Initial value a for both",
        min_value=a_min,
        max_value=a_max,
        value=40_000.0,
        step=10.0
    )
    a_linear = a_shared
    a_exp = a_shared
else:
    a_linear = st.sidebar.slider(
        "a_linear (for linear function)",
        min_value=a_min,
        max_value=a_max,
        value=40_000.0,
        step=10.0
    )
    a_exp = st.sidebar.slider(
        "a_exp (for exponential function)",
        min_value=a_min,
        max_value=a_max,
        value=40_000.0,
        step=10.0
    )

st.sidebar.markdown("### Slope / Rate b")

# Linear: absolute increase per period
b_lin = st.sidebar.slider(
    "Linear increase per period (b_linear)",
    min_value=-10_000.0,
    max_value=10_000.0,
    value=2_000.0,
    step=10.0,
    help="This is the constant amount added (or subtracted) each period."
)

# Exponential: percentage growth/decay
if same_rate:
    rate_pct_shared = st.sidebar.slider(
        "Growth/decay rate for exponential (%)",
        min_value=-100.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        help="Interpreted as b_exp = 1 + rate/100."
    )
    rate_exp_pct = rate_pct_shared
else:
    rate_exp_pct = st.sidebar.slider(
        "Exponential growth/decay rate (%)",
        min_value=-100.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        help="Interpreted as b_exp = 1 + rate/100."
    )

b_exp = 1.0 + rate_exp_pct / 100.0

st.sidebar.markdown(
    f"**Derived exponential base:**  \n"
    f"b_exp = 1 + r = **{b_exp:.4f}**"
)

# ------------------------
# COMPUTE DATA
# ------------------------
x = np.linspace(0, time_max, int(400))

y_linear = a_linear + b_lin * x
y_exp = a_exp * (b_exp ** x)

data = pd.DataFrame({
    "x": x,
    "Linear": y_linear,
    "Exponential": y_exp
})

# Long format for Altair
data_long = data.melt(id_vars="x", var_name="Function", value_name="Value")

# ------------------------
# FIND CROSSOVER POINT (approximate)
# ------------------------
crossover_x = None
crossover_y = None

diff = y_exp - y_linear
sign_change_indices = np.where(np.diff(np.sign(diff)) != 0)[0]

if len(sign_change_indices) > 0:
    idx = sign_change_indices[0]
    # Linear interpolation between x[idx] and x[idx+1]
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = diff[idx], diff[idx + 1]
    if y1 != y0:
        crossover_x = x0 - y0 * (x1 - x0) / (y1 - y0)
        crossover_y = a_linear + b_lin * crossover_x  # same as a_exp * b_exp**crossover_x

# ------------------------
# LAYOUT: CHART + TABLE
# ------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Function Comparison")

    base_chart = alt.Chart(data_long).mark_line().encode(
        x=alt.X("x", title="x (time)"),
        y=alt.Y("Value", title="y"),
        color=alt.Color("Function", title="Function"),
        tooltip=["x", "Function", "Value"]
    ).properties(
        width="container",
        height=400
    )

    charts = [base_chart]

    # Add crossover point if it exists
    if crossover_x is not None and 0 <= crossover_x <= time_max:
        cross_df = pd.DataFrame({
            "x": [crossover_x],
            "Value": [crossover_y],
            "Function": ["Crossover point"]
        })

        cross_chart = alt.Chart(cross_df).mark_point(size=80).encode(
            x="x",
            y="Value",
            shape=alt.value("diamond"),
            tooltip=["x", "Value"]
        )

        charts.append(cross_chart)

        st.markdown(
            f"✅ **Crossover:** Exponential equals linear at approximately "
            f"**x ≈ {crossover_x:.2f}**, y ≈ **{crossover_y:,.0f}**."
        )
    else:
        st.markdown(
            "ℹ️ **No crossover in this range:** the exponential function "
            "does not cross the linear function between x = 0 and "
            f"x = {time_max:.0f}."
        )

    final_chart = alt.layer(*charts).interactive()
    st.altair_chart(final_chart, use_container_width=True)

with col2:
    st.subheader("📋 Table of Values (integer x)")
    max_int_x = int(time_max)
    x_int = np.arange(0, max_int_x + 1)
    y_lin_int = a_linear + b_lin * x_int
    y_exp_int = a_exp * (b_exp ** x_int)

    table_df = pd.DataFrame({
        "x": x_int,
        "Linear": np.round(y_lin_int, 2),
        "Exponential": np.round(y_exp_int, 2),
        "Difference (Exp - Lin)": np.round(y_exp_int - y_lin_int, 2)
    })

    st.dataframe(table_df, use_container_width=True)

