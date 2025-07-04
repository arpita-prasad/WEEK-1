import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt
import pickle

st.set_page_config(page_title="HydroSense", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
       background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #4dd0e1 100%);
        min-height: 100vh;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .metric-card {
        background: #f5fafd;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 1.2em 1em;
        margin-bottom: 1em;
        text-align: center;
        border: 2px solid #b2ebf2;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-weight: 900; font-size: 4.2rem; color: #0077b6;">
            HydroSense
        </h1>
        <h4 style="font-weight: 800; font-size: 1.3rem; margin-top: 0; color: #555;">
            Know Your Water.
        </h4>
        <p style="font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 3rem">
            A smart app to predict and monitor water pollutant levels for safer water quality.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Pollutants and safe limits
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
safe_limits = {
    'O2': 5.0,
    'NO3': 10.0,
    'NO2': 0.1,
    'SO4': 250.0,
    'PO4': 0.1,
    'CL': 250.0
}

# Inputs
st.markdown(
    "<h3 style='text-align: center; margin-bottom: 1.5rem;'>Enter Input Details</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    year_input = st.number_input(
        "🗓️ Enter Prediction Year", 
        min_value=2000, 
        max_value=2100, 
        value=2022, 
        help="Enter a year between 2000 and 2100."
    )
    st.caption("The year for which you want to forecast water pollutant levels.")

with col2:
    station_id = st.selectbox(
        "🛰️ Monitoring Station", 
        options=[str(i) for i in range(1, 23)], 
        help="Choose from 22 available monitoring stations."
    )
    st.caption("The location ID of the water quality monitoring station.")

# Predict button    
st.markdown("<br>", unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns([1, 0.2, 1])
with button_col2:
    predict_clicked = st.button("Predict", key="predict")

# Output
if predict_clicked:
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    predicted_pollutants = model.predict(input_encoded)[0]

    st.markdown(
        f"<h3 style='text-align: center;'>Predicted pollutant levels for Station ID <code>{station_id}</code> in <strong>{year_input}</strong></h3>",
        unsafe_allow_html=True
    )

    # Metrics
    metric_cols = st.columns(3)
    for i, (p, val) in enumerate(zip(pollutants, predicted_pollutants)):
        col = metric_cols[i % 3]
        safe_limit = safe_limits[p]
        if safe_limit == 0:
            delta_str = "N/A"
            arrow = ""
            delta_color = "black"
        else:
            delta = ((val - safe_limit) / safe_limit) * 100
            delta_str = f"{abs(delta):.1f}%"
            if delta < 0:
                arrow = "▼"
                delta_color = "#ea3943"  # red
            else:
                arrow = "▲"
                delta_color = "#16c784"  # green

        metric_html = f"""
        <div class="metric-card">
            <h4 style="color:#0077b6; margin-bottom: 0.3em;">{p}</h4>
            <p style="font-size:1.3em; margin:0; font-weight:bold;">{val:.2f} mg/L</p>
            <div style="margin-top:0.5em;">
                <span style="font-size:1.1em; color:{delta_color}; font-weight:600;">
                    {arrow} {delta_str if delta_str != "N/A" else delta_str}
                </span>
            </div>
        </div>
        """
        col.markdown(metric_html, unsafe_allow_html=True)

    # Bar Chart
    st.markdown("### Pollutant Levels: Predicted vs Safe")
    chart_df = pd.DataFrame({
        'Pollutant': pollutants,
        'Predicted': predicted_pollutants,
        'Safe Limit': [safe_limits[p] for p in pollutants]
    })

    # Melt for altair
    chart_df_long = chart_df.melt(id_vars='Pollutant', var_name='Type', value_name='Value')

    # Main bar chart
    bar = alt.Chart(chart_df_long).mark_bar(
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8
    ).encode(
        x=alt.X('Pollutant:N', title="Pollutant", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Value:Q', title="Concentration (mg/L)", scale=alt.Scale(zero=True)),
        color=alt.Color('Type:N',
            scale=alt.Scale(domain=['Predicted', 'Safe Limit'],
                            range=['#0077b6', '#90be6d']),
            legend=alt.Legend(title="Legend")
        ),
        tooltip=[
            alt.Tooltip('Pollutant:N', title='Pollutant'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Value:Q', title='Value (mg/L)', format='.2f')
        ]
    )

    # Text labels only for bars above threshold
    label_threshold = 10
    text = alt.Chart(chart_df_long).mark_text(
        align='center',
        baseline='bottom',
        dy=-4,
        fontSize=13,
        fontWeight='bold'
    ).encode(
        x='Pollutant:N',
        y='Value:Q',
        detail='Type:N',
        text=alt.condition(
            alt.datum.Value >= label_threshold,
            alt.Text('Value:Q', format='.2f'),
            alt.value('')
        ),
        color=alt.value('#222')
    )

    final_chart = (bar + text).properties(width="container", height=400)

    chart_center = st.columns([0.1, 0.8, 0.1])
    with chart_center[1]:
        st.altair_chart(final_chart, theme="streamlit", use_container_width=True)

    # Information to the user
    invisible_pollutants = []
    for i, (p, pred, safe) in enumerate(zip(pollutants, predicted_pollutants, [safe_limits[p] for p in pollutants])):
        if pred < label_threshold and safe < label_threshold:
            invisible_pollutants.append(p)

    if invisible_pollutants:
        st.markdown(
            "<b>Note:</b> The following pollutants have predicted and safe limit values that are very small compared to the scale of the chart, "
            "so their bars may not be visible or may appear as a thin line. "
            "This usually happens when concentrations are close to zero or much lower than other pollutants.<br>"
            "Please refer to the numeric predictions above for exact values.<br>"
            "<b>Pollutants affected:</b> " +
            ", ".join([f"<b>{p}</b>" for p in invisible_pollutants]) +
            ".", unsafe_allow_html=True
        )

st.markdown(
    """
    <style>
    .custom-expander .streamlit-expanderHeader {
        background: linear-gradient(90deg, #e0f7fa 0%, #b2ebf2 100%);
        color: #0077b6 !important;
        font-weight: 800;
        font-size: 1.15rem;
        border-radius: 10px 10px 0 0;
        padding: 0.7em 1em;
    }
    .custom-expander .streamlit-expanderContent {
        background: #fafdff;
        border-radius: 0 0 10px 10px;
        padding-top: 0.5em;
        padding-bottom: 1em;
    }
    .pollutant-card {
        background: #e3f2fd;
        border-left: 6px solid #0077b6;
        border-radius: 8px;
        margin-bottom: 1.1em;
        padding: 0.7em 1em 0.7em 1.1em;
        box-shadow: 0 2px 8px rgba(0,119,182,0.05);
    }
    .pollutant-title {
        font-weight: 700;
        color: #0077b6;
        font-size: 1.08em;
        margin-bottom: 0.1em;
    }
    .pollutant-desc {
        font-size: 0.98em;
        color: #333;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Info dropdown
with st.expander("Know What You're Measuring", expanded=False):
    st.markdown(
        """
        <div style="background: #fff; border-radius: 14px; padding: 1.5em 1.2em; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="margin-bottom:1.5em; font-size:1.08em;">
                <b>What are water pollutants?</b><br>
                Water pollutants are substances that contaminate water bodies and can harm aquatic life, the environment, and human health. They come from sources like agriculture, industry, sewage, and urban runoff.
                <ul style="margin-top:0.5em;">
                    <li><b>Chemical pollutants</b> – e.g., nitrates, phosphates, heavy metals, industrial chemicals</li>
                    <li><b>Biological pollutants</b> – e.g., bacteria, viruses, parasites from sewage and animal waste</li>
                    <li><b>Physical pollutants</b> – e.g., sediments, plastics, heat (thermal pollution)</li>
                </ul>
                <b>Why are these measured?</b><br>
                Monitoring these pollutants helps ensure water is safe for drinking, agriculture, and aquatic life. Exceeding safe limits can cause health issues and disrupt ecosystems.
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">O₂ (Dissolved Oxygen)</div>
                <div class="pollutant-desc">Essential for aquatic life. Should be <b>&gt; 5 mg/L</b>.</div>
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">NO₃ (Nitrate)</div>
                <div class="pollutant-desc">From fertilizers/sewage. Can cause eutrophication and health risks if too high.</div>
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">NO₂ (Nitrite)</div>
                <div class="pollutant-desc">Toxic even at low levels. Should be <b>&lt; 0.1 mg/L</b>.</div>
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">SO₄ (Sulfate)</div>
                <div class="pollutant-desc">Naturally occurring; &gt;250 mg/L affects taste/corrosion and can cause digestive issues.</div>
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">PO₄ (Phosphate)</div>
                <div class="pollutant-desc">Promotes algal blooms. Should be <b>&lt; 0.1 mg/L</b>.</div>
            </div>
            <div class="pollutant-card">
                <div class="pollutant-title">CL (Chloride)</div>
                <div class="pollutant-desc">Affects water taste; can be toxic to freshwater species at high levels.</div>
            </div>
            <div style="margin-top:1.5em; font-size:0.98em; color:#444;">
                <b>Health & Environmental Impact:</b><br>
                - High nitrate/nitrite can cause serious illness, especially in infants.<br>
                - Low dissolved oxygen kills fish and aquatic organisms.<br>
                - Excess phosphates and nitrates lead to algal blooms, harming water quality.<br>
                - Monitoring these helps protect both people and ecosystems.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
