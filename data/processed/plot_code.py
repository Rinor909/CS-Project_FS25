
# Code to recreate the interactive scatter plot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
pred_actual_df = pd.read_csv('model_evaluation_results.csv')

# Create the scatter plot
fig = px.scatter(
    pred_actual_df, 
    x='Tatsächlicher Preis (CHF)', 
    y='Vorhergesagter Preis (CHF)',
    opacity=0.7
)

# Add a perfect prediction line
min_val = min(pred_actual_df['Tatsächlicher Preis (CHF)'].min(), 
              pred_actual_df['Vorhergesagter Preis (CHF)'].min())
max_val = max(pred_actual_df['Tatsächlicher Preis (CHF)'].max(), 
              pred_actual_df['Vorhergesagter Preis (CHF)'].max())

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val], 
        mode='lines', 
        name='Perfekte Vorhersage',
        line=dict(color='red', dash='dash')
    )
)

# Improve chart styling
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(l=40, r=20, t=30, b=20)
)

# Show the figure
fig.show()
