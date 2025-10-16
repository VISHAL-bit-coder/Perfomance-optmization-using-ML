import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create a system architecture diagram using Plotly
# Define the layers and their components
layers = {
    "Data Collection": {"color": "#1FB8CD", "y": 4, "components": ["System Monitor\n(psutil)", "Data Preprocessor\n(Python)", "SQLite Database\n(Storage)"]},
    "ML Models": {"color": "#2E8B57", "y": 3, "components": ["LSTM Forecaster\n(TensorFlow)", "Anomaly Detector\n(Isolation Forest)", "Model Trainer\n(Scikit-learn)"]},
    "API": {"color": "#D2BA4C", "y": 2, "components": ["Flask REST API\n(Python Flask)", "Real-time Stream\n(WebSocket)"]},
    "Monitoring": {"color": "#DB4545", "y": 1, "components": ["MLflow\n(Tracking)", "Grafana\n(Visualization)", "Prometheus\n(Metrics)", "Dashboard\n(Web UI)"]},
    "Deployment": {"color": "#5D878F", "y": 0, "components": ["Docker Containers\n(Containerization)", "Docker Compose\n(Orchestration)"]}
}

# Create the figure
fig = go.Figure()

# Add components as scatter points with text
x_positions = []
y_positions = []
texts = []
colors = []
sizes = []

# Define flows between components with better positioning
flows = [
    ("System Monitor", "Data Preprocessor", "Raw Metrics"),
    ("Data Preprocessor", "SQLite Database", "Processed Data"),
    ("SQLite Database", "Model Trainer", "Training Data"),
    ("Model Trainer", "LSTM Forecaster", "Trained Model"),
    ("Model Trainer", "Anomaly Detector", "Trained Model"),
    ("LSTM Forecaster", "Flask REST API", "Predictions"),
    ("Anomaly Detector", "Flask REST API", "Anomalies"),
    ("Flask REST API", "Real-time Stream", "Live Data"),
    ("Model Trainer", "MLflow", "Experiments"),
    ("System Monitor", "Prometheus", "Metrics"),
    ("Prometheus", "Grafana", "Visualization"),
    ("Docker Compose", "Docker Containers", "Orchestrates")
]

# Position components in each layer with better spacing
component_positions = {}
for layer_name, layer_info in layers.items():
    y = layer_info["y"]
    components = layer_info["components"]
    color = layer_info["color"]
    
    # Distribute components horizontally within each layer with more spacing
    if len(components) == 1:
        x_coords = [0.5]
    elif len(components) == 2:
        x_coords = [0.2, 0.8]
    elif len(components) == 3:
        x_coords = [0.1, 0.5, 0.9]
    elif len(components) == 4:
        x_coords = [0.05, 0.35, 0.65, 0.95]
    else:
        x_coords = np.linspace(0.05, 0.95, len(components))
    
    for i, component in enumerate(components):
        x = x_coords[i]
        x_positions.append(x)
        y_positions.append(y)
        texts.append(component)
        colors.append(color)
        sizes.append(35)  # Increased size for better visibility
        
        # Store position for flow arrows
        simple_name = component.split('\n')[0]
        component_positions[simple_name] = (x, y)

# Add components as scatter plot with better text formatting
fig.add_trace(go.Scatter(
    x=x_positions,
    y=y_positions,
    mode='markers+text',
    text=texts,
    textposition='middle center',
    textfont=dict(size=11, color='white', family="Arial Black"),  # Increased size and bold font
    marker=dict(
        size=sizes,
        color=colors,
        symbol='square',
        line=dict(width=3, color='white')  # Thicker border for better contrast
    ),
    hoverinfo='text',
    hovertext=texts,
    showlegend=False
))

# Add flow arrows with labels
arrow_annotations = []
for flow in flows:
    from_comp, to_comp, label = flow
    
    # Find positions
    from_pos = None
    to_pos = None
    
    for comp_name, pos in component_positions.items():
        if from_comp in comp_name:
            from_pos = pos
        if to_comp in comp_name:
            to_pos = pos
    
    if from_pos and to_pos:
        # Calculate midpoint for label placement
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        
        # Add arrow annotation
        arrow_annotations.append(dict(
            x=to_pos[0], y=to_pos[1],
            ax=from_pos[0], ay=from_pos[1],
            xref='x', yref='y',
            axref='x', ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#333333'
        ))
        
        # Add flow label
        arrow_annotations.append(dict(
            x=mid_x, y=mid_y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=9, color="#333333", family="Arial"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#333333",
            borderwidth=1
        ))

# Add layer background rectangles with better styling
for layer_name, layer_info in layers.items():
    y = layer_info["y"]
    color = layer_info["color"]
    
    fig.add_shape(
        type="rect",
        x0=-0.02, y0=y-0.4,
        x1=1.02, y1=y+0.4,
        fillcolor=color,
        opacity=0.15,
        layer="below",
        line=dict(color=color, width=2)
    )

# Add layer labels with better alignment
layer_labels = []
for layer_name, layer_info in layers.items():
    y = layer_info["y"]
    color = layer_info["color"]
    layer_labels.append(dict(
        x=-0.05, y=y,
        text=f"<b>{layer_name}</b>",
        showarrow=False,
        font=dict(size=14, color=color, family="Arial Black"),
        xanchor="right",
        yanchor="middle"
    ))

# Add technology stack labels
tech_labels = [
    dict(x=0.5, y=4.6, text="<b>Technologies: Python, psutil, SQLite</b>", 
         font=dict(size=10, color="#1FB8CD"), xanchor="center"),
    dict(x=0.5, y=3.6, text="<b>Technologies: TensorFlow, Scikit-learn</b>", 
         font=dict(size=10, color="#2E8B57"), xanchor="center"),
    dict(x=0.5, y=2.6, text="<b>Technologies: Flask, WebSocket</b>", 
         font=dict(size=10, color="#D2BA4C"), xanchor="center"),
    dict(x=0.5, y=1.6, text="<b>Technologies: MLflow, Grafana, Prometheus</b>", 
         font=dict(size=10, color="#DB4545"), xanchor="center"),
    dict(x=0.5, y=0.6, text="<b>Technologies: Docker, Docker Compose</b>", 
         font=dict(size=10, color="#5D878F"), xanchor="center")
]

# Update layout
fig.update_layout(
    title="ML System Optimizer Architecture",
    xaxis=dict(
        range=[-0.15, 1.15],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[-0.6, 5],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor='white',
    annotations=arrow_annotations + layer_labels + tech_labels,
    showlegend=False
)

# Save the chart
fig.write_image("ml_system_architecture.png")
fig.write_image("ml_system_architecture.svg", format="svg")

print("Enhanced system architecture diagram created successfully!")