import plotly.graph_objects as go
import plotly.io as pio

# Data from the user
sequences = ['3', '4', '5', '6', '7', '8']

# Accuracies list
hybrid_gru_accuracies = [79.27, 75.63, 77.42, 74.1, 77.57, 73.24]
hybrid_gru_adj_accuracies = [91.99, 90.77, 91.96, 90.9, 92.6, 91.28]
# basic_accuracies = [84.28, 81.52, 82.96, 79.21, 82.41, 79.73]

hybrid_gru_accuracies = [round(float(i / 100), 4) for i in hybrid_gru_accuracies]
hybrid_gru_adj_accuracies = [round(float(i / 100), 4) for i in hybrid_gru_adj_accuracies]

# Creating the bar chart for both sets of accuracies
fig = go.Figure()

# Adding Hybrid GRU model accuracies
fig.add_trace(go.Bar(
    x=sequences, 
    y=hybrid_gru_accuracies, 
    name='Strict Accuracy',
    text=hybrid_gru_accuracies, 
    textposition='auto', 
    marker_color='lightgreen'
))

# Adding basic model accuracies
fig.add_trace(go.Bar(
    x=sequences, 
    y=hybrid_gru_adj_accuracies, 
    name='Adjacent Accuracy',
    text=hybrid_gru_adj_accuracies, 
    textposition='auto', 
    marker_color='lightblue'
))

# Customizing the layout with a lighter theme
fig.update_layout(
    xaxis=dict(
        title='Sequence Length',
        tickfont=dict(size=15, color='black')
    ),
    title={
        'text': 'Accuracies under different sequence length',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font=dict(size=30, color='Black', family='Arial, sans-serif'),
    xaxis_title='Sequence Length',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0, 1]),
    template='plotly_white',
    barmode='group'
)

# Show the plot
fig.show()

# Save the figure to a file
file_path = '../plotly_result/diff_seq_acc.png'
pio.write_image(fig, file_path, width=1920, height=1080, scale=2) 
