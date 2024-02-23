import plotly.graph_objects as go
import plotly.io as pio

# Data for "Hybrid Weighted Linear"
weights_linear = ['2', '3', '4']
linear_accuracies = [81.68, 81.58, 81.38]

# Data for "Hybrid Weighted Exponential"
weights_exponential = ['4/3', '3/2', '2']
exponential_accuracies = [79.06, 80.38, 80.08]

# Create the plot for Hybrid Weighted Linear
fig_linear = go.Figure(data=[
    go.Bar(x=weights_linear, y=linear_accuracies, text=linear_accuracies, textposition='auto')
])

fig_linear.update_layout(
    title='Hybrid Weighted Linear',
    xaxis_title='Weights',
    yaxis_title='Accuracy (%)',
    template='plotly_white'
)

file_path = '../plotly_result/linear_weight.png'
pio.write_image(fig_linear, file_path)


# Create the plot for Hybrid Weighted Exponential
fig_exponential = go.Figure(data=[
    go.Bar(x=weights_exponential, y=exponential_accuracies, text=exponential_accuracies, textposition='auto')
])

fig_exponential.update_layout(
    title='Hybrid Weighted Exponential',
    xaxis_title='Weights',
    yaxis_title='Accuracy (%)',
    template='plotly_white'
)

file_path = '../plotly_result/exp_weight.png'
pio.write_image(fig_exponential, file_path)