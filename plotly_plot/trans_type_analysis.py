import plotly.express as px

categories = ['家-工作-家', '家-購物-家', '家-休閒-家', '其他']
percentages = [0.3, 47, 46, 6.7]

# Draw the pie chart
fig = px.pie(
    values=percentages, 
    names=categories, 
    title='Transport Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu
)

fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0, 0])
fig.show()
