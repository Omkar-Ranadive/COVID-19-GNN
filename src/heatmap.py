import plotly.graph_objects as go
import pandas as pd
from src.utils import load_pickle
from pathlib import Path

SOURCE_PATH = Path(__file__).parent

codes = load_pickle(SOURCE_PATH / '../dataset/timeseries/data/state_codes.pkl')
output = load_pickle(SOURCE_PATH / '../dataset/timeseries/data/outputs.pkl')
featues = load_pickle(SOURCE_PATH / '../dataset/timeseries/data/features.pkl')

df = pd.DataFrame({'code':list(codes.keys()), 'y':output, 'population':featues[:, -3] , 'density': featues[:, -2] ,'pop65': featues[:,-1]})

for col in df.columns:
    df[col] = df[col].astype(str)

df['text'] = 'Cases: ' + df['y'] + "<br>" + 'Population: ' + df['population'] + "<br>" + 'Population Density: ' \
             + df['density'] + "<br>" + 'Population above 65: ' + df['pop65']


fig = go.Figure(data=go.Choropleth(
    locations=df['code'],
    z=df['y'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=df['text'], # hover text
    marker_line_color='white', # line markers between states
    colorbar_title="No. cases"
))

fig.update_layout(
    title_text='No of coronavirus cases by State<br>(Hover for breakdown)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()