from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from virus_model import *
import argparse

def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'r': 0.6,
        'Color': '#66F',
        'Filled': 'true'
    }

    if agent.infected == True:
        portrayal['Color'] = '#F66'

    if agent.immune == True:
        portrayal['Color'] = '#6C6'

    if agent.lockdown == True:
        portrayal['Filled'] = 'false'

    return portrayal

grid = CanvasGrid(agent_portrayal, model_params['width'], model_params['height'], 1000, 700)

line_charts = ChartModule(series = [
    {'Label': '易感人群', 'Color': '#66F', 'Filled': 'true'}, 
    {'Label': '感染人群', 'Color': '#F66', 'Filled': 'true'},
    {'Label': '死亡人群', 'Color': 'black', 'Filled': 'true'},
    {'Label': '痊愈及免疫人群', 'Color': '#6C6', 'Filled': 'true'},
    {'Label': '隔离人群', 'Color': '#6FF', 'Filled': 'false', 'r': 0.8}
    ])

server = ModularServer(VirusModel, [grid, line_charts], '病毒传播模拟', model_params)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--port', '-n', help='name 端口，非必要参数', default=2233)
args = parser.parse_args()

if __name__ == '__main__':
    try:
        server.port = args.port  # default port if unspecified
        server.launch()
    except Exception as e:
        print(e)

