import os
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer,is_user_param,SocketHandler,PageHandler
from virus_model import *
import argparse
import tornado.web
class DynamicSocketHandler(SocketHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_message(self, message):
        """Receiving a message from the websocket, parse, and act accordingly."""
        if self.application.verbose:
            print(message)
        msg = tornado.escape.json_decode(message)

        if msg["type"] == "get_step":
            if not self.application.model.running:
                self.write_message({"type": "end"})
            else:
                self.application.model.step()
                self.write_message(self.viz_state_message)

        elif msg["type"] == "reset":
            self.application.reset_model()
            self.write_message(self.viz_state_message)

        elif msg["type"] == "submit_params":
            param = msg["param"]
            value = msg["value"]

            # Is the param editable?
            if param in self.application.user_params:
                if is_user_param(self.application.model_kwargs[param]):
                    self.application.model_kwargs[param].value = value
                else:
                    self.application.model_kwargs[param] = value
            self.application.update_model()
        else:
            if self.application.verbose:
                print("Unexpected message!")

class DynamicModelServer(ModularServer):
    def __init__(self, *args, **kwargs):
        super(DynamicModelServer, self).__init__(*args, **kwargs)
        self.handlers[1]=(r"/ws", DynamicSocketHandler)
        super(ModularServer,self).__init__(self.handlers, **self.settings)
    def update_model(self):
        model_params = {}
        for key, val in self.model_kwargs.items():
            if is_user_param(val):
                if val.param_type == "static_text" or not val.description == 'dynamic':
                    # static_text is never used for setting params
                    continue
                model_params[key] = val.value
            else:
                model_params[key] = val
        self.model.update_param(**model_params)
        
def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'id': agent.unique_id,
        'status': "易感" if agent.infected == False and agent.immune == False 
            else "感染" if agent.infected == True 
            else "免疫" if agent.immune == True 
            else "死亡" + ' ' + 
            "隔离" if agent.lockdown == True else "",
        'Layer': 0,
        'r': 0.6,
        'Color': '#66F',
        'Filled': 'true',
        'masked': "戴口罩" if agent.masked == True else "未戴口罩",
        'lockdown': "隔离" if agent.lockdown == True else "未隔离",
    }

    if agent.infected == True:
        portrayal['Color'] = '#F66'

    if agent.immune == True:
        portrayal['Color'] = '#6C6'

    if agent.lockdown == True:
        portrayal['Filled'] = 'false'
        # portrayal['Shape'] = 'rect'
        # portrayal['w'] = 0.6
        # portrayal['h'] = 0.6
        portrayal['Color'] = '#DD1' # 黄色
    return portrayal

grid = CanvasGrid(agent_portrayal, model_params['width'], model_params['height'], 1000, 700)

line_charts = ChartModule(series = [
    {'Label': '易感人群', 'Color': '#66F', 'Filled': 'true'}, 
    {'Label': '隔离人群', 'Color': '#DD1', 'Filled': 'true'},
    {'Label': '感染人群', 'Color': '#F66', 'Filled': 'true'},
    {'Label': '死亡人群', 'Color': 'black', 'Filled': 'true'},
    {'Label': '痊愈及免疫人群', 'Color': '#6C6', 'Filled': 'true'},
    
    ],
    canvas_height=700,
    canvas_width=1800,
    )

server = DynamicModelServer(VirusModel, [grid, line_charts], '病毒传播模拟', model_params)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--port', '-n', help='name 端口，非必要参数', default=2233)
args = parser.parse_args()

if __name__ == '__main__':
    try:
        server.port = args.port  # default port if unspecified
        server.launch()
    except Exception as e:
        print(e)

