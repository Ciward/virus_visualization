import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.visualization.UserParam import UserSettableParameter                                               
import scipy.stats as ss


# Simulation model parameters
# -------------------------------------------------------------------------------------------
model_params = {
    'no_agents': UserSettableParameter(
        'number', '模拟人数', 500, 5, 5000, 5),
    'width': 70,
    'height': 50,
    'infected_import': UserSettableParameter(
        'slider', '每步输入阳性病例概率', 0.1, 0, 1, 0.05),
    'init_infected': UserSettableParameter(
        'slider', '初始感染率 (百分比)', 0.2, 0, 1, 0.05),
    'perc_masked': UserSettableParameter(
        'slider', '口罩普及率 (百分比)', 0.5, 0, 1, 0.05),
    'prob_trans_masked': UserSettableParameter(
        'slider', '带口罩被传染概率', 0.25, 0, 1, 0.05),
    'prob_trans_unmasked': UserSettableParameter(
        'slider', '不带口罩被传染概率', 0.75, 0, 1, 0.05),
    'prob_fatal': UserSettableParameter(
        'slider', '感染后转危重率', 0.1, 0, 1, 0.05),
    'infection_period': UserSettableParameter(
        'slider', '感染到痊愈所需模拟步数', 50, 5, 200, 5),
    'immunity_period': UserSettableParameter(
        'slider', '免疫力消失所需模拟步数', 200, 10, 1000, 10),
    'isolation_enabled': UserSettableParameter(
        'choice', '是否实行隔离措施', value='否', choices=['否', '是'])
}


# Agent class in mesa
# -------------------------------------------------------------------------------------------
class Agent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.masked = bool(ss.bernoulli.rvs(self.model.perc_masked))
        self.infected = bool(ss.bernoulli.rvs(self.model.init_infected))
        self.immune = False
        self.fatal = False
        self.lockdown = False
        self.recovery_countdown = 0
        self.immunity_countdown = 0
        # Random recovery countdown considers agents got infected at different times
        if self.infected:
            self.recovery_countdown = np.random.randint(1, self.model.infection_period + 1)

    def move(self):
        x, y = self.pos
        new_x = min(max(np.random.choice(
            [-1, 0, 1]) + x, 0), self.model.grid.width - 1)
        new_y = min(max(np.random.choice(
            [-1, 0, 1]) + y, 0), self.model.grid.height - 1)
        if not self.lockdown:
            self.model.grid.move_agent(self, (new_x, new_y))

    def update_infected(self):
        # Infected or immune agents cannot become infected
        if self.infected | self.immune:
            return None
        # Workaround for potential bug
        pos = tuple([int(self.pos[0]), int(self.pos[1])])
        # List of agents in the same grid cell
        cell_agents = self.model.grid.get_cell_list_contents(pos)
        # Checks if any of the agents in the cell are infected
        any_infected = any(a.infected and (not a.lockdown) for a in cell_agents)
        if any_infected:
            if self.masked:
                # Probability of getting infected when masked
                self.infected = bool(ss.bernoulli.rvs(self.model.prob_trans_masked))
            elif ~self.masked:
                # Probability of getting infected when not masked
                self.infected = bool(ss.bernoulli.rvs(self.model.prob_trans_unmasked))
        # Once infected countdown to recovery begins
        if self.infected:
            self.recovery_countdown = self.model.infection_period
            self.fatal = bool(ss.bernoulli.rvs(self.model.prob_fatal))

    def update_recovered(self):
        if self.recovery_countdown == 1:
            self.infected = False
            self.immune = True
            # After recovery countdown to immunity going away begins
            self.immunity_countdown = self.model.immunity_period
        if self.recovery_countdown > 0:
            self.recovery_countdown += -1

    def update_susceptible(self):
        # After immunity wanes away, agent becomes susceptible
        if self.immunity_countdown == 1:
            self.immune = False
        if self.immunity_countdown > 0:
            self.immunity_countdown = self.immunity_countdown - 1

    def update_lockdown(self):
        # If lockdown enabled, the movement of infected agent will be limited
        if (not self.infected | self.immune) or self.model.isolation_enabled == '否':
            self.lockdown = False
        else:
            # the probability 0.9 is a empiric parameter
            self.lockdown = bool(ss.bernoulli.rvs(0.9))

    def step(self):
        self.move()
        self.update_infected()
        self.update_recovered()
        self.update_susceptible()
        self.update_lockdown()


# Model class in Mesa
# -------------------------------------------------------------------------------------------
class VirusModel(Model):
    """A SIR-like model with a number of agents that potentially transmit COVID
    when they are on the same cell of the grid"""

    def __init__(self, no_agents, width, height,
                 init_infected, perc_masked, prob_trans_masked,
                 prob_trans_unmasked, infection_period, immunity_period, 
                 isolation_enabled, prob_fatal, infected_import):
        self.no_agents = no_agents
        self.grid = MultiGrid(width, height, False)
        self.init_infected = init_infected
        self.perc_masked = perc_masked
        self.prob_trans_masked = prob_trans_masked
        self.prob_trans_unmasked = prob_trans_unmasked
        self.infection_period = infection_period
        self.immunity_period = immunity_period
        self.isolation_enabled = isolation_enabled
        self.prob_fatal = prob_fatal
        self.infected_import = infected_import
        self.schedule = RandomActivation(self)
        self.running = True

        self.deaths = 0

        # Create agents
        for i in range(self.no_agents):
            a = Agent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Collect count of susceptible, infected, and recovered agents
        self.datacollector = DataCollector({
            '易感人群': 'susceptible',
            '感染人群': 'infected',
            '死亡人群': 'death',
            '痊愈及免疫人群': 'immune'})

    @property
    def susceptible(self):
        agents = self.schedule.agents
        susceptible = [not(a.immune | a.infected) for a in agents]
        return int(np.sum(susceptible))

    @property
    def infected(self):
        agents = self.schedule.agents
        infected = [a.infected for a in agents]
        return int(np.sum(infected))

    @property
    def immune(self):
        agents = self.schedule.agents
        immune = [a.immune for a in agents]
        return int(np.sum(immune))
    
    @property
    def death(self):
        return self.deaths

    def remove_deaths(self):
        for a in self.schedule.agents:
            if a.fatal and bool(ss.bernoulli.rvs(0.1)):
                self.schedule.remove(a)
                self.grid.remove_agent(a)
                self.deaths += 1

    def add_import_infected(self):
        if ss.bernoulli.rvs(self.infected_import):
            a = Agent(self.random.randrange(9999, 999999), self)
            a.infected = True
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.remove_deaths()
        if bool(ss.bernoulli.rvs(self.infected_import)):
            self.add_import_infected()