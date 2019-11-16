import math

import numpy as np
import networkx as nx

import carla
from agents.navigation.agent import Agent, AgentState
from gmt_planner import *


class CudaAgent(Agent):
    def __init__(self, vehicle, target_speed=50):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        super(CudaAgent, self).__init__(vehicle)
        pass

    def set_destination(self, location):
        pass

    def _trace_route(self, start_waypoint, end_waypoint):
        pass

    def run_step(self, debug=False):
        pass