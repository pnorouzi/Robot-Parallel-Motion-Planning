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
        self._target_speed = target_speed

    def set_destination(self, location):
        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(carla.Location(location[0], location[1], location[2]))

    def _trace_route(self, start_waypoint, end_waypoint):
        pass

    def run_step(self, debug=False):
        pass