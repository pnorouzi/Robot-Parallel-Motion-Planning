import math
import numpy as np
import networkx as nx

import carla
# from agents.navigation.agent import Agent, AgentState
from agents.tools.misc import draw_waypoints
from agents.tools.misc import get_speed

from localized_controller import VehiclePIDController
from gmt_planner import *


class CudaAgent(object):
    def __init__(self, vehicle, target_speed=50):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        super(CudaAgent, self).__init__(vehicle)
        self._target_speed = target_speed # km/h
        self._vehicle = vehicle 
        self._proximity_threshold = 10.0  # meters
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()

        self.current_location = self._vehicle.get_transform() 
        self.current_speed = get_speed(self._vehicle)
        self.obstacle_list = []

        self._dt = 1.0 / 20.0
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        self._vehicle_controller = VehiclePIDController(self._vehicle, args_lateral=args_lateral_dict, args_longitudinal=args_longitudinal_dict)

    def set_destination(self, location):
        self.start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.end_waypoint = self._map.get_waypoint(carla.Location(location[0], location[1], location[2]))

    def _trace_route(self):
        ## TODO ## 
        # obstacle detection #
        # path planning #

        # obstacle_list = [] # detection
        # gmt(self._vehicle.get_location(), self.end_waypoint, obstacle_list)
        # waypoint = world.map.get_waypoint(world.player.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))

        obstacles = []
        for vehicle in self._world.get_actors().filter('vehicle.*'):
                #print(vehicle.bounding_box)
                # draw Box
                transform = vehicle.get_transform()
                bounding_box = vehicle.bounding_box
                bounding_box.location += transform.location
                my_location = self.current_location.location
                dist = np.sqrt((my_location.x-bounding_box.location.x)**2 + (my_location.y-bounding_box.location.y)**2 + (my_location.z-bounding_box.location.z)**2)

                if dist <=30:
                    vehicle_box = [bounding_box.location.x - bounding_box.extent.x,bounding_box.location.y - bounding_box.extent.y,bounding_box.location.x + bounding_box.extent.x,bounding_box.location.y + bounding_box.extent.y]
                    obstacles.append(vehicle_box)

        if len(obstacle) == 0:
            self._obstacles = np.array([[-1,-1,-1,-1]]).astype(np.float32)
            self.num_obs = np.int32(0)
        else:
            self._obstacles = np.array(obstacles).astype(np.float32)
            self.num_obs = np.int32(self._obstacles.shape[0])

        pass

    def run_step(self, debug=False):
        ## TODO ## 
        # state estimation #
        # velocity estimation #
        self.current_location = self._vehicle.get_transform()
        self.current_speed = get_speed(self._vehicle)

        route = self._trace_route() # get plan

        control = self._vehicle_controller.run_step(self._target_speed, current_speed, route[0], current_location) # execute first step of plan

        if debug: # draw plan
            draw_waypoints(self._vehicle.get_world(), route, self._vehicle.get_location().z + 1.0)