#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import math
import numpy as np
import networkx as nx
from timeit import default_timer as timer

import carla
from agents.tools.misc import *

# from localized_controller import VehiclePIDController
from gmt_planner import *

class TestAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=50):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(TestAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()

    def set_destination(self, start_waypoint, end_waypoint):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        self.create_samples(start_waypoint, end_waypoint)

        route_trace = self._trace_route()
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

    def create_samples(self, start, goal, waypoint_dist = 4, disk_radius = 4*math.sqrt(2), num_yaw = 8):
        print(f'Creating samples {waypoint_dist}m apart with {num_yaw} yaw vaules and neighbors within {disk_radius}m.')

        wp = []
        for mp in self._map.generate_waypoints(waypoint_dist):
            wp.append(mp.transform)

        wp.append(goal)
        wp.append(start)

        states = []
        neighbors = []
        num_neighbors = []

        # for each waypoint wp
        for i, wi in enumerate(wp):
            li = wi.location
            ni = []
            num  = 0
            # find other waypoints within disk radius
            for j, wj in enumerate(wp):
                lj = wj.location
                if li == lj:
                    continue
                elif li.distance(lj) <= disk_radius:
                    # account for index shifts with adding in orientation
                    for k in range(num_yaw):
                        if k == (num_yaw)/2:
                            continue
                        elif k > (num_yaw)/2:
                            ni.append(j*(num_yaw-1) + k-1)
                        else:
                            ni.append(j*(num_yaw-1) + k)
                        num += 1
                        if wj == start or wj == goal:
                            break

            # add in number of yaw orientations to waypoint list        
            ri = wi.rotation
            for k in range(num_yaw):
                if k == (num_yaw)/2:
                    continue

                num_neighbors.append(num)
                neighbors += ni

                theta = ri.yaw + k*360/(num_yaw)
                if theta >= 180:
                    theta = theta - 360
                elif theta <= -180:
                    theta = 360 - theta
                states.append([li.x, li.y, theta*np.pi/180])
                if wi == start or wi == goal:
                    break

        self.states = np.array(states).astype(np.float32)
        self.neighbors = np.array(neighbors).astype(np.int32)
        self.num_neighbors = np.array(num_neighbors).astype(np.int32)

        init_parameters = {'states':self.states, 'neighbors':self.neighbors, 'num_neighbors':self.num_neighbors, 'threadsPerBlock':128}
        self.start = self.states.shape[0] - 1
        self.goal = self.states.shape[0] - 2

        print(f'start: {self.start} goal: {self.goal} total states: {self.states.shape[0]}')
        print(f'start location: {self.states[self.start]}, goal location: {self.states[self.goal]}')
    
        self.gmt_planner = GMT(init_parameters, debug=False)
        # self.gmt_planner = GMTmem(init_parameters, debug=False)
        # self.gmt_planner = GMTstream(init_parameters, debug=False)

    @staticmethod
    def _create_bb_points(vehicle):

        cords = np.zeros((3, 4))
        extent = vehicle.bounding_box.extent

        cords[0, :] = np.array([extent.x + 2.2, extent.y + 2.2, extent.z, 1])
        cords[1, :] = np.array([-extent.x - 2.2, -extent.y - 2.2, extent.z, 1])
        cords[2, :] = np.array([0, 0, 0, 1])    # center

        return cords

    @staticmethod
    def _vehicle_to_world(cords, vehicle):

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = TestAgent.get_matrix(bb_transform)
        vehicle_world_matrix = TestAgent.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def get_matrix(transform):

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _trace_route(self, debug=False):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """
        self.radius = 2
        self.threshold  = 1

        obstacles = []
        for vehicle in self._world.get_actors().filter('vehicle.*'):
                #print(vehicle.bounding_box)
                # draw Box
                bb_points = TestAgent._create_bb_points(vehicle)
                global_points= TestAgent._vehicle_to_world(bb_points, vehicle)
                global_points /= global_points[3,:]

                my_bb_points = TestAgent._create_bb_points(self._vehicle)
                my_global_points = TestAgent._vehicle_to_world(my_bb_points, self._vehicle)

                my_global_points /= my_global_points[3,:]
                dist = np.sqrt((my_global_points[0,2]-global_points[0,2])**2 + (my_global_points[1,2]-global_points[1,2])**2 + (my_global_points[2,2]-global_points[2,2])**2)

                if 0<dist:
                    vehicle_box = [global_points[0,0],global_points[1,0],global_points[0,1],global_points[1,1]]
                    obstacles.append(vehicle_box)
                    print(f'vehicle box: {vehicle_box}')

        print('number of near obstacles: ', len(obstacles))
        if len(obstacles) == 0:
            self.obstacles = np.array([[-1,-1,-1,-1]]).astype(np.float32)
            self.num_obs = self.num_obs = np.array([0]).astype(np.int32)
        else:
            self.obstacles = np.array(obstacles).astype(np.float32)
            self.num_obs = self.num_obs = np.array([self.obstacles.shape[0]]).astype(np.int32)

        iter_parameters = {'start':self.start, 'goal':self.goal, 'radius':self.radius, 'threshold':self.threshold, 'obstacles':self.obstacles, 'num_obs':self.num_obs}
        
        start_timer = timer()
        route = self.gmt_planner.run_step(iter_parameters, iter_limit=1000, debug=debug)
        end_timer = timer()

        print("elapsed time: ", end_timer-start_timer)    

        trace_route = []
        for r in route:
            wp = carla.Transform(carla.Location(self.states[r][0].item(), self.states[r][1].item(), 1.2), carla.Rotation(roll=0,pitch=0, yaw=(self.states[r][2]*180/np.pi).item()))
            trace_route.append(wp)
        draw_route(self._vehicle.get_world(), trace_route)

        index = len(route)-1
        trace_route = []
        for i in range(len(route)-1):
            wp = self._map.get_waypoint(carla.Location(self.states[route[index]][0].item(), self.states[route[index]][1].item(), 1.2)) # , carla.Rotation(roll=0,pitch=0, yaw=(self.states[r][2]*180/np.pi).item()
            trace_route.append((wp,-1))
            index -= 1

        return trace_route

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles

        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")


        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = False, None # 
        # light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step(debug)

        return control
