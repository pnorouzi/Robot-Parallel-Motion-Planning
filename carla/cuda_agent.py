import math
import numpy as np
import networkx as nx
from timeit import default_timer as timer

import carla
from agents.navigation.agent import Agent, AgentState
from agents.tools.misc import draw_waypoints
from agents.tools.misc import get_speed

from localized_controller import VehiclePIDController
from gmt_planner import *

class CudaAgent(Agent):
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
        self.plan = True

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

        steering_list = []
        for v in self._vehicle.get_physics_control().steering_curve:
            steering_list.append([v.x, v.y])

        self.steering_curve = np.array(steering_list)    

    def set_destination(self, start_waypoint, end_waypoint):
        self.create_samples(start_waypoint, end_waypoint)

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

        init_parameters = {'states':self.states, 'neighbors':self.neighbors, 'num_neighbors':self.num_neighbors}
        self.start = self.states.shape[0] - 1
        self.goal = self.states.shape[0] - 2

        print(f'start: {self.start} goal: {self.goal} total states: {self.states.shape[0]}')
        print(f'start location: {self.states[self.start]}, goal location: {self.states[self.goal]}')

        # self.gmt_planner = GMT(init_parameters, debug=False)
        self.gmt_planner = GMTmem(init_parameters, debug=False)
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
        bb_vehicle_matrix = CudaAgent.get_matrix(bb_transform)
        vehicle_world_matrix = CudaAgent.get_matrix(vehicle.get_transform())
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
        ## TODO ## 
        # obstacle detection #
        # path planning #

        self.obstacles = np.array([[-1,-1,-1,-1]]).astype(np.float32)
        self.num_obs = self.num_obs = np.array([0]).astype(np.int32)

        # obstacles = []
        # for vehicle in self._world.get_actors().filter('vehicle.*'):
        #         # draw Box
        #         bb_points = CudaAgent._create_bb_points(vehicle)
        #         global_points= CudaAgent._vehicle_to_world(bb_points, vehicle)
        #         global_points /= global_points[3,:]

        #         my_bb_points = CudaAgent._create_bb_points(self._vehicle)
        #         my_global_points = CudaAgent._vehicle_to_world(my_bb_points, self._vehicle)

        #         my_global_points /= my_global_points[3,:]
        #         dist = np.sqrt((my_global_points[0,2]-global_points[0,2])**2 + (my_global_points[1,2]-global_points[1,2])**2 + (my_global_points[2,2]-global_points[2,2])**2)

        #         if 0<dist <=30:
        #             vehicle_box = [global_points[0,0],global_points[1,0],global_points[0,1],global_points[1,1]]
        #             obstacles.append(vehicle_box)

        # print('number of near obstacles: ', len(obstacles))
        # if len(obstacles) == 0:
        #     self.obstacles = np.array([[-1,-1,-1,-1]]).astype(np.float32)
        #     self.num_obs = self.num_obs = np.array([0]).astype(np.int32)
        # else:
        #     self.obstacles = np.array(obstacles).astype(np.float32)
        #     self.num_obs = self.num_obs = np.array([self.obstacles.shape[0]]).astype(np.int32)

        print(self.current_speed)

        closest_speed = self.steering_curve[:,0] - self.current_speed
        idx = np.argmin(closest_speed)
        angle = self.steering_curve[idx,1]

        iter_parameters = {'start':self.start, 'goal':self.goal, 'radius':self.radius*angle, 'threshold':self.threshold*angle, 'obstacles':self.obstacles, 'num_obs':self.num_obs}
        
        start = timer()
        route = self.gmt_planner.run_step(iter_parameters, iter_limit=60, debug=debug)
        end = timer()

        print("elapsed time: ", end-start)

        if debug:
            print('route: ', route)
        return route

    def run_step(self, debug=False):
        ## TODO ## 
        # state estimation #
        # velocity estimation #
        self.current_location = self._vehicle.get_transform()
        self.current_speed = get_speed(self._vehicle)

        self.radius = 2
        self.threshold  = 0.5

        if self.plan:
            self.plan = False
            self.route = self._trace_route(debug) # get plan

            if len(self.route) == 0:
                wp = self.start
            else:
                wp = self.route[-2]

            self.waypoint = self._map.get_waypoint(carla.Location(self.states[wp][0].item(), self.states[wp][1].item(), self.current_location.location.z))
        else:
            self.update_start()

        control = self._vehicle_controller.run_step(self._target_speed, self.current_speed, self.waypoint, self.current_location) # execute first step of plan

        if debug: # draw plan
            trace_route = []
            for r in self.route:
                wp = carla.Transform(carla.Location(self.states[r][0].item(), self.states[r][1].item(), 1.2), carla.Rotation(roll=0,pitch=0, yaw=(self.states[r][2]*180/np.pi).item()))
                trace_route.append(wp)
            draw_route(self._vehicle.get_world(), trace_route)

        return control

    def update_start(self):
        self.current_location = self._vehicle.get_transform()
        route_location = self.route[-3::]

        current_state = np.array([self.current_location.location.x, self.current_location.location.y , self.current_location.rotation.yaw])
        route_state = self.states[route_location, 0:2]

        dist = np.linalg.norm(current_state[0:2]-route_state, axis=1)

        new_start = np.argmin(dist)

        oldStart = self.start
        self.start = route_location[new_start]
        if oldStart != self.start:
            self.plan = True




        