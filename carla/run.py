#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import math
import numpy as np
import random
import time
import threading

import carla
from agents.navigation.basic_agent import BasicAgent
from test_agent import *

from cuda_agent import *
from environment import *

DEBUG = False
NUM_OBSTACLES = 0
SPAWN_POINT_INDICES = [116,198]
AGENT = 'test'


def game_loop(options_dict):
    world = None

    try:
        # load the client and change the world
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)

        print('Changing world to Town 5.')
        client.load_world('Town05') 

        # create world object
        world = World(client.get_world())
        spawn_points = world.world.get_map().get_spawn_points()

        # spawn vehicle
        vehicle_bp = 'model3'
        vehicle_transform = spawn_points[options_dict['spawn_point_indices'][0]]
        vehicle_transform.location.x -= 6
        print(f'Starting from {vehicle_transform}.')
        vehicle = Car(vehicle_bp, vehicle_transform, world)

        # # add obstacles and get sample nodes
        # world.block_road()
        # world.swerve_obstacles()
        world.random_obstacles(options_dict['num_obstacles'])

        world_snapshot = None
        while not world_snapshot:
            world.world.tick()
            world_snapshot = world.world.wait_for_tick(10.0)

        # select control agent
        if options_dict['agent'] == 'cuda':
            agent = CudaAgent(vehicle.vehicle)
        elif options_dict['agent'] == 'test':
            agent = TestAgent(vehicle.vehicle)
        else:
            agent = BasicAgent(vehicle.vehicle)
        
        # get and set destination
        destination_transform = spawn_points[options_dict['spawn_point_indices'][1]]
        # destination_transform = carla.Transform(carla.Location(vehicle_transform.location.x -50, vehicle_transform.location.y, vehicle_transform.location.z), carla.Rotation(yaw=vehicle_transform.rotation.yaw))
        destination_point = destination_transform.location

        print(f'Starting from {vehicle_transform}.')
        print(f'Going to {destination_transform}.')
        # agent.set_destination((destination_point.x, destination_point.y, destination_point.z))
        agent.set_destination(vehicle_transform, destination_transform)
        # agent.create_samples(vehicle_transform, destination_transform)
        
        # attach sensors to vehicle
        sensor_bp = ['sensor.camera.rgb', "sensor.camera.semantic_segmentation", "sensor.camera.depth"]
        sensor_transform = carla.Transform(carla.Location(x= 2.5,z=2))

        depth = Camera(sensor_bp[2], sensor_transform, vehicle, agent)
        segment= Camera(sensor_bp[1], sensor_transform, vehicle, agent)


        # run the simulation
        print('Starting the simulation.')
        prev_location = vehicle.vehicle.get_location()
        sp = 2
        while True:
            # wait for server to be ready
            world.world.tick()
            world_snapshot = world.world.wait_for_tick(10.0)

            if not world_snapshot:
                continue

            # wait for sensors to sync
            # while world_snapshot.frame_count!=depth.frame_n or world_snapshot.frame_count!=segment.frame_n:
            #     time.sleep(0.05)

            # plan, get control inputs, and apply to vehicle
            control = agent.run_step(options_dict['debug'])
            # print(control)
            vehicle.vehicle.apply_control(control)

            # check if destination reached
            current_location = vehicle.vehicle.get_location()
            # kind of hacky way to test destination reached and doesn't always work - may have to manually stop with ctrl c
            if current_location.distance(prev_location) <= 0.0 and current_location.distance(destination_point) <= 10: 
                print('distance from destination: ', current_location.distance(destination_point))
                # if out of destinations break else go to next destination
                if len(options_dict['spawn_point_indices']) <= sp:
                    break
                else:
                    destination_point = spawn_points[options_dict['spawn_point_indices'][sp]].location
                    print('Going to ', destination_point)
                    agent.set_destination((destination_point.x, destination_point.y, destination_point.z))
                    sp += 1

            prev_location = current_location

    finally:
        if world is not None:
            world.destroy()


if __name__ == '__main__':
    sensor_dict = {
        'IM_WIDTH': 400,
        'IM_HEIGHT': 300,
        'SENSOR_TICK': 0.0,
        'FOV': 120
    }
    sensor_attributes(sensor_dict)

    options_dict = {
        'agent': AGENT,
        'spawn_point_indices': SPAWN_POINT_INDICES,
        'num_obstacles': NUM_OBSTACLES,
        'debug': DEBUG
    }
    game_loop(options_dict)
