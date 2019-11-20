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

from cuda_agent import *
from environment import *

DEBUG = False
NUM_OBSTACLES = 20
SPAWN_POINT_INDICES = [116,198,116]
AGENT = 'basic'

def do_something(world_snapshot):
    print(world_snapshot)



def game_loop(options_dict):
    world = None

    try:
        # load the client and change the world
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)

        print('Changing world to Town 5')
        client.load_world('Town05') 

        world = World(client.get_world())

        spawn_points = world.world.get_map().get_spawn_points()

        vehicle_bp = 'model3'
        vehicle_transform = spawn_points[options_dict['spawn_point_indices'][0]]
        
        vehicle = Car(vehicle_bp, vehicle_transform, world)

        if options_dict['agent'] == 'cuda':
            agent = CudaAgent(vehicle.vehicle)
        else:
            agent = BasicAgent(vehicle.vehicle)
        
        destination_point = spawn_points[options_dict['spawn_point_indices'][1]].location

        print('Going to ', destination_point)
        agent.set_destination((destination_point.x, destination_point.y, destination_point.z))
        
        camera_bp = ['sensor.camera.rgb', "sensor.camera.semantic_segmentation", "sensor.camera.depth"]
        camera_transform = carla.Transform(carla.Location(x= 2.5,z=2))

        

        #camera = Camera(camera_bp[0], camera_transform, vehicle, agent)
        
        #camera = Camera(camera_bp[0], camera_transform, vehicle, agent)
        
        
        #
        #depth = Camera(camera_bp[2], camera_transform, vehicle, agent)

        world.create_obstacles(options_dict['num_obstacles'])

        prev_location = vehicle.vehicle.get_location()

        depth = Camera(camera_bp[2], camera_transform, vehicle, agent)

        segment= Camera(camera_bp[1], camera_transform, vehicle, agent)

        #print(depth.sensor)

        

        sp = 2
        while True:
            world.world.tick()

            world_snapshot = world.world.wait_for_tick(10.0)

            #world.world.on_tick(lambda world_snapshot: do_something(world_snapshot))
            #print(world_snapshot.frame_count,'world frame')



            if not world_snapshot:
                continue

            while world_snapshot.frame_count!=depth.frame_n or world_snapshot.frame_count!=segment.frame_n:
                #print('True')
                time.sleep(0.05)
            #print('Made it')

            control = agent.run_step(options_dict['debug'])
            vehicle.vehicle.apply_control(control)

            

            current_location = vehicle.vehicle.get_location()

            if current_location.distance(prev_location) <= 0.0 and current_location.distance(destination_point) <= 10:
                print('distance from destination: ', current_location.distance(destination_point))
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
