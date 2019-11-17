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

import random
import time
import threading
import weakref

import carla

IM_WIDTH = 400
IM_HEIGHT = 300
SENSOR_TICK = 0.2

def sensor_attributes(options_dict):
    IM_WIDTH = options_dict['IM_WIDTH']
    IM_HEIGHT = options_dict['IM_HEIGHT']
    SENSOR_TICK = options_dict['SENSOR_TICK']


class World(object):
    def __init__(self, carla_world):
        self.world = carla_world
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

    def create_obstacles(self, num_obstacles):
        obstacles = 0
        while obstacles < num_obstacles:
            transform = random.choice(self.world.get_map().get_spawn_points())
            transform.rotation.yaw = random.randrange(-180.0, 180.0, 1.0)

            bp = random.choice(self.blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = self.world.try_spawn_actor(bp, transform)
            if npc is not None:
                self.actor_list.append(npc)
                obstacles += 1
                # print('obstacle created %s' % npc.type_id)            

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('actors destroyed')


class Car(object):
    def __init__(self, vehicle_bp, transform, carla_world):
        self.world = carla_world

        bp = self.world.blueprint_library.filter(vehicle_bp)[0]
        self.vehicle_transform = transform
        self.vehicle = self.world.world.spawn_actor(bp, self.vehicle_transform)
        self.world.actor_list.append(self.vehicle)


class Camera(object):
    def __init__(self, sensor_bp, transform, parent_actor, agent):
        self.vehicle = parent_actor
        self.camera_transform = transform
        self.world = self.vehicle.world
        self.agent = agent

        bp = self.world.blueprint_library.find(sensor_bp)
        bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        bp.set_attribute('sensor_tick', f'{SENSOR_TICK}')

        self.sensor = self.world.world.spawn_actor(bp, self.camera_transform, attach_to=self.vehicle.vehicle)
        
        self.world.actor_list.append(self.sensor)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera.callback(weak_self,image))

    @staticmethod
    def callback(weak_self, data):
        self = weak_self()
        if not self:
            return
        ## TODO ##
        # pass in agent to update location and velocity as well as obstacle list #


class Lidar(object):
    def __init__(self, sensor_bp, transform, parent_actor, agent):
        self.vehicle = parent_actor
        self.camera_transform = transform
        self.world = self.vehicle.world
        self.agent = agent

        bp = self.world.blueprint_library.find(sensor_bp)
        bp.set_attribute('sensor_tick', f'{SENSOR_TICK}')

        self.sensor = self.world.world.spawn_actor(bp, self.camera_transform, attach_to=self.vehicle.vehicle)
        
        self.world.actor_list.append(self.sensor)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Lidar.callback(weak_self,image))

    @staticmethod
    def callback(weak_self, data):
        self = weak_self()
        if not self:
            return
        ## TODO ##
        # pass in agent to update location and velocity as well as obstacle list #


def test():
    world = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        world = World(client.get_world())

        vehicle_bp = 'model3'
        vehicle_transform = random.choice(world.map.get_spawn_points())

        vehicle = Car(vehicle_bp, vehicle_transform, world)

        
        camera_bp = ['sensor.camera.rgb', 'sensor.camera.rgb', 'sensor.lidar.ray_cast']
        camera_transform = [carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=40)), carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15, yaw=-40)), carla.Transform(carla.Location(x=1.5, z=2.4))]

        cam1 = Camera(camera_bp[0], camera_transform[0], vehicle)
        cam2 = Camera(camera_bp[1], camera_transform[1], vehicle)
        lidar = Lidar(camera_bp[2], camera_transform[2], vehicle)


        time.sleep(1)

    finally:

        if world is not None:
            world.destroy()


if __name__ == '__main__':
    test()
