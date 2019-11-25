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

import numpy as np
import cv2
from collections import defaultdict

import carla

IM_WIDTH = 400
IM_HEIGHT = 300
SENSOR_TICK = 0.0
FOV = 120

def sensor_attributes(options_dict):
    IM_WIDTH = options_dict['IM_WIDTH']
    IM_HEIGHT = options_dict['IM_HEIGHT']
    SENSOR_TICK = options_dict['SENSOR_TICK']
    FOV = options_dict['FOV']


class World(object):
    def __init__(self, carla_world):
        self.world = carla_world
        
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.sensor_buffer = {}

        print('Enabling synchronous mode.')
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

    def create_samples(self, start, goal, waypoint_dist = 5, disk_radius = 10, num_yaw = 8):
        print(f'Creating samples {waypoint_dist}m apart with {num_yaw} yaw vaules and neighbors within {disk_radius}m.')

        wp = []
        for mp in self.map.generate_waypoints(waypoint_dist):
            wp.append(mp.transform)

        wp.append(start)
        wp.append(goal)

        self.waypoints = []
        self.neighbors = []

        # for each waypoint wp
        for i, wi in enumerate(wp):
            li = wi.location
            ni = []
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
            
            # add in number of yaw orientations to waypoint list        
            ri = wi.transform.rotation
            for k in range(num_yaw):
                if k == (num_yaw)/2:
                    continue

                self.neighbors.append(ni)

                theta = ri.yaw + k*360/(num_yaw)
                if theta >= 180:
                    theta = theta - 360
                elif theta <= -180:
                    theta = 360 - theta
                self.waypoints.append([li.x, li.y, theta])

    def create_obstacles(self, num_obstacles):
        print(f'Creating {num_obstacles} obstacles.')
        obstacles = 0
        # continue randomly spawning obstacles until desired number reached
        while obstacles < num_obstacles:
            transform = random.choice(self.world.get_map().get_spawn_points())
            transform.rotation.yaw = random.randrange(-180.0, 180.0, 1.0) # get random yaw orientation so vehicles don't line up

            bp = random.choice(self.blueprint_library.filter('vehicle')) # random vehicle for fun

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = self.world.try_spawn_actor(bp, transform)
            if npc is not None:
                self.actor_list.append(npc)
                obstacles += 1            

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
        self.world.actor_list.append(self.vehicle) # add to actor_list of world so we can clean up later



class Camera(object):

    def __init__(self, sensor_bp, transform, parent_actor, agent):
        self.vehicle = parent_actor
        self.camera_transform = transform
        self.world = self.vehicle.world
        self.agent = agent
        self.typeofCamera = sensor_bp
        self.frame_n = 0

        bp = self.world.blueprint_library.find(sensor_bp)
        bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        bp.set_attribute('sensor_tick', f'{SENSOR_TICK}')
        bp.set_attribute('fov', f'{FOV}')

        self.sensor = self.world.world.spawn_actor(bp, self.camera_transform, attach_to=self.vehicle.vehicle)
        
        self.world.actor_list.append(self.sensor) # add to actor_list of world so we can clean up later

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera.callback(weak_self,image))

    @staticmethod
    def callback(weak_self, data):
        self = weak_self()
        if not self:
            return
        ## TODO ##
        # update locatoin, velocity, and obstacle list in agent #

        if self.typeofCamera == "sensor.camera.depth":
            self.process_depth(weak_self,data)

        elif self.typeofCamera == "sensor.camera.semantic_segmentation":
            self.process_segment(weak_self,data)
        else:
            self.process_img(weak_self,data)


    @staticmethod
    def process_depth(weak_self,depthData):
        self = weak_self()
        if not self:
            return
        image = np.array(depthData.raw_data)
        #print(image.shape)
        image = image.reshape((IM_HEIGHT,IM_WIDTH,4))
        image = image[:,:,:3]
        normalized_depth = np.dot(image, [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0

        #print(self.world.sensor_buffer.keys())

        self.world.sensor_buffer['depth'] = normalized_depth
        self.frame_n = depthData.frame_number

        # if depthData.frame_number in self.world.sensor_buffer:
        #     print(depthData.frame_number,' worked!')
        #     self.world.sensor_buffer[str(depthData.frame_number)] = normalized_depth
        #     del self.world.sensor_buffer[depthData.frame_number]

        # in_meters_arr = np.zeros((normalized_depth.shape[0],normalized_depth.shape[1],1))

        # in_meters_arr[:,:,0] = normalized_depth



        
        # cv2.imwrite('./out/DEPTH_'+str(depthData.frame_number)+'.png', image)
        # cv2.imshow("Depth",in_meters_arr)
        # cv2.waitKey(1)


    @staticmethod
    def process_img(weak_self,imageData):
        self = weak_self()
        if not self:
            return

        image = np.array(imageData.raw_data)
        image = image.reshape((IM_HEIGHT,IM_WIDTH,4))
        image = image[:,:,:3]

        # print('IMG_'+str(imageData.frame_number)+'.png')
        # cv2.imwrite('IMG_'+str(imageData.frame_number)+'.png', image)

        #print(image.shape)
        #cv2.imshow("Camera",image)
        #cv2.waitKey(1)

    @staticmethod
    def process_segment(weak_self,segmentData):
        self = weak_self()
        if not self:
            return
            # We can do Cuda

        image = np.array(segmentData.raw_data)
        image = image.reshape((IM_HEIGHT,IM_WIDTH,4))
        image = image[:,:,:3]

        bboxes =self.create_bbox(weak_self,image[:,:,2])
        # print('SEG_'+str(segmentData.frame_number)+'.png')


        self.world.sensor_buffer['segment'] = bboxes
        self.frame_n = segmentData.frame_number


        #vis_img = self.trans_vis_segment(image)



        # if 10 in bboxes.keys():

        #     for i in bboxes[10]:
        #          start_point = (i[1],i[0])
        #          end_point = (i[3],i[2])
        #          vis_img = cv2.rectangle(vis_img, start_point, end_point, (0, 0, 142), 2)
            

        #print('SEG_'+str(segmentData.frame_number)+'.png')
        # cv2.imwrite('./out/SEG_'+str(segmentData.frame_number)+'.png', vis_img)
        #cv2.imshow("Camera",image)
        #cv2.waitKey(1)


    @staticmethod
    def create_bbox(weak_self,segment_map):
        self = weak_self()
        if not self:
            return

        bbox_global = {}
        visited_global = {}

        size_x = segment_map.shape[0]
        size_y = segment_map.shape[1]

        for i in range(segment_map.shape[0]):
            for j in range(segment_map.shape[1]):
                if (i,j) in visited_global.keys() or segment_map[i,j] != 10:
                    continue


                visited_global[(i,j)] = True
                visited={}
                #print(i)
                queue = []
                #global_val = segment_map[i,j]
                
                queue.append((i,j))

                visited[(i,j)] = True
                #print(queue)
                while len(queue) >0:
                    #print(len(queue))
                    ind = queue.pop(0)
                    
                    #print(ind)
                    #print(ind)
                    val = segment_map[ind[0],ind[1]]

                    if ind[0]+1<size_x and ind[1]-1<size_y and ind[0]+1>=0 and ind[1]-1>=0 and segment_map[ind[0]+1,ind[1]-1] == val and (ind[0]+1,ind[1]-1) not in visited.keys() and (ind[0]+1,ind[1]-1) not in visited_global.keys():
                        queue.append((ind[0]+1,ind[1]-1))
                        visited[(ind[0]+1,ind[1]-1)] = True
                        visited_global[(ind[0]+1,ind[1]-1)] = True

                    if ind[0]+1<size_x and ind[1]<size_y and ind[0]+1>=0 and ind[1]>=0 and segment_map[ind[0]+1,ind[1]] == val and (ind[0]+1,ind[1]) not in visited.keys() and (ind[0]+1,ind[1]) not in visited_global.keys():
                        queue.append((ind[0]+1,ind[1]))
                        visited[(ind[0]+1,ind[1])] = True
                        visited_global[(ind[0]+1,ind[1])] = True

                    if ind[0]+1<size_x and ind[1]+1<size_y and ind[0]+1>=0 and ind[1]+1>=0 and segment_map[ind[0]+1,ind[1]+1] == val and (ind[0]+1,ind[1]+1) not in visited.keys() and (ind[0]+1,ind[1]+1) not in visited_global.keys():
                        queue.append((ind[0]+1,ind[1]+1))
                        visited[(ind[0]+1,ind[1]+1)] = True
                        visited_global[(ind[0]+1,ind[1]+1)] = True

                    if ind[0]<size_x and ind[1]-1<size_y and ind[0]>=0 and ind[1]-1>=0 and segment_map[ind[0],ind[1]-1] == val and (ind[0],ind[1]-1) not in visited.keys() and (ind[0],ind[1]-1) not in visited_global.keys():
                        queue.append((ind[0],ind[1]-1))
                        visited[(ind[0],ind[1]-1)] = True
                        visited_global[(ind[0],ind[1]-1)] = True

                    if ind[0]<size_x and ind[1]+1<size_y and ind[0]>=0 and ind[1]+1>=0 and segment_map[ind[0],ind[1]+1] == val and (ind[0],ind[1]+1) not in visited.keys() and (ind[0],ind[1]+1) not in visited_global.keys():
                        queue.append((ind[0],ind[1]+1))
                        visited[(ind[0],ind[1]+1)] = True
                        visited_global[(ind[0],ind[1]+1)] = True

                    if (ind[0]-1<size_x) and (ind[1]-1<size_y) and (ind[0]-1>=0) and (ind[1]-1>=0) and (segment_map[ind[0]-1,ind[1]-1] == val) and ((ind[0]-1,ind[1]-1) not in visited.keys()) and ((ind[0]-1,ind[1]-1) not in visited_global.keys()):
                        #print('here 1')
                        queue.append((ind[0]-1,ind[1]-1))
                        visited[(ind[0]-1,ind[1]-1)] = True
                        visited_global[(ind[0]-1,ind[1]-1)] = True

                    if (ind[0]-1<size_x) and (ind[1]<size_y) and (ind[0]-1>=0) and (ind[1]>=0) and (segment_map[ind[0]-1,ind[1]] == val) and ((ind[0]-1,ind[1]) not in visited.keys()) and ((ind[0]-1,ind[1]) not in visited_global.keys()):
                        #print('here 2')
                        queue.append((ind[0]-1,ind[1]))
                        visited[(ind[0]-1,ind[1])] = True
                        visited_global[(ind[0]-1,ind[1])] = True

                    if (ind[0]-1<size_x) and (ind[1]+1<size_y) and (ind[0]-1>=0) and (ind[1]+1>=0) and (segment_map[ind[0]-1,ind[1]+1] == val) and ((ind[0]-1,ind[1]+1) not in visited.keys()) and ((ind[0]-1,ind[1]+1) not in visited_global.keys()):
                        #print('here 3')
                        queue.append((ind[0]-1,ind[1]+1))
                        visited[(ind[0]-1,ind[1]+1)] = True
                        visited_global[(ind[0]-1,ind[1]+1)] = True
                
                left = j
                up = i
                right = j
                down = i

                for key in visited.keys():
                    if key[0] < up:
                        up = key[0]
                    elif key[0] > down:
                        down = key[0]

                    if key[1] < left:
                        left = key[1]
                    elif key[1] > right:
                        right = key[1]

                if val in bbox_global.keys():
                    bbox_global[val].append([up,left,down,right])
                else:
                    bbox_global[val] = [[up,left,down,right]]

                
        return bbox_global

    @staticmethod
    def trans_vis_segment(image):

        idx_build  = np.where(image[:,:,2] == 1)
        image[idx_build[0],idx_build[1],:] = np.array([70,70,70])

        idx_fence  = np.where(image[:,:,2] == 2)
        image[idx_fence[0],idx_fence[1],:] = np.array([190,153,153])

        idx_other  = np.where(image[:,:,2] == 3)
        image[idx_other[0],idx_other[1],:] = np.array([250, 170, 160])

        idx_ped  = np.where(image[:,:,2] == 4)
        image[idx_ped[0],idx_ped[1],:] = np.array([220, 20, 60])

        idx_pole  = np.where(image[:,:,2] == 5)
        image[idx_pole[0],idx_pole[1],:] = np.array([153, 153, 153])

        idx_roadline  = np.where(image[:,:,2] == 6)
        image[idx_roadline[0],idx_roadline[1],:] = np.array([157, 234, 50])

        idx_road  = np.where(image[:,:,2] == 7)
        image[idx_road[0],idx_road[1],:] = np.array([128,64,128])

        idx_side  = np.where(image[:,:,2] == 8)
        image[idx_side[0],idx_side[1],:] = np.array([244, 35, 232])

        idx_veg  = np.where(image[:,:,2] == 9)
        image[idx_veg[0],idx_veg[1],:] = np.array([107, 142, 35])

        idx_car  = np.where(image[:,:,2] == 10)
        image[idx_car[0],idx_car[1],:] = np.array([0, 0, 142])

        idx_wall = np.where(image[:,:,2] == 11)
        image[idx_wall[0],idx_wall[1],:] = np.array([102, 102, 156])

        idx_traf = np.where(image[:,:,2] == 12)
        image[idx_traf[0],idx_traf[1],:] = np.array([220, 220, 0])
        return image



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
        # update locatoin, velocity, and obstacle list in agent #


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
