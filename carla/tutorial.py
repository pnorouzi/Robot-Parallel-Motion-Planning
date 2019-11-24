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

import carla

import random
import time

import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector
from agents.tools.misc import draw_waypoints

def get_topology(mp):
    """
    Accessor for topology.
    This function retrieves topology from the server as a list of
    road segments as pairs of waypoint objects, and processes the
    topology into a list of dictionary objects.

    return: list of dictionary objects with the following attributes
            entry   -   waypoint of entry point of road segment
            entryxyz-   (x,y,z) of entry point of road segment
            exit    -   waypoint of exit point of road segment
            exitxyz -   (x,y,z) of exit point of road segment
            path    -   list of waypoints separated by 1m from entry
                        to exit
    """
    topology = []
    # Retrieving waypoints to construct a detailed topology
    for segment in mp.get_topology():
        wp1, wp2 = segment[0], segment[1]
        l1, l2 = wp1.transform.location, wp2.transform.location
        # Rounding off to avoid floating point imprecision
        x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
        wp1.transform.location, wp2.transform.location = l1, l2
        seg_dict = dict()
        seg_dict['entry'], seg_dict['exit'] = wp1, wp2
        seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
        seg_dict['path'] = []
        endloc = wp2.transform.location
        if wp1.transform.location.distance(endloc) > 1:
            w = wp1.next(1)[0]
            while w.transform.location.distance(endloc) > 1:
                seg_dict['path'].append(w)
                w = w.next(1)[0]
        else:
            seg_dict['path'].append(wp1.next(1/2.0)[0])
        topology.append(seg_dict)
    return topology

def build_graph(topology):
    """
    This function builds a networkx  graph representation of topology.
    The topology is read from self._topology.
    graph node properties:
        vertex   -   (x,y,z) position in world map
    graph edge properties:
        entry_vector    -   unit vector along tangent at entry point
        exit_vector     -   unit vector along tangent at exit point
        net_vector      -   unit vector of the chord from entry to exit
        intersection    -   boolean indicating if the edge belongs to an
                            intersection
    return      :   graph -> networkx graph representing the world map,
                    id_map-> mapping from (x,y,z) to node id
                    road_id_to_edge-> map from road id to edge in the graph
    """
    graph = nx.DiGraph()
    id_map = dict() # Map with structure {(x,y,z): id, ... }
    road_id_to_edge = dict() # Map with structure {road_id: {lane_id: edge, ... }, ... }

    for segment in topology:
        entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
        path = segment['path']
        entry_wp, exit_wp = segment['entry'], segment['exit']
        intersection = entry_wp.is_intersection
        road_id, lane_id = entry_wp.road_id, entry_wp.lane_id

        for vertex in entry_xyz, exit_xyz:
            # Adding unique nodes and populating id_map
            if vertex not in id_map:
                new_id = len(id_map)
                id_map[vertex] = new_id
                graph.add_node(new_id, vertex=vertex)
        n1 = id_map[entry_xyz]
        n2 = id_map[exit_xyz]
        if road_id not in road_id_to_edge:
            road_id_to_edge[road_id] = dict()
        road_id_to_edge[road_id][lane_id] = (n1, n2)

        # Adding edge with attributes
        graph.add_edge(
            n1, n2,
            length=len(path) + 1, path=path,
            entry_waypoint=entry_wp, exit_waypoint=exit_wp,
            entry_vector=vector(
                entry_wp.transform.location,
                path[0].transform.location if len(path) > 0 else exit_wp.transform.location),
            exit_vector=vector(
                path[-1].transform.location if len(path) > 0 else entry_wp.transform.location,
                exit_wp.transform.location),
            net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
            intersection=intersection, type=RoadOption.LANEFOLLOW)

    return graph, id_map, road_id_to_edge


def main():
    actor_list = []

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        # print('Changing world to Town 7')
        # client.load_world('Town07') 

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()
        mp = world.get_map()

        # wp = mp.generate_waypoints(5)
        # print(len(wp))

        # disk_radius = 10
        # num_yaw = 4
        # waypoint_dist = 5

        # print(f'creating samples {waypoint_dist}m apart with {num_yaw} yaw vaules and neighbors within {disk_radius}m.')


        # waypoints = []
        # neighbors = []

        # for i, wi in enumerate(wp):
        #     li = wi.transform.location
        #     ni = []
        #     for j, wj in enumerate(wp):
        #         lj = wj.transform.location
        #         if li == lj:
        #             continue
        #         elif li.distance(lj) <= disk_radius:
        #             for k in range(num_yaw):
        #                 ni.append(j*num_yaw + k)
            
        #     ri = wi.transform.rotation
        #     for k in range(num_yaw):
        #         neighbors.append(ni)

        #         ri.yaw = ri.yaw + k*360/num_yaw
        #         if ri.yaw >= 360:
        #             ri.yaw = ri.yaw - 360
        #         waypoints.append([li.x, li.y, ri.yaw])

        # print(len(waypoints), len(neighbors))

        # draw_waypoints(world, wp) 

        # spawn vehicle
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = 'model3'
        vehicle_transform = spawn_points[0]

        bp = world.get_blueprint_library().filter(vehicle_bp)[0]
        print(bp)
        vehicle = world.spawn_actor(bp, vehicle_transform)
        print(vehicle)
        actor_list.append(vehicle)


        time.sleep(5)

    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()
