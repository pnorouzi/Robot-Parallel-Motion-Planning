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

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()
        mp = world.get_map()

        topology = get_topology(mp)
        G, id_map, road_id_to_edge = build_graph(topology)
        # plt.subplot(121)

        nx.draw(G) #, with_labels=True, font_weight='bold')
        # plt.subplot(122)

        # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
        plt.show()
        plt.savefig("path.png")

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        vehicle_bp = blueprint_library.filter('model3')[0]

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        vehicle_transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        # camera_bp = [blueprint_library.find('sensor.camera.rgb'), blueprint_library.find('sensor.lidar.ray_cast')]
        # camera_transform = [carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)), carla.Transform(carla.Location(x=1.5, z=2.4))]
        # for i, sensor in enumerate(camera_bp):
        #     camera = world.spawn_actor(camera_bp[i], camera_transform[i], attach_to=vehicle)
        #     actor_list.append(camera)
        #     print('created %s' % camera.type_id)
        #     camera.listen(lambda image: image.save_to_disk('_out/%08d' % image.frame_number))

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # converting the pixels to gray-scale.
        # cc = carla.ColorConverter.LogarithmicDepth
        

        time.sleep(5)

    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()
