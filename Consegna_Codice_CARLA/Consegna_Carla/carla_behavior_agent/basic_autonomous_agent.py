#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from behavior_agent import BehaviorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import importlib
from misc import compute_distance, get_speed

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

import json
from utils import Streamer

def get_entry_point():
    return 'MyTeamAgent'

class MyTeamAgent(AutonomousAgent):


    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False
    _dict_vehicles_id_counters = dict()

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._agent = None

        self._my_flag = False

        self._dict_vehicles_id_counters = dict()
        
        with open(path_to_conf_file, "r") as f:
            self.configs = json.load(f)
            f.close()
        
        self.__show = len(self.configs["Visualizer_IP"]) > 0
        
        if self.__show:
            self.showServer = Streamer(self.configs["Visualizer_IP"])

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """
        return self.configs["sensors"]
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord
        self._global_plan = global_plan_gps

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation. 
        """
        if not self._agent:
            
            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    # To show telemetry information about the ego-vehicle
                    # hero_actor.show_debug_telemetry(True)

            if not hero_actor:
                return carla.VehicleControl()
        
            # This is to get the plan of the ego-vehicle as a list of tuples. Each tuple is a tuple of (Waypoint, RoadOption)
            plan = [(CarlaDataProvider.get_map().get_waypoint(x[0].location),x[1]) for x in self._global_plan_world_coord]

            # Snippet to calculate the average distance between two waypoints 
            """
            sum = 0
            i = 0
            while i < len(plan)-1:
                sum += compute_distance(plan[i][0].transform.location, plan[i+1][0].transform.location)
                i += 1
            avg = sum/len(plan)
            print(f"/n/n/n COORDINATE DEL MONDO {plan}, la media è {avg} /n/n/n")
            """


            
            # Creating an instance of BehaviorAgent
            self._agent = BehaviorAgent(hero_actor, opt_dict=self.configs, path_waypoints = plan)

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        else:
            spectator = CarlaDataProvider.get_world().get_spectator()
            
            # Define the location and rotation for the Spectator camera
            camera_location = carla.Location(x=0, y=-3, z=4)  # Adjust the coordinates as needed
            camera_rotation = carla.Rotation(pitch=-10, yaw=0, roll=0)  # Adjust the angles as needed

            vehicle_location = self._agent._vehicle.get_transform().location
            vehicle_rotation = self._agent._vehicle.get_transform().rotation

            spectator_location = carla.Location(x = camera_location.x + vehicle_location.x,
                                                y = camera_location.y + vehicle_location.y, 
                                                z = camera_location.z + vehicle_location.z)
            spectator_rotation = carla.Rotation(pitch = camera_rotation.pitch + vehicle_rotation.pitch,
                                                yaw = camera_rotation.yaw + vehicle_rotation.yaw,
                                                roll = camera_rotation.roll + vehicle_rotation.roll)
            
            # Apply the location and rotation to the Spectator camera
            #spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

            """def dist(v): return v.get_location().distance(self._agent._vehicle.get_location())
            vehicle_list = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
            for v in vehicle_list:
                if v.id == 3697 or v.id == 3720:
                    print(f"La distanza dal veicolo {v} è {dist(v)}")
                    target_transform = v.get_transform()
                    target_forward_vector = target_transform.get_forward_vector() 
                    target_extent = v.bounding_box.extent.x
                    print("Il bounding box del veicolo è ", v.bounding_box)
                    CarlaDataProvider.get_world().debug.draw_box(v.bounding_box, target_transform.rotation)
                    print("la target extent è ", target_extent)
                    target_rear_transform = target_transform
                    target_rear_transform.location -= carla.Location(
                        x=target_extent * target_forward_vector.x,
                        y=target_extent * target_forward_vector.y,
                    )
                    print(target_rear_transform.location)
                    CarlaDataProvider.get_world().debug.draw_string(target_rear_transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=255, b=255), life_time=3.0,
                                       persistent_lines=True)
                #if get_speed(v) == 0:
                    w = CarlaDataProvider.get_map().get_waypoint(v.get_location())
                    #print(f"La lane del veicolo {v} è {w.lane_id}")
                    #print(f"la nostra lane è {CarlaDataProvider.get_map().get_waypoint(self._agent._vehicle.get_location()).lane_id}")
                    CarlaDataProvider.get_world().debug.draw_string(w.transform.location, 'X', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                        persistent_lines=True)
                """

            # Release other vehicles
            
            '''vehicle_list = CarlaDataProvider.get_world().get_actors().filter("vehicle.*")
            for actor in vehicle_list:
                if not('role_name' in actor.attributes and actor.attributes['role_name'] == 'hero'):
                    actor.destroy()'''
            
            

            """
            vehicle_list = CarlaDataProvider.get_world().get_actors().filter("*vehicle.diamondback*")
            vehicle_list1 = [v for v in vehicle_list]

            vehicle_list = CarlaDataProvider.get_world().get_actors().filter("*vehicle.bh*")
            vehicle_list2 = [v for v in vehicle_list]

            vehicle_list = vehicle_list1 + vehicle_list2 

            # Controlling other vehicles on the road to deviate them from the ego-vehicle path
            
            control_steering = carla.VehicleControl()
            control_steering.throttle = 0.5
            control_steering.steer = 0.5
            control_steering.brake = 0.0
            control_steering.hand_brake = False

            control_straight = carla.VehicleControl()
            control_straight.throttle = 0.5
            control_straight.steer = 0.0
            control_straight.brake = 0.0
            control_straight.hand_brake = False

            for actor in vehicle_list:
                if actor.id not in self._dict_vehicles_id_counters:
                    self._dict_vehicles_id_counters[actor.id] = 0
                if not('role_name' in actor.attributes and actor.attributes['role_name'] == 'hero'):
                    if(self._dict_vehicles_id_counters[actor.id] < 40):
                        actor.apply_control(control_steering)
                        self._dict_vehicles_id_counters[actor.id]  += 1
                    else:
                        actor.apply_control(control_straight)
            """

            controls = self._agent.run_step()
            if self.__show:
                self.showServer.send_frame("RGB", input_data["Center"][1])
                self.showServer.send_data("Controls",{ 
                "steer":controls.steer, 
                "throttle":controls.throttle, 
                "brake": controls.brake,
                })

            if len(self.configs["SaveSpeedData"]) > 0:
                with open("team_code/"+self.configs["SaveSpeedData"],"a") as fp:
                    fp.write(str(timestamp)+";\t"+str(input_data["Speed"][1]["speed"] * 3.6)+";\t"+str(self.configs["target_speed"])+"\n")
                    fp.close()
                    
            return controls

    def destroy(self):
        print("DESTROY")
        if self._agent:
            self._agent.reset()
            

