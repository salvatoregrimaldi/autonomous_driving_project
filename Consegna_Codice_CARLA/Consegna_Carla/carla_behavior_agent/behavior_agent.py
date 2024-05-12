# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>. 


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
import heapq as hq
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from shapely.geometry import Polygon, Point
import copy
import math
from time import time


from misc import get_speed, positive, is_within_distance, compute_distance, compute_length, is_bike, get_trafficlight_trigger_location

ZERO_SPEED_TH = 0.01            # Speed under which the vehicle is considered to be stopped

JUNCTION_DETECTION_DISTANCE = 13

FIXED_TIME_STEP = 0.05
STOP_SIGN_SECONDS = 3

STOP_SIGN_TYPE = "206"
LANE_NARROWING_CONE_THRESHOLD = 8
LANE_NARROWING_LATERAL_OFFSET = 0.9

SAMPLING_RESOLUTION = 1         # Average euclidean distance between two consecutive waypoints in the global path
MAX_OBSTACLE_PROXIMITY = 100     # Distance under wich a static object on the road is considered a potential obstacle
MAX_VEHICLE_PROXIMITY = 30      # Distance under wich a vehicle on the road is considered a potential obstacle
MAX_WALKER_PROXIMITY = 9
WALKER_DISTANCE_PROXIMITY = 35
MAX_CROSSING_PROXIMITY = 9
MAX_OTHER_LANE_PROXIMITY = 300  # Max distance of view for the vehicles coming from the opposite lane 

MEAN_STOP_ACCELERATION = 20.4
OBSTACLE_STOP_DISTANCE = 12

OVERTAKE_MARGIN = 5            # Security margin in terms of overtake space
OVERTAKE_TIME_MARGIN = 3      # Security margin in terms of overtake time
OTHER_LANE_DETECTION_MARGIN = 10

BIKE_OVERTAKE_MINUMUM_ANGLE = 40
START_BIKE_OVERTAKE_DISTANCE = 20
BIKE_OVERTAKE_LATERAL_OFFSET = 0.9

OVERTAKE_DETECTED = 1
OVERTAKE_EVALUATION = 2
OVERTAKE_WAITING = 3   
OVERTAKE_IN_PROGRESS = 4
OVERTAKE_NOT_DETECTED = 0

JUNCTION_DETECTED = 1
JUNCTION_CHECK_STOP = 2
JUNCTION_EVALUATION = 3
JUNCTION_WAITING = 4   
JUNCTION_IN_PROGRESS = 5
JUNCTION_NOT_DETECTED = 0
JUNCTION_APPROCHING = 6
JUNCTION_BIG_STEP = 7

JUNCTION_TIME_MARGIN = 0.0

OVERTAKE_COUNTER = 20

OVERTAKE_HORIZON = 3        # Number of waypoints to look ahead after an overtake

NO_TRAFFIC_LIGHT = 0
TRAFFIC_LIGHT_OK = 1
TRAFFIC_LIGHT_NOT_OK = -1

BOUNDING_BOX_PROJECTION = 1.5


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """
    
    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None, path_waypoints= None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # For overtake purposes
        self.standard_controller = self._local_planner._args_lateral_dict

        # Bike overtake behaviour
        self.overtaking_bike = False
        self.last_bike = None

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        # Other vehicle information
        self._obstacle_detected = None
        self._overtake_counter = 0
        self._overtake_last_vehicle = None
        self._overtake_t1 = None
        self._initial_speed = 0
        self._sum_acceleration = 0
        self._overtake_state = OVERTAKE_NOT_DETECTED                 # It manages different stages of a lane changing
        self._path_waypoints = path_waypoints   # List of tuples. Each tuple is a tuple of (Waypoint, RoadOption)
        self._restart_index = None              # The index of the waypoint from which resuming the original path:
                                                # there could be deviations from the original path due to obastacles
        self._last_overtake_waypoint = None

        self._last_stop_sign = None
        self._last_stop_sign_count = 0
        self._stopped_at_stop_sign = False

        #self.at_junction = False
        #self.last_change_wp = None
        self._junction_state = JUNCTION_NOT_DETECTED 
        self._junction_detected = False
        self._junction_reached = False
        self._first_junction_wp = None
        self._junction_speed_sum = 0 
        self._junction_speed_counter = 0
        self._last_transform = None
        self._step_counter = 0
        self._previous_right_junction = None
        self._previous_right_junction_distance = None
        self._previous_left_junction = None
        self._previous_left_junction_distance = None


        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        #self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self, debug = False):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        #def dist(l): return l.get_location().distance(wp.transform.location)
        #lights_list = [l for l in lights_list if dist(l) < 30]

        affected, tf = self._affected_by_traffic_light(lights_list)

        if tf is None:
            return NO_TRAFFIC_LIGHT

        else: 
            if debug:
                trigger_location = get_trafficlight_trigger_location(tf)
                s1 = "TL" + str(tf.id)
                self._world.debug.draw_string(tf.get_location(), s1, draw_shadow=False,
                        color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                        persistent_lines=True)
                self._world.debug.draw_string(trigger_location, s1, draw_shadow=False,
                        color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                        persistent_lines=True)
                
            if tf.state == carla.TrafficLightState.Green:
                return TRAFFIC_LIGHT_OK
        return TRAFFIC_LIGHT_NOT_OK



    def stop_sign_manager(self, wp, max_distance = 8):
        """
        This method is in charge of behavior when encountering a stop sign.
        """

        if self._stopped_at_stop_sign and self._last_stop_sign_count <  STOP_SIGN_SECONDS / FIXED_TIME_STEP :
            print("Sto aspettando davanti allo stop")
            #print(str(self._last_stop_sign) +"   " +str(self._last_stop_sign_count))
            self._last_stop_sign_count += 1 
            return True

        self._last_stop_sign_count = 0
        self._stopped_at_stop_sign = False

        stop_signs_list = wp.get_landmarks_of_type(distance = max_distance, type = STOP_SIGN_TYPE, stop_at_junction = False)
        stop_signs = [(l, l.distance) for l in stop_signs_list]
        stop_signs.sort(key=lambda a: a[1])
        #stop_signs = wp.get_landmarks(distance = max_distance)
        if len(stop_signs) != 0:
            for s in stop_signs:
                s = s[0]
                if s.name == "Stencil_STOP":
                    continue
                
                #print(f"*****Landmark name: {s.name}, Landmark distance: {s.distance}")

                trigger_wp = s.waypoint

                ego_vehicle_location = self._vehicle.get_location()
                ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

                #if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                    #print("condizione road_id non soddisfatta")
                    #continue

                if s.id == self._last_stop_sign:
                    continue
                

                ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
                wp_dir = trigger_wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                if dot_ve_wp < 0:
                    #print("condizione prodotto scalare NON soddisfatta")
                    continue
                
                #print("Ho individuato un nuovo stop davanti al quale fermarmi")
                self._last_stop_sign = s.id
                self._stopped_at_stop_sign = True
                return True
        
        return False

    def junction_crossing_manager(self, wp, distance = 15):
        """
        This method is in charge of detecting a junction and determine if it has already been crossed or not:
        returns True if it has been detected but not completely crossed, False when not detected or already crossed
        """
        #print("JUNCTION DETECTED: ", self._junction_detected)
        #print("JUNCTION REACHED: ", self._junction_reached)
        #print("IN JUNCTION: ", wp.is_junction)
        if not self._junction_detected and not self._junction_reached:
            next_road_wp = self._local_planner.get_incoming_waypoint_and_direction(steps=distance)[0]
            if not next_road_wp: return False
            #print("Next WP Junction: ", next_road_wp.is_junction)
            #print(next_road_wp)
            
            '''self._world.debug.draw_string(next_road_wp.transform.location, 'X', draw_shadow=False,
                        color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                        persistent_lines=True)'''

            if next_road_wp.is_junction and not wp.is_junction:
                self._junction_detected = True
                self._junction_reached = False
                return True
            return False
        elif self._junction_detected and not self._junction_reached:
            if wp.is_junction:
                self._junction_reached = True
            return True
        else: #both flags true
            if not wp.is_junction :
                self._junction_detected = False 
                self._junction_reached = False
                return False
            return True



    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)
                                         

    def lane_narrowing_manager(self, waypoint, search_distance = 50):
        static_list = self._world.get_actors().filter('*static.prop.constructioncone*')
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        cone_list = [(dist(s),s) for s in static_list if dist(s) < search_distance and s.id != self._vehicle.id]           #get vehicles in the other lane, in the radius of distance
        #print("len cone_list = ", len(cone_list))
        return len(cone_list) >= LANE_NARROWING_CONE_THRESHOLD

        
        #for s in list:
        #    if self._map.get_waypoint(s[1].get_location()
        #    ).lane_id != waypoint.lane_id:
        #        print(f"**************Oggetto sull'altra lane:{s}") 
        #self.local_planner._vehicle_controller._lat_controller.set_offset(1)
         

    def check_blacklist(self, v):
        if (v.id != 3711 and v.type_id != "vehicle.nissan.patrol_2021") and \
            (v.id != 3731 and v.type_id != "vehicle.mini.cooper_s") and \
            (v.type_id != 'static.prop.dirtdebris01'):
            return True
        return False

    def collision_and_obstacle_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision with obstacles
        (e.g. static objects on the road)

            :param waypoint: current waypoint of the agent
            :return object_state: True if there is a obstacle nearby, False if not
            :return target_object: nearby obstacle
            :return obj_distance: distance to nearby obstacle 
        """
        static_list = self._world.get_actors().filter('*static*')
        vehicle_list = self._world.get_actors().filter('*vehicle*')
        # Take only static objects within a certain distance on the road
        temp_list = [s for s in static_list if (self.check_blacklist(s))]
        temp_list1 = [v for v in vehicle_list if (self.check_blacklist(v))]

        def dist(v): return v.get_location().distance(waypoint.transform.location)
        #object_list = [(dist(v),v) for v in object_list if dist(v) < MAX_OBSTACLE_PROXIMITY and v.id != self._vehicle.id]
        object_list = [(dist(s),s) for s in temp_list + temp_list1 if dist(s) < MAX_OBSTACLE_PROXIMITY and s.id != self._vehicle.id]
        if len(object_list) == 0:
            return (False, None, None)
        object_list.sort(key=lambda a: a[0])
        #hq.heapify(object_list)

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        
        target_object_list = []
        obj_distance_list = []

        if self._direction == RoadOption.CHANGELANELEFT:
            target_object_list, obj_distance_list  = self._object_obstacle_detected(
                object_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)

        elif self._direction == RoadOption.CHANGELANERIGHT:
            target_object_list, obj_distance_list  = self._object_obstacle_detected(
                object_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)

        else:
            target_object_list, obj_distance_list  = self._object_obstacle_detected(object_list, 50, up_angle_th=30)
        
        if len(target_object_list) > 0:
            return True, target_object_list, obj_distance_list
        else: return False, None, None

    def detect_other_lane_vehicle(self):
        """
        This module is in charge of identify the vehicles that are coming from the opposite lane, and are
        going towards us.

        """
        ego_vehicle_loc = self._vehicle.get_location()
        ego_waypoint = self._map.get_waypoint(ego_vehicle_loc)

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(ego_waypoint.transform.location)
        vehicle_list = [(dist(v),v) for v in vehicle_list if dist(v) < MAX_OTHER_LANE_PROXIMITY and v.id != self._vehicle.id]           #get vehicles in the other lane, in the radius of distance
        vehicle_list = [v for v in vehicle_list if self._map.get_waypoint(v[1].get_location()).lane_id != ego_waypoint.lane_id]
        ahead_list = []
        # iterate through the list of vehicle to select only those ahead of us on the opposite lane
        ego_vehicle_loc = ego_waypoint.transform.location
        ego_forward_vector = self._vehicle.get_transform().rotation.get_forward_vector()
        
        for v in vehicle_list:
            target_vehicle_loc = v[1].get_location()
            # computes a vector from the ego vehicle to the other
            ego_target_vector = target_vehicle_loc - ego_vehicle_loc
            ego_target_vector = carla.Vector3D(x=ego_target_vector.x, y=ego_target_vector.y, z=ego_target_vector.z)
            
        
            # if the dotprod between the front vector of the ego veichle and the other vector is > 0 then the vehichle is ahead us,
            # otherwise the vehichle is behind us
            if ego_forward_vector.get_vector_angle(ego_target_vector) *180/math.pi < 90 + OTHER_LANE_DETECTION_MARGIN:
                ahead_list.append(v) 

            #if ego_waypoint.transform.rotation.get_forward_vector().make_unit_vector().dot(vector_3d) >  math.cos(90 + OVERTAKE_MINUMUM_ANGLE):
        # Sort according to the distance
        ahead_list.sort(key=lambda a: a[0])
        return ahead_list
    
    def crossing_object_detector(self, wp):
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(wp.transform.location)
        vehicle_list = [(dist(v),v) for v in vehicle_list if is_bike(v) and dist(v) < MAX_CROSSING_PROXIMITY]
        if len(vehicle_list) == 0:
            return False
        vehicle_list.sort(key=lambda a: a[0])
        v = vehicle_list[0][1]

        ego_forward_vector = wp.transform.rotation.get_forward_vector()
        target_forward_vector = v.get_transform().rotation.get_forward_vector()
        #target_location = v.bounding_box.location
        #target_forward_vector = carla.Vector3D(x = target_forward_vector.x + target_location.x,
        #                                        y = target_forward_vector.y + target_location.y,
        #                                        z = target_forward_vector.z + target_location.z) 


        #print("Targer_forward_vector", target_forward_vector)
        #print("Targer_location", target_location)
        #print("ego forward vector", ego_forward_vector)
        #print("Ego location", wp.transform.location)

        #self._world.debug.draw_line(self, target_location, second_location, thickness=0.1, color=(255,0,0), life_time=-1.0)

        #print("L'angolo con la bici che attraversa: ", ego_forward_vector.get_vector_angle(target_forward_vector) * 180 / math.pi)
        if 80 < (ego_forward_vector.get_vector_angle(target_forward_vector)) * 180 / math.pi < 100:
            return True
        return False

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        # Take only vehicles within a certain distance
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [(dist(v), v) for v in vehicle_list if dist(v) < MAX_VEHICLE_PROXIMITY and v.id != self._vehicle.id]

        if len(vehicle_list) == 0:
            return (False, None, -1)
        vehicle_list.sort(key=lambda a: a[0])
        #print("Vehicle List:", vehicle_list)
        vehicle_list = [v[1] for v in vehicle_list]

        ego_vehicle_loc = self._vehicle.get_location()
        #ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        target_vehicle_list = []
        vehicle_distance_list = []
        
        if self._direction == RoadOption.CHANGELANELEFT:
            target_vehicle_list, vehicle_distance_list = self._vehicle_obstacle_detected_2(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2, MAX_VEHICLE_PROXIMITY), up_angle_th=180, lane_offset=-1)

        elif self._direction == RoadOption.CHANGELANERIGHT:
            target_vehicle_list, vehicle_distance_list = self._vehicle_obstacle_detected_2(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2, MAX_VEHICLE_PROXIMITY), up_angle_th=180, lane_offset=1)

        else:
            #print("Passo questa lista a Vehicle Obstacle Detected 2")
            #for v in vehicle_list: print(v)
            target_vehicle_list, vehicle_distance_list = self._vehicle_obstacle_detected_2(vehicle_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3, MAX_VEHICLE_PROXIMITY), up_angle_th=30)
            
            # Check for tailgating
            if not len(target_vehicle_list) and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        if len(target_vehicle_list) > 0:
            return True, target_vehicle_list, vehicle_distance_list
        else: return False, None, None

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        print("LISTA IN PEDESTRIAN AVOID MANAGER: ")
        for w in walker_list:
            print(w.type_id)
        if len(walker_list)==0: # to fix CARLA bug that consider stationary bicycles as pedestrians
            return (False, None, 0)
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < MAX_WALKER_PROXIMITY]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            #print("SITUAZIONE CRITICA, TTC:", ttc)
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            #print("SITUAZIONE NON TROPPO CRITICA, SEGUO MACCHINA,  TTC:", ttc)
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            #print("SITUAZIONE TRANQUILLA, NON SEGUO NESSUNO,  TTC:", ttc)
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def bike_following_and_overtake_manager(self, bike_list = [], debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if(not self.overtaking_bike):
            self.overtaking_bike = True
            self.last_bike = bike_list[-1]
            print("Overtaking Bike")
            self._local_planner._vehicle_controller._lat_controller.set_offset(-BIKE_OVERTAKE_LATERAL_OFFSET)
            self._local_planner.set_speed(self._behavior.bike_overtake_speed) 

        else:
            ego_vehicle_loc = self._vehicle.get_location()
            ego_rear_loc = carla.Location(x = ego_vehicle_loc.x - self._vehicle.bounding_box.extent.x,  
                                          y = ego_vehicle_loc.y, z = ego_vehicle_loc.z)
            target_vehicle_loc = self.last_bike.get_location()
            # computes a vector from the ego vehicle to the other
            vector3D = target_vehicle_loc - ego_rear_loc
            vector_3d = carla.Vector3D(x=vector3D.x, y=vector3D.y, z=vector3D.z)
            
            #normalize vector
            vector_3d = vector_3d.make_unit_vector()
            
            # if the dotprod between the front vector of the ego veichle and the other vector is > 0 then the vehichle is ahead us,
            # otherwise the vehichle is behind us
            if math.acos(self._vehicle.get_transform().rotation.get_forward_vector().dot(vector_3d)) * 180 / math.pi > 90 + BIKE_OVERTAKE_MINUMUM_ANGLE:
                self.overtaking_bike = False
                self.last_bike = None
                self._local_planner._vehicle_controller._lat_controller.set_offset(0)
                self._local_planner.set_speed(self._behavior.max_speed) 
               
                control = self._local_planner.run_step(debug=debug)
                return control

        control = self._local_planner.run_step(debug=debug)
        return control
            

    def overtake_manager(self, same_direction_road = False, debug = False):
        """
        Module in charge of the overtaking behavior
        """

        obstacle = self._obstacle_detected

        if obstacle is None or obstacle[0] is False:
            return None

        if self._overtake_state == OVERTAKE_DETECTED:
            if get_speed(self._vehicle) > ZERO_SPEED_TH:
                return self.emergency_stop()
            print("**************************Mi sono fermato a distanza: ", compute_distance(obstacle[1][0].get_location(), self._vehicle.get_location()))
            self._overtake_state = OVERTAKE_EVALUATION
            return self.emergency_stop()
        
        elif self._overtake_state == OVERTAKE_EVALUATION:

            if len(obstacle[1]) == 1:
                obstacle_length = max(obstacle[1][0].bounding_box.extent.y, obstacle[1][0].bounding_box.extent.x) * 2
            else:
                residual_first_length =  max(obstacle[1][0].bounding_box.extent.y, obstacle[1][0].bounding_box.extent.x)

                residual_last_length =  max(obstacle[1][-1].bounding_box.extent.y, obstacle[1][-1].bounding_box.extent.x) 

                obstacle_length = (obstacle[2][-1] - obstacle[2][0])  + residual_first_length + residual_last_length
                
            overtake_length = obstacle_length + OVERTAKE_MARGIN

            ego_vehicle_loc = self._vehicle.get_location()
            ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

            #if not same_direction_road :
            overtake_path = self._generate_overtake_path(ego_vehicle_wp, direction='left', distance_same_lane=0,
                            distance_other_lane=overtake_length, lane_change_distance=5,
                            check=False, lane_changes=1, step_distance=1, first_half = True)
            
            last_waypoint = overtake_path[-1][0]

            second_overtake_path = self._generate_overtake_path(last_waypoint, direction='left', distance_same_lane=3,
                            distance_other_lane= 2, lane_change_distance=5,
                            check=False, lane_changes=1, step_distance=1, first_half = False)
        
            overtake_path.extend(second_overtake_path)
            
            last_waypoint = overtake_path[-1][0]
            self._last_overtake_waypoint = last_waypoint

            # Finding the waypoint from which resuming the original path after the overtake
            restart_index = -1
            for i, wp in enumerate(self._path_waypoints):
                curr_dist = compute_distance(wp[0].transform.location, last_waypoint.transform.location)
                if curr_dist < SAMPLING_RESOLUTION:
                    restart_index = i
                    break
                
            self._restart_index = restart_index
            self.set_global_plan(overtake_path)
            
            # Draws the overtake path on the map
            for w in overtake_path:
                self._world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=255, b=255), life_time=120.0,
                                       persistent_lines=True)
            
            # Computing the time required for completing the overtake maneuver
            i = 0
            overtake_distance = 0
            while i < len(overtake_path)-1:
                overtake_distance += compute_distance(overtake_path[i][0].transform.location, overtake_path[i+1][0].transform.location)
                i += 1


            velocity_list = [self._behavior.overtake_speed, self._speed_limit - self._behavior.speed_lim_dist] 
            target_speed = min(velocity_list)
            
            overtake_t1 = (overtake_distance / (target_speed / 3.6)) 
            self._local_planner.set_speed(target_speed)
            # print("Lista delle velocita' e' ", velocity_list)
            # print("target speed con cui sorpasso = ", target_speed)

            # Save the time for later
            self._overtake_t1 = overtake_t1            

            self._overtake_state = OVERTAKE_WAITING
            return self.emergency_stop()
            
        elif self._overtake_state == OVERTAKE_WAITING:
            
            # Detecting vehicles on the opposite lane
            ahead_vehicles =  self.detect_other_lane_vehicle()
            if len(ahead_vehicles) == 0:
                self.standard_controller = self._local_planner._args_lateral_dict
                self._local_planner._args_lateral_dict = {'K_V': 3.0, 'K_S': 0.05, 'dt': 0.05}
                self._overtake_state = OVERTAKE_IN_PROGRESS
                return self._local_planner.run_step(debug=debug)
 
            # Computing the time that the closest vehicle on the opposite lane takes to reach a position that could determine a collision with the ego-vehicle
            _, next_vehicle = ahead_vehicles[0]
            if self._overtake_last_vehicle is None or next_vehicle.id != self._overtake_last_vehicle.id:
                self._overtake_last_vehicle = next_vehicle
                self._overtake_counter = 0
                self._initial_speed = get_speed(next_vehicle)  / 3.6
                self._sum_acceleration = next_vehicle.get_acceleration().length() / 3.6
            
            else:
                #print("*********OVERTAKE COUNTER = ", self._overtake_counter)
                self._overtake_counter += 1
                self._initial_speed = get_speed(next_vehicle)  / 3.6
                self._sum_acceleration += next_vehicle.get_acceleration().length() / 3.6
            
            if self._overtake_counter == OVERTAKE_COUNTER-1: 
                #calcola media speed e accellerazione
                next_vehicle_initial_speed = self._initial_speed 
                next_vehicle_acceleration_mean = self._sum_acceleration / OVERTAKE_COUNTER
                
                next_vehicle_location = self._overtake_last_vehicle.get_location()

                #print(f"Il veicolo {next_vehicle.id}, ha velocita' {next_vehicle_initial_speed * 3.6}m/s, e accelarazione {next_vehicle_acceleration_mean}m/s^2")
                
                other_lane_waypoint = self._last_overtake_waypoint.get_left_lane()
                dist = compute_distance(next_vehicle_location, other_lane_waypoint.transform.location)
                overtake_t2 = (-next_vehicle_initial_speed + (math.sqrt(next_vehicle_initial_speed**2 + 2*next_vehicle_acceleration_mean*dist))) / next_vehicle_acceleration_mean  #t2=(-v0+sqrt(v0^2+2as))/a dalla legge oraria del moto unif. acc.
                
                # If the time required for completing the overtake maneuver is smaller than the 
                # time that the closest vehicle on the opposite lane takes to reach a position that could determine a collision with the ego-vehicle,
                # then the overtake maneuver can start
                
                print("T1 + MARGIN: ", self._overtake_t1 + OVERTAKE_TIME_MARGIN)
                print("T2: ", overtake_t2)

                if self._overtake_t1 + OVERTAKE_TIME_MARGIN < overtake_t2 :
                    self.standard_controller = self._local_planner._args_lateral_dict
                    self._local_planner._args_lateral_dict = {'K_V': 3.0, 'K_S': 0.05, 'dt': 0.05}
                    self._overtake_state = OVERTAKE_IN_PROGRESS
                    return self._local_planner.run_step(debug=debug)                
                
            return self.emergency_stop()

        elif self._overtake_state == OVERTAKE_IN_PROGRESS:
            if len(self._local_planner._waypoints_queue) > 0:
                # The overtake is not completed yet
                return self._local_planner.run_step(debug=debug)
            else:
                # The overtake is complete
                self._local_planner._args_lateral_dict = self.standard_controller 

                velocity_list = [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]
                target_speed = min(velocity_list)
                self._local_planner.set_speed(target_speed)

                self._overtake_t1 = None
                self._obstacle_detected = None
                self.set_global_plan(self._path_waypoints[self._restart_index + OVERTAKE_HORIZON:])
                self._restart_index = None
                self._overtake_state = OVERTAKE_NOT_DETECTED
                return self._local_planner.run_step(debug=debug)


        
    def junction_manager(self, wp):
        # Raccolta auto
        
        #print("Stato Junction: ", self._junction_state)

        light_state = self.traffic_light_manager(wp)

        if light_state != NO_TRAFFIC_LIGHT:
            if light_state == TRAFFIC_LIGHT_OK:
                self._junction_state = JUNCTION_IN_PROGRESS
                #print("Il semaforo è verde e sto passando")
                return None
            #print("Il semaforo è rosso e sono fermo")
            return self.emergency_stop()
             
        if self._junction_state == JUNCTION_NOT_DETECTED:
            
            #print("SONO IN JUNCTION NOT DETECTED")
            junction_crossing = self.junction_crossing_manager(wp, distance = JUNCTION_DETECTION_DISTANCE)
            if not junction_crossing:
                #print("Non rilevo junctions")
                return None
            self._junction_state = JUNCTION_DETECTED
            return None
        
        elif self._junction_state == JUNCTION_DETECTED:

            #print("SONO IN JUNCTION DETECTED")
            self._behavior.max_speed = self._behavior.junction_crossing_speed
            if self._first_junction_wp == None:
                i = 0
                while True:
                    next_road_wp = self._local_planner.get_incoming_waypoint_and_direction(steps=i)[0]
                    if next_road_wp.is_junction:
                        self._first_junction_wp = next_road_wp
                        #print("Il primo waypoint del junction ha ID: ", self._first_junction_wp.id)
                        if self._first_junction_wp.id == 15738739296028174304:
                            self._junction_state = JUNCTION_IN_PROGRESS
                            return None
                        break
                    i +=1

            """self._world.debug.draw_string(self._first_junction_wp.transform.location,'WP', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=0), life_time=120.0,
                persistent_lines=True)"""

            distance_from_junction = compute_distance(wp.transform.location, self._first_junction_wp.transform.location)
            if distance_from_junction > 6.5:
                #print("Distanza dal primo wp del junction: ", distance_from_junction)
                return None
            else:
                self._junction_state = JUNCTION_CHECK_STOP
                return self.emergency_stop()
  
        elif self._junction_state == JUNCTION_CHECK_STOP:

            #print("SONO IN JUNCTION JUNCTION_CHECK_STOP")
            if self.stop_sign_manager(wp, max_distance  = JUNCTION_DETECTION_DISTANCE + 5):
                return self.emergency_stop()
            #print("FINITO STOP")
            
            if get_speed(self._vehicle) > ZERO_SPEED_TH:
                return self.emergency_stop()
            self._last_transform = wp.transform
            self._junction_state = JUNCTION_APPROCHING
            #print("FINITO DI ASPETTARE ")
            return self.emergency_stop()
        
        elif self._junction_state == JUNCTION_APPROCHING:
            print("SONO IN JUNCTION APPROCCHING")
            junction = self._first_junction_wp.get_junction()
            junction_center = junction.bounding_box.location
            
            ego_loc = wp.transform.location
            distance = compute_distance(ego_loc, junction_center)
            
            print("La distanza dal centro dell'incrocio e' ", distance)
            if distance > 12:
                return None
            self._junction_state = JUNCTION_WAITING

        elif self._junction_state == JUNCTION_WAITING:
            def polygon_distance(v):
                target_polygon = self.get_polygon(v)
                ego_polygon = self.get_polygon(self._vehicle)
                return target_polygon.distance(ego_polygon)
            
            junction = self._first_junction_wp.get_junction()

            watch_distance = 2 * max(junction.bounding_box.extent.x, junction.bounding_box.extent.y) + 20
            print("Guardo veicoli a distanza massima di ", watch_distance)

            vehicle_list = self._world.get_actors().filter("*vehicle*")
            def last_dist(v): return v.get_location().distance(self._last_transform.location)
            vehicle_list = [(v, last_dist(v)) for v in vehicle_list if not is_bike(v) and last_dist(v) < watch_distance and v.id != self._vehicle.id]
            vehicle_list.sort(key=lambda a: a[1])

            #print("SONO IN FASE DI ATTESA")
            junction_vehicles = []  

            # Check if there are vehicles inside the junction
            for v, _ in vehicle_list:
                if self._map.get_waypoint(v.get_location()).is_junction:
                    #print("L'INCROCIO NON E' SGOMBRO")
                    #print("Il veicolo nel junction va a velocità: ", get_speed(v))
                    self._junction_speed_sum += get_speed(v)
                    self._junction_speed_counter += 1 
                    junction_vehicles.append(v)
                    
            left_vehicles = []
            right_vehicles = []
            front_vehicles =  []        

            ego_location = self._last_transform.location
            ego_right_vector = self._last_transform.rotation.get_right_vector()
            #print("Vettore destro dell'ego:", ego_right_vector)
            ego_forward_vector = self._last_transform.rotation.get_forward_vector()

            ## Distinguere destri e sinistri
            for v, _ in vehicle_list:
                if v in junction_vehicles: continue
                target_location = v.get_location()
                target_forward_vector = v.get_transform().rotation.get_forward_vector()
                ego_target_vector = target_location - ego_location
                ego_target_vector = carla.Vector3D(x = ego_target_vector.x, y = ego_target_vector.y, z = ego_target_vector.z)


                # Check per non considerare i veicoli collocati DIETRO l'ego-veicolo
                if ego_forward_vector.get_vector_angle(ego_target_vector) *180/math.pi > 100:
                    continue
                
                right_vector_angle = ego_right_vector.get_vector_angle(ego_target_vector) *180/math.pi
                #print("right_vector_angle appena calcolato", right_vector_angle)
                #print("Vettore che congiunge il nostro veicolo al target", ego_target_vector.make_unit_vector())

                if 0 < right_vector_angle < 85:
                    # list of vehicles coming from right and oriented towards the junction
                    if ego_right_vector.get_vector_angle(target_forward_vector) *180/math.pi > 170:
                        #print("Veicolo RIGHT Pericoloso angolo uguale a: ", ego_right_vector.get_vector_angle(target_forward_vector) *180/math.pi)
                        right_vehicles.append((v, right_vector_angle))
                
                elif 85 < right_vector_angle < 120:
                    # list of vehicles coming from front and oriented towards the junction
                    if ego_forward_vector.get_vector_angle(target_forward_vector) *180/math.pi > 170:
                        front_vehicles.append((v, right_vector_angle))
                else: 
                    # list of vehicles coming from left and oriented towards the junction
                    if ego_right_vector.get_vector_angle(target_forward_vector) *180/math.pi < 10:
                        #print("Veicolo Left Pericoloso angolo uguale a: ", ego_right_vector.get_vector_angle(target_forward_vector) *180/math.pi)
                        left_vehicles.append((v, right_vector_angle))
            
            #print("\n\n\n")

            """print("SULLA MIA SINISTRA HO:")
            for v, a in left_vehicles:
                self._world.debug.draw_string(v.get_location(), 'X', draw_shadow=False,
                color=carla.Color(r=0, g=255, b=0), life_time=120.0,
                persistent_lines=True)
                print("Auto: ", v.type_id)
                print("Lane Id", self._map.get_waypoint(v.get_location()).lane_id)
                print("Road_id: ", self._map.get_waypoint(v.get_location()).road_id)
                print("Id Auto: ", v.id)
                print("Angle respect ego right vector: ", a)

            
            print("SULLA MIA DESTRA HO:")
            for v, a in  right_vehicles:
                self._world.debug.draw_string(v.get_location(), 'O', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                persistent_lines=True)
                print("Auto: ", v.type_id)
                print("Lane Id", self._map.get_waypoint(v.get_location()).lane_id)
                print("Road_id: ", self._map.get_waypoint(v.get_location()).road_id)
                print("Id Auto: ", v.id)
                print("Angle respect ego right vector: ", a)


            print("DI FRONTE HO:")
            for v, a in  front_vehicles:
                self._world.debug.draw_string(v.get_location(), '¢', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                persistent_lines=True)
                print("Auto: ", v.type_id)
                print("Lane Id", self._map.get_waypoint(v.get_location()).lane_id)
                print("Road_id: ", self._map.get_waypoint(v.get_location()).road_id)
                print("Id Auto: ", v.id)
                print("Angle respect ego right vector: ", a)"""
            

            junction_waypoint, junction_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=5)

            """self._world.debug.draw_string(junction_waypoint.transform.location, '8', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                persistent_lines=True)

            junction_center_location = self._last_junction.bounding_box.location
            self._world.debug.draw_string(junction_center_location, 'Y', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=255), life_time=120.0,
                persistent_lines=True)"""
            
            front_ok = False
            right_ok = False
            left_ok = False    
            
            if junction_direction == RoadOption.LEFT or junction_direction == RoadOption.STRAIGHT :
                """print("devo girare a sinistra")

                print("CERCO DI PASSARE':")
                print("Len Front_Vehicles: ", len(front_vehicles))
                print("Len left_Vehicles: ", len(left_vehicles))
                print("Len right_Vehicles: ", len(right_vehicles))
                print("Len in_junction: ", len(junction_vehicles))
                if len(left_vehicles) > 0:
                    print(f"Il veicolo che viene da SINISTRA e' {left_vehicles[0][0]}, ha velocita' {get_speed(left_vehicles[0][0])} ed accellerazione {left_vehicles[0][0].get_acceleration().length()}")
                
                if len(right_vehicles) > 0:
                    print(f"Il veicolo che viene da DESTRA {right_vehicles[0][0]}, ha velocita' {get_speed(right_vehicles[0][0])} ed accellerazione {right_vehicles[0][0].get_acceleration().length()}")

                if len(front_vehicles) > 0:
                    print(f"Il veicolo che viene da DI FRONTE {front_vehicles[0][0]}, ha velocita' {get_speed(front_vehicles[0][0])} ed accellerazione {front_vehicles[0][0].get_acceleration().length()}")

                if len(junction_vehicles) == 1:
                    print("Condizioni su junction_vehicles", self.exiting_junction(junction_vehicles[0]))
                """

                front_ok = (len(front_vehicles) == 0)  or (len(front_vehicles) != 0  and get_speed(front_vehicles[0][0]) < 5 and front_vehicles[0][0].get_acceleration().length() < 2) or \
                    (len(front_vehicles) != 0 and get_speed(front_vehicles[0][0]) < 40 and front_vehicles[0][0].get_acceleration().length() < 10)

                left_ok = (len(left_vehicles) == 0 )or (len(left_vehicles) != 0 and get_speed(left_vehicles[0][0]) < 5 and left_vehicles[0][0].get_acceleration().length() < 2) or \
                    (len(left_vehicles) != 0 and get_speed(left_vehicles[0][0]) < 40 and left_vehicles[0][0].get_acceleration().length() < 10)

                right_ok = (len(right_vehicles) == 0) or (len(right_vehicles) != 0  and get_speed(right_vehicles[0][0]) < 5 and right_vehicles[0][0].get_acceleration().length() < 2) or \
                    (len(right_vehicles) != 0 and get_speed(right_vehicles[0][0]) < 40 and right_vehicles[0][0].get_acceleration().length() < 10) 
        
            elif junction_direction == RoadOption.RIGHT:
                # Computing the time required to the closest left vehicle to reach the ego-vehicle's last waypoint in the junction.
                # Then this time is compared to the time required to the ego vehicle to cross the junction, i.e. the time that it takes 
                # to reach its last waypoint in the junction.
                front_ok = True
                right_ok = True

                left_ok = (len(left_vehicles) == 0 )or (len(left_vehicles) != 0 and get_speed(left_vehicles[0][0]) < 5 and left_vehicles[0][0].get_acceleration().length() < 2) or \
                    (len(left_vehicles) != 0 and get_speed(left_vehicles[0][0]) < 40 and left_vehicles[0][0].get_acceleration().length() < 10)

            elif junction_direction == RoadOption.LANEFOLLOW:
                self._behavior.junction_crossing_speed = 20 
                print("Ho finito di attraversare")
                self._junction_state = JUNCTION_IN_PROGRESS
                return None

            others_vehicles = left_vehicles + right_vehicles + front_vehicles
            junction_cleaning = len(junction_vehicles) > 0 and any([self.exiting_junction(v, front_projection = 6,  rear_projection=1) for v in junction_vehicles])
            all_stopped = (len(junction_vehicles) >= 0 and all([get_speed(v) <= ZERO_SPEED_TH for v in junction_vehicles]))
            noone_entering = (len(junction_vehicles) == 0) and all( [self.exiting_junction(v, exiting_junction=False, front_projection = 4,  rear_projection=1.5) for v, _ in others_vehicles] )
            #empty_junction = len(junction_vehicles) == 0 

            #junction_ok = (junction_cleaning and noone_entering) or (all_stopped and noone_entering): #forse va controllato che il junction sia vuoto
            
            junction_ok = False
            """print("IN JUNCTION")
            for v in junction_vehicles: print(v)
            print("FINE LISTA")"""
            left = [v for v, _ in left_vehicles]
            right = [v for v, _ in right_vehicles]
            # Prendo i veicoli che si trovano a destra e a sinistra dell'ego veicolo durante l'incrocio
            right_junction, left_junction = self.get_lateral_targets(junction_vehicles+left+right)

            left_is_same = False
            if len(right_junction) and len(left_junction):
                right_junction = [(v, polygon_distance(v)) for v in right_junction]
                left_junction = [(v, polygon_distance(v)) for v in left_junction]

                right_junction.sort(key=lambda a: a[1])
                left_junction.sort(key=lambda a: a[1])

                """print("DESTRA E SINISTRA")
                for v in right_junction: print(v)
                for v in left_junction: print(v)"""

                current_right_junction = False
                current_left_junction = False
                if len(right_junction) > 0:
                    current_right_junction = right_junction[0][0]
                    """self._world.debug.draw_string(current_right_junction.get_location(), 'X', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=1.0,
                                persistent_lines=True)"""
                if len(left_junction) > 0:
                    current_left_junction = left_junction[0][0]
                    """self._world.debug.draw_string(current_left_junction.get_location(), 'X', draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255), life_time=1.0,
                    persistent_lines=True)"""

                #print("VALUTO JUNCTION_OK A SINISTRA")

                if junction_direction == RoadOption.LEFT:
                    if current_left_junction:
                        #print("VALUTO JUNCTION_OK")
                        right_overtook = (self._previous_right_junction in [v for v, _ in left_junction])
                        left_is_same = self._previous_left_junction_distance in [d for _, d in left_junction]

                        #print("Mi superano a destra", right_overtook)
                        #print("A sinistra ho sempre la stessa macchina", left_is_same)
                        #print("La vecchia distanza a sinistra è ", self._previous_left_junction_distance)
                        #print("Le distanze dei veicoli a sinistra sono", [d for _, d in left_junction])

                        junction_ok = (right_overtook and left_is_same) or (left_is_same and junction_cleaning)
                        #print(self._previous_right_junction, current_left_junction)
                        #print(self._previous_left_junction_distance == current_left_junction)

                if current_right_junction:
                    self._previous_right_junction, self._previous_right_junction_distance = right_junction[0]
                if current_left_junction:
                    self._previous_left_junction, self._previous_left_junction_distance = left_junction[0]

            if junction_direction == RoadOption.RIGHT or junction_direction == RoadOption.STRAIGHT:
                junction_ok = noone_entering or junction_cleaning or all_stopped

            if left_ok and right_ok and front_ok and junction_ok:
                if self._junction_speed_counter > 30:
                    mean_speed = self._junction_speed_sum / self._junction_speed_counter
                    self._behavior.junction_crossing_speed = mean_speed + 2 if mean_speed > 20 else 20
                else:
                    self._behavior.junction_crossing_speed = 20
                self._junction_state = JUNCTION_IN_PROGRESS
                return None

            """
                print("Sono passato perche':")
                print("Len Front_Vehicles: ", len(front_vehicles))
                print("Len left_Vehicles: ", len(left_vehicles))
                print("Len right_Vehicles: ", len(right_vehicles))
                if len(left_vehicles) > 0:
                    print(f"Il veicolo che viene da SINISTRA e' {left_vehicles[0][0]}, ha velocita' {get_speed(left_vehicles[0][0])} ed accellerazione {left_vehicles[0][0].get_acceleration().length()}")
                
                if len(right_vehicles) > 0:
                    print(f"Il veicolo che viene da DESTRA {right_vehicles[0][0]}, ha velocita' {get_speed(right_vehicles[0][0])} ed accellerazione {right_vehicles[0][0].get_acceleration().length()}")

                if len(front_vehicles) > 0:
                    print(f"Il veicolo che viene da DI FRONTE {front_vehicles[0][0]}, ha velocita' {get_speed(front_vehicles[0][0])} ed accellerazione {front_vehicles[0][0].get_acceleration().length()}")

                if len(junction_vehicles) == 1:
                    print("Condizioni su junction_vehicles", self.exiting_junction(junction_vehicles[0]))
            """
            """ print("Prima Condizione SINISTRA, Front Vehicles: ", len(front_vehicles) == 0 )
            print("Seconda condizione SINISTRA, Front Vehicles: ", (get_speed(front_vehicles[0][0]) < 5 and front_vehicles[0][0].get_acceleration().length() < 2))
            print("Terza Condizione SINISTRA, Front Vehicles: ", (get_speed(front_vehicles[0][0]) < 40 and front_vehicles[0][0].get_acceleration().length() < 10 and len(junction_vehicles) == 1 and self.exiting_junction(junction_vehicles[0])))
            
            print("Prima Condizione SINISTRA, Left Vehicles: ", len(left_vehicles) == 0 )
            print("Seconda condizione SINISTRA, Left Vehicles: ", (len(left_vehicles) == 0 and get_speed(left_vehicles[0][0]) < 5 and front_vehicles[0][0].get_acceleration().length() < 2))
            print("Terza Condizione SINISTRA, Left Vehicles: ", (get_speed(front_vehicles[0][0]) < 40 and front_vehicles[0][0].get_acceleration().length() < 10 and len(junction_vehicles) == 1 and self.exiting_junction(junction_vehicles[0])))

            print("Prima Condizione DESTRA, Left Vehicles: ", len(left_vehicles) == 0 )
            print("Prima Condizione FRONT: ", len(front_vehicles) == 0 )

            print("Seconda Condizione SINISTRA, Front Vehicles: ", len(front_vehicles) == 0 )
            """            
           
            """print("Location last transform", self._last_transform.location)
            print("Location Ego Vehicle: ", wp.transform.location)"""
            vehicle_list = self._world.get_actors().filter("*vehicle*")
            frontal_targets = self.get_frontal_targets(vehicle_list)
            vehicle_list = [(v, polygon_distance(v))for v in frontal_targets if not is_bike(v) and v.id != self._vehicle.id]
            vehicle_list.sort(key=lambda a: a[1])
            v, d = vehicle_list[0]
            #print(f"Distanza tra poligoni {v}, pari a {d} metri")

            ego_rear_location = self.get_rear_location(self._vehicle)
            distance_from_center = compute_distance(ego_rear_location, junction.bounding_box.location)

            if (d > 10 and distance_from_center > 2) or (left_is_same and right_junction[0][1] > 10):
                self._behavior.junction_crossing_speed = 20
                self._junction_state = JUNCTION_BIG_STEP
            #else:
                #print("Mi sto fermo")
            #self._dont_go = True
            
            return self.emergency_stop()

        elif self._junction_state == JUNCTION_BIG_STEP:
            if self.exiting_junction(self._vehicle):
                #print("ESCO DAL JUNCTION PERCHÈ E' VALIDA LA CONDIZIONE CHE MI FA USCIRE DE'")
                self._junction_state = JUNCTION_IN_PROGRESS
                return None
            if self._step_counter < 20:
                self._step_counter += 1
                self._behavior.junction_crossing_speed = 10
                #print("Faccio un passettino")
                return None
            #print("Ho fatto un BIG STEP")
            self._step_counter = 0
            self._junction_state = JUNCTION_WAITING

        elif self._junction_state == JUNCTION_IN_PROGRESS:
            #print("SONO IN JUNCTION_IN_PROGRESS") 
            if self.junction_crossing_manager(wp, distance = JUNCTION_DETECTION_DISTANCE):
                #print("STO ATTRAVERSANDO L'INCROCIO") 
                return None
            #print("HO FINITO DI ATTRAVERSARE L'INCROCIO")
            self._junction_state = JUNCTION_NOT_DETECTED
            self._last_transform = None
            self._first_junction_wp = None
            self._junction_speed_counter = 0
            self._junction_speed_sum = 0
            self._step_counter = 0
            self._previous_right_junction = None
            self._previous_left_junction = None
            self._previous_right_junction_distance = None
            self._previous_left_junction_distance = None
            return None
        
    def get_lateral_targets(self, targets):
        right_targets = []
        left_targets = []

        for target in targets:
            ego_location = self._vehicle.get_location()
            ego_forward_vector = self._vehicle.get_transform().rotation.get_forward_vector()
            ego_right_vector = self._vehicle.get_transform().rotation.get_right_vector()


            target_location = target.get_location()
            #target_forward_vector = target.get_transform().rotation.get_forward_vector()
            ego_target_vector = target_location - ego_location
            ego_target_vector = carla.Vector3D(x = ego_target_vector.x, y = ego_target_vector.y, z = ego_target_vector.z)


            # Check per non considerare i veicoli collocati DIETRO l'ego-veicolo
            if ego_forward_vector.get_vector_angle(ego_target_vector) *180/math.pi > 90:
                continue
            
            right_vector_angle = ego_right_vector.get_vector_angle(ego_target_vector) *180/math.pi
            #print("right_vector_angle appena calcolato", right_vector_angle)
            #print("Vettore che congiunge il nostro veicolo al target", ego_target_vector.make_unit_vector())

            if 0 < right_vector_angle < 90:
                right_targets.append(target)     # list of vehicles coming from right 
            else: 
                left_targets.append(target)

        return right_targets, left_targets

    
    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        start_time = time()

        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
                
        print("LOCATION DELL'EGO VEICOLO: ", ego_vehicle_loc)

        if not self.overtaking_bike:
            if self.crossing_object_detector(ego_vehicle_wp):
                return self.emergency_stop()

        lane_narrowing = self.lane_narrowing_manager(ego_vehicle_wp)
        #junction_crossing = self.junction_crossing_manager(ego_vehicle_wp, distance = JUNCTION_DETECTION_DISTANCE)

        lane_controller = self._local_planner._vehicle_controller._lat_controller

        # 1: Lane Narrowing and Junction Crossing Managment
        if lane_narrowing:
            self._behavior.max_speed = self._behavior.lane_narrowing_speed
            lane_controller.set_offset(+LANE_NARROWING_LATERAL_OFFSET)
        # elif junction_crossing:
        #     self._behavior.max_speed = self._behavior.junction_crossing_speed
        else: 
            lane_controller.set_offset(0)
            self._behavior.max_speed = 50

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        walker_list = [(dist(w), w) for w in walker_list if dist(w) < WALKER_DISTANCE_PROXIMITY]
        for distance, _ in walker_list:
            if distance  < MAX_WALKER_PROXIMITY:
                return self.emergency_stop()
            elif distance < WALKER_DISTANCE_PROXIMITY:
                print("Ho rilevato un pedone")
                self._behavior.max_speed = self._behavior.pedestrian_proximity_speed
        

        """# 2: Red lights and stops behavior
        tl_state = self.traffic_light_manager(ego_vehicle_wp)
        if tl_state == TRAFFIC_LIGHT_NOT_OK:
            print("Sono davanti ad un semaforo rosso")
            return self.emergency_stop()
        elif tl_state == TRAFFIC_LIGHT_OK:
            print("Posso andare")"""
        

        if self.stop_sign_manager(ego_vehicle_wp):
            return self.emergency_stop()

        
        control = self.junction_manager(ego_vehicle_wp)
        if control is not None:
            return control
        

        # 3: Obstacle and bike overtake behavior 
        control = self.overtake_manager(same_direction_road=False, debug=debug)
        if control is not None:
            return control
        
        if self.overtaking_bike:
           lane_controller.set_offset(-BIKE_OVERTAKE_LATERAL_OFFSET)
           self._local_planner.set_speed(self._behavior.bike_overtake_speed)
           return self.bike_following_and_overtake_manager()
        
        #    control = self.bike_following_and_overtake_manager()
        #    if control is not None:
        #       return control

        # 4: Pedestrian avoidance behaviors
        #walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        
        #if walker_state:
        #    print("***PEDESTRIAN AVOID MANAGER WALKER STATE: ", walker_state)
        #    print("***PEDESTRIAN AVOID MANAGER WALKER ID: ", walker.id)
        #    print("***PEDESTRIAN AVOID MANAGER WALKER TYPE ID: ", walker.type_id)
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
        #    distance = w_distance - max(
        #        walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
        #            self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
        #    if distance < self._behavior.braking_distance:
        #        return self.emergency_stop()

        # 5: Obstacle avoidance behaviors
        if not self._junction_state == JUNCTION_IN_PROGRESS or not ego_vehicle_wp.is_junction:  
            object_state, target_object_list, obj_distance_list = self.collision_and_obstacle_avoid_manager(ego_vehicle_wp)
            
            if object_state: 
                # Distance is computed from the center of the ego-veicle and of the obstacle,
                # we use bounding boxes to calculate the actual distance
                first_obstacle_distance = obj_distance_list[0] - max(
                    target_object_list[0].bounding_box.extent.y, target_object_list[0].bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x) 

                # Emergency brake if the object is very close.
                # DETERMINARE LA DISTANZA SULLA BASE DELLA NOSTRA DECELERAZIONE E VELOCITà: USARE FORMULA d = v^2/2a
                stop_distance = (get_speed(self._vehicle) / 3.6) ** 2 / (2 * MEAN_STOP_ACCELERATION ) #da km/h a m/s divido per 3.6
                if first_obstacle_distance < stop_distance + OBSTACLE_STOP_DISTANCE:
                    self._overtake_state = OVERTAKE_DETECTED
                    self._obstacle_detected = (object_state, target_object_list, obj_distance_list)
                    #print("HO INDIVIDUATO UN OSTACOLO")
                    return self._local_planner.run_step(debug=debug)

        # 6: Car following behaviors
        vehicle_state, vehicle_list, vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            vehicle = vehicle_list[0]
            distance = vehicle_distance[0]         
            
            if not is_bike(vehicle):#not vehicle.type_id.startswith('vehicle.bh') and not vehicle.type_id.startswith('vehicle.diamondback') and not vehicle.type_id.startswith('vehicle.gazelle'):
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                distance = distance - max(
                    vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

                # Emergency brake if the car is very close.
                #print("La distanza del car following e' ", distance)
                if distance < self._behavior.braking_distance and self._junction_state != JUNCTION_IN_PROGRESS:
                    #print("Mi fermo perchè sono troppo vicino!")
                    return self.emergency_stop()
                else:
                    #print("Ho chiamato car_following_manager")
                    control = self.car_following_manager(vehicle, distance)
            
            else:
                if not self.overtaking_bike and distance < START_BIKE_OVERTAKE_DISTANCE:
                    #print("**********HO DAVANTI ALMENO UNA BICI A MENO DI 20 m E STO PER INIZIARE UN SORPASSO**********")
                    #print("Ho almeno una bici davanti. Tutti i veicoli visti sono:")
                    #for v in vehicle_list:
                    #    print(v.type_id) 
                    bike_list = [vehicle] #
                    bike_distance = [distance]
                    for i in range(1, len(vehicle_list), 1):
                        if is_bike(vehicle_list[i]):
                            bike_list.append(vehicle_list[i])
                            bike_distance.append(vehicle_distance[i])
                        else:
                            break
                    #print("*****************HO IDENTIFICATO LE SEGUENTI BICI*******************")
                    #print("***************ABBIAMO INIZIATO IL SORPASSO, cambio velocita'************")
                    #print("FACCIO BIKE FOLLOWING")
                    return self.bike_following_and_overtake_manager(bike_list, bike_distance)
                control = self._local_planner.run_step(debug=debug)

        # 7: Intersection behavior
        elif ego_vehicle_wp.is_junction: #self._junction_state == JUNCTION_IN_PROGRESS:
            #print("Sono nell'elif") 
            target_speed = min([
                self._behavior.junction_crossing_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 8: Normal behavior
        else:
            #print("Sono nell'else")
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            #print("Limite di velocità attuale: ", self._speed_limit)
            
            #target_speed = self._behavior.max_speed - self._behavior.speed_lim_dist
            #print("Setto speed a ", target_speed)
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        elapsed_time = time() - start_time
        print("Tempo impiegato per fare il run step:", elapsed_time)
        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False

        current_steering = self._local_planner._vehicle_controller._lat_controller.run_step()

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self._local_planner._vehicle_controller.past_steering + 0.1:
            current_steering = self._local_planner._vehicle_controller.past_steering + 0.1
        elif current_steering < self._local_planner._vehicle_controller.past_steering - 0.1:
            current_steering = self._local_planner._vehicle_controller.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self._local_planner._vehicle_controller.max_steer, current_steering)
        else:
            steering = max(-self._local_planner._vehicle_controller.max_steer, current_steering)

        control.steer = steering
        #print("Steering control:", steering)
        return control


    def get_polygon(self, target):
        target_bb = target.bounding_box
        target_vertices = target_bb.get_world_vertices(target.get_transform())
                #print("Ostacolo: ", target_object)
                #print("ID Ostacolo: ", target_object.id)
                #print(f"*************************L'ostacolo ha {len(target_vertices)} vertici")
                #print(f"*************************Location del waypoint: {target_object.get_location()}")
        """for item in target_vertices:
            #print("Stampo i vertici della bounding box dell'ostacolo")
            w = self._map.get_waypoint(item, lane_type=carla.LaneType.Any)
            self._world.debug.draw_string(w.transform.location, '[]', draw_shadow=False,
                                color=carla.Color(r=255, g=255, b=0), life_time=1.0,
                                persistent_lines=True)"""
        target_list = [[v.x, v.y, v.z] for v in target_vertices]
        #if len(target_list) < 3: return None
        target_polygon = Polygon(target_list)
        return target_polygon

    def get_front_location(self, target, projection = 1):
        target_extent = target.bounding_box.extent.x
        target_front_vector = target.get_transform().get_forward_vector()
        target_location = target.get_transform().location

        target_front_location = target_location + carla.Location(
            x= projection * target_extent * target_front_vector.x,
            y= projection * target_extent * target_front_vector.y,
        )
        return target_front_location

    def get_rear_location(self, target, projection = 1):
        target_extent = target.bounding_box.extent.x
        target_front_vector = target.get_transform().get_forward_vector()
        target_location = target.get_transform().location

        target_rear_location = target_location - carla.Location(
            x= projection * target_extent * target_front_vector.x,
            y= projection * target_extent * target_front_vector.y,
        )
        return target_rear_location

    def exiting_junction(self, vehicle, exiting_junction = True, front_projection = BOUNDING_BOX_PROJECTION, rear_projection = 1):

        front_location = self.get_front_location(vehicle, projection = front_projection)
        rear_location = self.get_rear_location(vehicle, projection = rear_projection) 

        """if not exiting_junction:
            self._world.debug.draw_string(front_location, 'FRONT' , draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=1.0,
            persistent_lines=True)

            self._world.debug.draw_string(rear_location, 'REAR' , draw_shadow=False,
            color=carla.Color(r=0, g=0, b=255), life_time=1.0,
            persistent_lines=True)"""

        # print("Posterior: ", vehicle_rear_location)
        # print("Front: ", vehicle_front_location)

        front_in_junction = self._map.get_waypoint(front_location).is_junction
        rear_in_junction = self._map.get_waypoint(rear_location).is_junction

        #print("Front of target in junction:", front_in_junction)
        #print("Rear of target in junction:", rear_in_junction)

        if exiting_junction:
            return not front_in_junction and rear_in_junction
        else: return front_in_junction and not rear_in_junction

    def get_frontal_targets(self, targets):
        frontal_targets = []

        for target in targets:
            ego_location = self._vehicle.get_location()
            ego_forward_vector = self._vehicle.get_transform().rotation.get_forward_vector()
            ego_right_vector = self._vehicle.get_transform().rotation.get_right_vector()


            target_location = target.get_location()
            #target_forward_vector = target.get_transform().rotation.get_forward_vector()
            ego_target_vector = target_location - ego_location
            ego_target_vector = carla.Vector3D(x = ego_target_vector.x, y = ego_target_vector.y, z = ego_target_vector.z)


            # Check per non considerare i veicoli collocati DIETRO l'ego-veicolo
            if ego_forward_vector.get_vector_angle(ego_target_vector) *180/math.pi > 100:
                continue
            
            frontal_targets.append(target)

        return frontal_targets
