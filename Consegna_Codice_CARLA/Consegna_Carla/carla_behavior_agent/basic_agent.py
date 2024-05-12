#PROVA COMMENTO COMMIT

# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
import heapq as hq
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)
# from perception.perfectTracker.gt_tracker import PerfectTracker

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False #mettere a True
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._ignore_obstacles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        self._base_object_threshold = 5.0  # meters

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'ignore_obstacles' in opt_dict:
            self._ignore_obstacles = opt_dict['ignore_obstacles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def ignore_obstacles(self, active=True):
        """(De)activates the checks for obstacles"""
        self._ignore_obstacles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light_1(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

            """        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)"""

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        
        """
        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
        """
        for traffic_light in lights_list:

            trigger_location = get_trafficlight_trigger_location(traffic_light)
            trigger_wp = self._map.get_waypoint(trigger_location)
            #self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ego_front_vector = ego_vehicle_waypoint.transform.get_forward_vector()
            trigger_front_vector = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ego_front_vector.x * trigger_front_vector.x + ego_front_vector.y * trigger_front_vector.y + ego_front_vector.z * trigger_front_vector.z

            if dot_ve_wp < 0:
                continue

            #if traffic_light.state != carla.TrafficLightState.Red:
            #    return (True, traffic_light)

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                print("C'è un semaforo nei paraggi: ", traffic_light)
                #self._last_traffic_light = traffic_light
            return (True, traffic_light)
                
        #print("Non c'è un semaforo nei paraggi")
        return (False, None)
    

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state == carla.TrafficLightState.Green:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ego_front_vector = ego_vehicle_waypoint.transform.get_forward_vector()
            trigger_front_vector = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ego_front_vector.x * trigger_front_vector.x + ego_front_vector.y * trigger_front_vector.y + ego_front_vector.z * trigger_front_vector.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected_old(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

        
    '''
    def _object_obstacle_detected(self, object_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is an object in front of the agent blocking its path.

            :param object_list: list contatining objects.
                If None, all objects in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        
        if self._ignore_obstacles:
            return (False, None, -1)
        
        if not object_list:
            object_list = self._world.get_actors().filter("*static*")

        if not max_distance:
            max_distance = self._base_obstacle_threshold
        
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1
        
        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        dangerous_object_list = []
        distance_object_list = []

        for i in range(len(object_list)):
            #target_object = hq.heappop(object_list)[1]
            target_object = object_list[i][1]
            target_transform = target_object.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            
            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:

                if target_object.type_id.startswith("vehicle"):
                    if  get_speed(target_object) > 0.1 or target_wpt.lane_id != ego_wpt.lane_id+1:
                        continue
                else:
                    if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                        if not next_wpt:
                            continue
                        if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                            continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_object.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    dangerous_object_list.append(target_object)
                    distance_object_list.append(compute_distance(target_transform.location, ego_transform.location))
        
        return dangerous_object_list, distance_object_list'''



    def _object_obstacle_detected(self, object_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is an object in front of the agent blocking its path.

            :param object_list: list contatining objects.
                If None, all objects in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        
        if self._ignore_obstacles:
            return (False, None, -1)
        
        if not object_list:
            object_list = self._world.get_actors().filter("*static*")

        if not max_distance:
            max_distance = self._base_obstacle_threshold
        
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1
        
        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        dangerous_object_list = []
        distance_object_list = []

        for i in range(len(object_list)):
            #target_object = hq.heappop(object_list)[1]
            target_object = object_list[i][1]
            target_transform = target_object.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            
            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:

                if target_object.type_id.startswith("vehicle"):
                    if  get_speed(target_object) > 0.1 or target_wpt.lane_id != ego_wpt.lane_id+1:
                        continue
                else:
                    if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                        if not next_wpt:
                            continue
                        if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                            continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_object.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                '''if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    dangerous_object_list.append(target_object)
                    distance_object_list.append(compute_distance(target_transform.location, ego_transform.location))'''
                
                route_bb = []
                app_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])
                app_bb.append(p1)
                app_bb.append(p2)

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])
                    app_bb.append(p1)
                    app_bb.append(p2)

                '''for item in app_bb:
                    w = self._map.get_waypoint(item, lane_type=carla.LaneType.Any)
                    self._world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                       persistent_lines=True)'''

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return [], []
                ego_polygon = Polygon(route_bb)

                
                target_extent = target_object.bounding_box.extent.x
                if target_object.id == self._vehicle.id:
                    continue
                if ego_location.distance(target_object.get_location()) > max_distance:
                    continue
                
                #print("L'ID dell'ostacolo è", target_object.id)
                #print("Il tipo del veicolo è ", target_object.type_id)
                target_bb = target_object.bounding_box
                '''self._world.debug.draw_string(target_object.get_location(), 'X', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)'''
                target_vertices = target_bb.get_world_vertices(target_object.get_transform())
                #print("Ostacolo: ", target_object)
                #print("ID Ostacolo: ", target_object.id)
                #print(f"*************************L'ostacolo ha {len(target_vertices)} vertici")
                #print(f"*************************Location del waypoint: {target_object.get_location()}")
                """for item in target_vertices:
                    #print("Location del bounding box: ", item)
                    w = self._map.get_waypoint(item, lane_type=carla.LaneType.Any)
                    self._world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=255, b=0), life_time=120.0,
                                       persistent_lines=True)"""
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    #return (True, target_object, compute_distance(target_object.get_location(), ego_location))
                    dangerous_object_list.append(target_object)
                    distance_object_list.append(compute_distance(target_transform.location, ego_transform.location))

        
        return dangerous_object_list, distance_object_list

    def _vehicle_obstacle_detected_2(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
    
        if self._ignore_vehicles:
            return None, -1
        
        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold
        
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        dangerous_vehicle_list = []
        distance_vehicle_list = []

        for i in range(len(vehicle_list)):
            #target_object = hq.heappop(object_list)[1]
            target_vehicle = vehicle_list[i]
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            
            #if (ego_wpt.is_junction and target_wpt.is_junction) or not ego_wpt.is_junction:

            # Simplified version for outside junctions
            #if target_vehicle.id == 3786:
            #    print("is junction dell'ego_vehicle: ", str(ego_wpt.is_junction), ", is junction del target: ", str(target_wpt.is_junction)) 


            # IF is 
            if (ego_wpt.is_junction and target_wpt.is_junction) or not ego_wpt.is_junction:
                #if target_vehicle.id == 3786:
                #    print("CONSIDERO IF PER L'IMPALA")
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:#se l'ostacolo non è sulla mia stessa strada o se rispetto alla corsia dove sono non mi dà fastidio
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0] #prendo come prossimo waypoint il terzo
                    if not next_wpt: #se sono arrivato a destinazione
                        #if target_vehicle.id == 3786:
                        #    print("USCITA 1")

                        continue #esce e passa alla prox iterazione
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:#non sono ancora a destinazione ma non mi dà fastidio l'ostacolo
                        #if target_vehicle.id == 3786:
                        #    print("USCITA 2")

                        #print(f"il veicolo è {target_vehicle.id}, la road id del veicolo è  {target_wpt.road_id}, e la road id del waypoint è {next_wpt.road_id}, lane id del veicolo è {target_wpt.lane_id}, lane id del waypoint {next_wpt.lane_id}")
                        continue

                target_forward_vector = target_transform.get_forward_vector() 
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                #print("Distanza tra ego e target vehicle: ", compute_distance(target_rear_transform.location, ego_front_transform.location))
                #print("MAX allowed distance :", max_distance)
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    dangerous_vehicle_list.append(target_vehicle)
                    distance_vehicle_list.append(compute_distance(target_transform.location, ego_transform.location))
                    #if target_vehicle.id == 3786:
                    #    print("AGGIUNGO L'IMPALA")

                    continue
                #if target_vehicle.id == 3786:

                #print(" USCITA 3")

            else:
                #if target_vehicle.id == 3786:
                #    print("CONSIDERO L'ELSE PER L'IMPALA")
                '''if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    dangerous_object_list.append(target_object)
                    distance_object_list.append(compute_distance(target_transform.location, ego_transform.location))'''
                
                route_bb = []
                app_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])
                app_bb.append(p1)
                app_bb.append(p2)

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])
                    app_bb.append(p1)
                    app_bb.append(p2)

                """for item in app_bb:
                    w = self._map.get_waypoint(item, lane_type=carla.LaneType.Any)
                    self._world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=1.0,
                                        persistent_lines=True)"""

                if len(route_bb) < 3:
                    print("NO POLIGONO, ESCO")
                    # 2 points don't create a polygon, nothing to check
                    return [], []
                ego_polygon = Polygon(route_bb)

                
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == self._vehicle.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    #if target_vehicle.id == 3786:
                    #    print("NON INSERISCO L'IMPALA PERCHE' TROPPO DISTANTE")
                    continue
                
                #print("L'ID dell'ostacolo è", target_vehicle.id)
                #print("Il tipo del veicolo è ", target_vehicle.type_id)
                target_bb = target_vehicle.bounding_box
                """self._world.debug.draw_string(target_vehicle.get_location(), 'X', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1.0,
                                        persistent_lines=True)"""
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                #print("Ostacolo: ", target_object)
                #print("ID Ostacolo: ", target_object.id)
                #print(f"*************************L'ostacolo ha {len(target_vertices)} vertici")
                #print(f"*************************Location del waypoint: {target_object.get_location()}")
                """for item in target_vertices:
                    #print("Stampo i vertici della bounding box dell'ostacolo")
                    w = self._map.get_waypoint(item, lane_type=carla.LaneType.Any)
                    self._world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=255, b=0), life_time=1.0,
                                        persistent_lines=True)"""
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    #if target_vehicle.id == 3786:
                    #    print("INTERSECO IL BOUNDING BOX DELL'IMPALA")
                    #return (True, target_object, compute_distance(target_object.get_location(), ego_location))
                    dangerous_vehicle_list.append(target_vehicle)
                    distance_vehicle_list.append(compute_distance(target_transform.location, ego_transform.location))
                    continue
                #if target_vehicle.id == 3786:
                #    print("NON INTERSECO IL BOUNDING BOX DELL'IMPALA ")

        
        return dangerous_vehicle_list, distance_vehicle_list

    
    
    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            #print("Il veicolo è ", target_vehicle, "ha road id")
            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                #print("ENTRO NELL'IF, USO I WAYPOINTS")

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:#se l'ostacolo non è sulla mia stessa strada o se rispetto alla corsia dove sono non mi dà fastidio
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0] #prendo come prossimo waypoint il terzo
                    if not next_wpt: #se sono arrivato a destinazione
                        continue #esce e passa alla prox iterazione
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:#non sono ancora a destinazione ma non mi dà fastidio l'ostacolo
                        #print(f"il veicolo è {target_vehicle.id}, la road id del veicolo è  {target_wpt.road_id}, e la road id del waypoint è {next_wpt.road_id}, lane id del veicolo è {target_wpt.lane_id}, lane id del waypoint {next_wpt.lane_id}")
                        continue

                target_forward_vector = target_transform.get_forward_vector() 
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )
                """
                w = self._map.get_waypoint(target_rear_transform.location)
                self._world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=255, b=255), life_time=3.0,
                                       persistent_lines=True)
                """


                #print("Location point of target vehicle: ", target_rear_transform.location)
                #print("distance between ego and target vehicle: ", compute_distance(target_transform.location, ego_transform.location))
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                #print("ENTRO NELL'ELSE")
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

                return (False, None, -1)

        return (False, None, -1)
        


    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=True, lane_changes=1, step_distance=2):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        # It generates a list of waypoints in the same lane in order to traverse a given distance (distance_same_lane)
        # before changing lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)  # It takes the last waypoint from the position of the vehicle
                                                        # and generate waypoints up to step_distance distance
            if not next_wps:    # It ends when arrived at destination
                return []
            next_wp = next_wps[0]   # Otherwise it returns the first of the waypoints generated before
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location) # It updates the distance covered by the generated waypoints along the same lane
            plan.append((next_wp, RoadOption.LANEFOLLOW))
        
        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0                                       # Counter of lane changes done
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes: # Loop that terminates after reaching the number of lane changes indicated by the parameter 'lane_changes' 

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan

            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].previous(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan

    def _generate_overtake_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=True, lane_changes=1, step_distance=2, first_half = True):
        """
        This methods generates a path that results in a lane change for a overtake manouver.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        param: first_half True when the first half of the overtake manouver should be generated, 
                            False otherwise. 
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        # It generates a list of waypoints in the same lane in order to traverse a given distance (distance_same_lane)
        # before changing lane
        distance = 0
        while distance < distance_same_lane:
            
            # This and the subsequent checks are necessary to ensure that the waypoints are generated in the right direction with respect to the vehicle direction, and not in the direction of the lane
            if first_half:
                next_wps = plan[-1][0].next(step_distance)  # It takes the last waypoint from the position of the vehicle # and generate waypoints up to step_distance
            else:
                next_wps = plan[-1][0].previous(step_distance) 
                                                        
            if not next_wps:    # It ends when arrived at destination
                return []
            next_wp = next_wps[0]    # Otherwise it returns the first of the waypoints generated before
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location) # It updates the distance covered by the generated waypoints along the same lane
            plan.append((next_wp, RoadOption.LANEFOLLOW))
        
        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0                                       # Counter of lane changes done
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes: # Loop that terminates after reaching the number of lane changes indicated by the parameter 'lane_changes' 

            if first_half:
                next_wps = plan[-1][0].next(lane_change_distance)
            else:
                next_wps = plan[-1][0].previous(lane_change_distance)

            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            if first_half:
                next_wps = plan[-1][0].previous(step_distance)
            else:
                next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan