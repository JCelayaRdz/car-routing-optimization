# car-routing-optimization
This repository contains the final project for the course **Bioinspired Algorithms for Optimization** at **ETSISI - Universidad Politécnica de Madrid (UPM)**.

## Problem definition
Given a weighted directed graph representing the road network of Madrid, extracted from OpenStreetMap data and processed to include information on geometry, speed, estimated fuel consumption and low emission zones (LEZ), the problem of finding optimal routes between origin-destination node pairs under multiple objectives and constraints is posed.

The objective is to simultaneously minimize:

* Total distance traveled (based on edge length).

* Total travel time (calculated from estimated speed).

* Estimated fuel consumption (in liters), based on track type and length.

Subject to:

Hard constraints:
* Prohibited to cross edges intersecting the ZBE zone if the vehicle does not meet the permitted criteria.

* Oneway: No oneway travel in the opposite direction on one-way roads.

Soft constraints:
Penalization of streets with:

* Few lanes, which may mean less capacity or more congestion.

* Presence of traffic lights (highway = traffic_signals).

* Low quality or accessibility sections (tertiary, unclassified, living_street...).