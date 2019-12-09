Robot Parallel Motion Planning
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Peyman Norouzi: [LinkedIn](https://www.linkedin.com/in/peymannorouzi)

* Klayton Wittler: [LinkedIn](https://www.linkedin.com/in/klayton-wittler)

Industrial road | Running from the cops 
:-------------------------:|:-------------------------:
![](images/swerve.gif)| ![](images/block.gif)

## Table of Content:

- [Overview](#overview)
  * [Parrallel Motion Planning on the GPU](#parrallel-motion-planning-on-the-gpu)
  * [CARLA](#carla)
- [GMT* Algorithm](#gmt-algorithm)
  * [Sampling Based Motion Planning](#sampling-based-motion-planning)
- [Implementation](#implementation)
  * [Enviroments](#enviroments)
  * [Pipeline](#pipeline)
  * [State Space Samples](#state-space-samples)
  * [Dubin's Model](#dubins-model)
  * [Collison Check](#collison-check)
  * [Debugging/Unit Testing](#debuggingunit-testing)
- [Performance Analysis](#performance-analysis)
  * [Bloopers](#bloopers)
- [Milestones](#milestones)
- [Credits/Sources](#creditssources)


## Overview:

Autonomous vehicles are a new sensation of the 21st-century economy. Many big/small companies are fiercely competing in their path of full driving autonomy. The goal of this project is to make use of the parallelization power of the GPU to create an autonomous robot (vehicle) capable of reaching a defined goal while navigating in an unknown enviroment.

An autonomous vehicle needs to be able to do the following 4: 

**1. Identify:**

An autonomous vehicle needs to be able to comprehend its surroundings and identify objects that may affect its decision making. Identification is usually done using the car's perception/vision system. The perception system usually consists of a combination of RGB cameras, Lidar and Radar.

**2. Predict:**

Now that the vehicle knows about objects surrounding it, it needs to predict surroundings behavior over a short and long time so the vehicle can appropriately behave in various scenarios. For example, If a car is moving on the lane next to it is moving at 30 MPH, the vehicle should be able to predict what the car is going to do in the next 5 seconds. 

**3. Decide:**

With the current knowledge of the surroundings and the desired destination, the car then needs to decide on how to move (motion planning) to reach the desired goal safely and efficiently.


**4. Execute:**

When a decision is made by the car's computer system, the car needs to be able to appropriately execute the plan in the environment that it is in using its engine wheel and transmission system


This project is mainly focused on number **3. Decide**. We implemented GMT* motion planning algorithm on the GPU and tested its effectiveness in various unknown environments in CARLA car simulator.

### Parrallel Motion Planning on the GPU:

Motion planning algorithms are meant to help an autonomous robot to find a sequence of valid configurations that moves it from the source to the desired destination. Due to the sequential nature of the task, many different sequential algorithms can successfully achieve this in various scenarios. One example of such an algorithm is [FMT*(Fast Marching Tree)](https://arxiv.org/pdf/1306.3532.pdf). Although FMT* works great for high dimensional state space, due to its sequential nature it is not able to be run on the GPU. For the algorithm to be able to be run on the GPU, we need to alter the algorithm to be able to run in parallel at some abstract level. This where [GMT* (Group Marching Tree)](https://arxiv.org/abs/1705.02403) paper comes into play. Brian Itcher et al proposed a variation to the FMT* that would allow it to run on the GPU. We implemented the algorithm on the GPU and tested our result on CARLA driving simulator. We will discuss the algorithm in more detail later in the read me!

### CARLA:

Our testing environment for this project is [CARLA driving simulator](http://carla.org/), which is an open-source development, training, and validation environment for autonomous driving systems. CARLA is built on top of UE4 and has a python client API that allows us to engage with a car and driving environment. We decided to go with CARLA due to time and financial resources (check our bloopers :D) that an actual robot would have taken from us to complete this project.

### Build:

To be able to run our code, ...

## GMT* Algorithm:

### Sampling Based Motion Planning

![](images/gmtExample1.gif)


## Implementation:

In this section we go through our implementation that allows our virtual [Tesla Model 3](https://twitter.com/elonmusk?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) drive around the city so smoothly!

### Enviroments:

There are 3 different subsystems/environments that work with each other enabling our autonomous car to perform its tasks. CARLA, as the simulator, gives us the actual environment, obstacles, physics rules, etc. [PyCuda](https://documen.tician.de/pycuda/) (that is a Cuda wrapper for Python), on the other hand, deals with all the operations that need to be done in the GPU. Python is the binding environment between CARLA and PyCuda. Python gets environment information from CARLA and transfers it appropriately to PyCuda for GPU operations. After Path calculation on the GPU, python transfers the path to CARLA to be executed in the environment. You can see the three systems' interactions in the figure below:

<p align="center">
  <img src="images/enviroment.png">
</p>

### Pipeline:

<p align="center">
  <img src="images/pipeline.png">
</p>

You can see the overall pipeline of our code implementation. This pipeline is a generic pipeline for many self-driving robotic applications. We implemented the whole pipeline using the three environments explained above. The red rectangle indicates the main part of our implementation in which we created from the ground up using the three environments (This is where we implement GMT* on the GPU). The rest of the pipeline was implemented so that we have the proper follow of information but we used CARLA for gathering the required information. The information was gathered in a way though to simulate an actual flow of information in a real self-driving environment. For example, in the sensors section, we did not create our sensor configuration but rather used the simulator (CARLA) for the gathering of important obstacle information (such as their location in the map). The information was quired in a way to simulate the location of the obstacles only when we get to 30 meters approximately of them (just like how real sensor configuration would behave). 


### State Space Samples

Discretizing space is required when you are trying to plan your motion in a continuous space. We defined our state space to be in the following format:

<p align="center">
  <img src="http://bit.ly/36h0qbI">
</p>

Where X and Y are the locations with respect to CARLA's world coordinate system and θ is the yaw angle of the car (Car's orientation with respect to Z (height)). We queried, all of the possible location states from CARLA that lie in the streets where the car can move. The left image in the figure below shows the states that CARLA gives us. We designed state space points(waypoints) to be at least 4 meters apart since 4 gave us a good balance between performance and precision. Since CARLA only offers orientations that correspond with the flow of traffic on the corresponding side of the road, we had to add more 6 more directions for each given X and Y combination so that we would be able to deal with obstacles we face during our motion. We made sure not to have orientation opposite to the flow of traffic on each respective side of the road, making sure that after each maneuver around obstacles, our car doesn't continue driving on the wrong side of the road. You can see our final set up our state-space on the right animation in the figure below. 

CARALA state samples | Up sample orientation 
:-------------------------:|:-------------------------:
![](images/carlaStateSample.png)| ![](images/upSampleStates.gif)

### Dubin's Cost Model:

Now that we have our state space deiscrtezed, we need to figure out a way to evaluate how costly it is to move between each  of the states. This way when we are 

A 4-wheel vehcile is a non-holomonic 

Dubin's Model | Example 
:-------------------------:|:-------------------------:
![](images/dubinsModel.png)| ![](images/dubinExample.gif)

### Collison Check:




### Debugging/Unit Testing:

Since we are dealing with three environments for the completion of this project, there can be many opportunities in which our simulation might fail! But we knew simulation failure didn't necessarily mean that our Implementation of the GMT* algorithm is wrong! We also knew that it would be really hard to test for every test/edge cases using the full CARLA simulation, thus we had to come up with some clever Unit test so that we can check our GPU implementation frequently and efficiently! These tests saved us many many hours of unnecessary grief while debugging our code! You can see the three Unit tests all below (red box represent an obstacle): (top image is the state space and the bottom picture is the correct output path that our algorithm found)


Unit Test 1 | Unit Test 2 | Unit Test 3 
:-------------------------:|:-------------------------:|:-------------------------:
![](images/unitTest1.png)| ![](images/unitTest2.png)| ![](images/unitTest3.png) 

## Performance Analysis:

### Results:

### Optimization and Future work:

### Bloopers:


## Milestones:

* [Pitch](files/CIS565_finalProject-pitch.pdf)
* [Milestone 1](files/CIS565_RPMP-milestone1.pdf)
* [Milestone 2](files/CIS565_RPMP-milestone2.pdf)
* [Milestone 3](files/CIS565_RPMP-milestone3.pdf)

## Credits/Sources:

FMT: https://arxiv.org/pdf/1306.3532.pdf
GMT: https://arxiv.org/abs/1705.02403
CARLA: http://carla.org/
PyCuda: https://documen.tician.de/pycuda/


