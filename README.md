# Python Animation Simulator for Manhattan Grid Routing

This Python animation simulator demonstrates the routing process in a Manhattan grid system using various pathfinding algorithms like A*, BFS (Breadth-First Search), and DFS (Depth-First Search). The simulator visually represents the exploration of paths from a single source to multiple destinations, helping to understand and compare the efficiency and path selection of each algorithm.

## Project Structure

The project is divided into two main files:

- `algorithms.py`: Contains the implementation of the pathfinding algorithms A*, BFS, and DFS. This file provides the core functionality for determining the routes in the Manhattan grid system.
  
- `maze_solver.py`: Manages the animation and user interaction. This file creates the graphical representation of the Manhattan grid and controls the simulation based on user input and algorithmic outputs from `algorithms.py`.

## Features

- **Multiple Algorithms**: Compare the performance and pathfinding capabilities of A*, BFS, and DFS.
- **Animation Control**: Control the simulation to start, pause, or reset, allowing a step-by-step observation of the routing process.
- **Visual Representation**: Offers a clear visual representation of how each algorithm explores the grid and determines the path.

## Prerequisites

Before you run the simulator, ensure that Python 3.x is installed on your system. You will also need several Python libraries that handle animations, graph structures, and general computation. Here's a list of required libraries:

- `matplotlib`: For creating visualizations and animations.
- `networkx`: To handle graph-based data structures used in pathfinding algorithms.
- `IPython`: For displaying images and animations in IPython notebooks if you want to run the simulator in a notebook environment.

You can install all required libraries using pip. Run the following command:

```bash
pip install matplotlib networkx IPython 
