# Maze Pathfinding Visualization

## Overview

This project implements and visualizes multiple pathfinding algorithms on a maze. The goal is to analyze and compare their efficiency in terms of nodes expanded and path length. The visualization provides an animated representation of how each algorithm explores the maze.

## Features

Supports five different pathfinding algorithms:

- Depth-First Search (DFS)

- Breadth-First Search (BFS)

- Greedy Best-First Search

- A* (A-Star) Search

- Random Search

Reads maze configurations from text files.

Visualizes search progress and final path using Matplotlib animations.

Tracks and reports the number of nodes expanded and the length of the found path for each algorithm.

### Maze Format

The maze is loaded from a .txt file where:

- X represents walls (obstacles).

-   (space) represents open paths.

- start x y defines the starting position.

- end x y defines the goal position.
