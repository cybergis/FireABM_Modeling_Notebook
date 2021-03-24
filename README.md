
## Evaluating Implications of Problematic Routing Assumptions in Spatially Explicit Agent-Based Models of Wildfire Evacuation
### Code Base Documentation

#### By Rebecca Vandewalle, Jeon-Young Kang, and Shaowen Wang 
This repository contains code needed to run an agent-based emergency evacuation simulation in Python. Agents can use one of 3 routing strategies to determine a path from their initial location to out of the evacuation zone. Simultaneously, a wildfire object can spread throughout the simulation and close roads.

![An example output simulation](img/example_run.gif)

### Start here
Run [FireABM\_Demo_Notebook.ipynb](FireABM_Demo_Notebook.ipynb) for an overview of how the simulation works.

### Purpose 
This code directory contains code needed to replicate the experiments using a spatially-explicit agent-based model of wildfire evacuation in a forthcoming manuscript by Vandewalle, Kang, and Wang, as well as companion documentation and a code demonstration Jupyter notebook. 

The main code base is flexible and can serve a variety of purposes; not all available parameters are used in the manuscript. 

Due to the time taken, everything needed to run the same experiments is included in this repo, but the actual simulations runs for the experiment are intended to be run on HPC such as Keeling/Virtual Roger.

### What does the code do
The code models the process of evacuation on a road network in which roads are progressively closed by wildfire spread. Individual households, represented by a vehicle, must navigate out of the danger zone and re-route if the road they are currently on becomes blocked by the wildfire. The forthcoming manuscript specifically looks at patterns in evacuation clearance and congestion that change based on how vehicle routing decisions are modeled. Specifically three driving strategies are compared, 2 based off of common modeling assumptions (quickest path and shortest path), and one that attempts to more closely model evacuee behavior (preference for major roads). These strategies are described in more detail in the manuscript and the demonstration notebook.

### Code and documentation contents
This package contains all files needed to run experiments and generate output used in the forthcoming manuscript.

### How to run the Jupyter Notebook
No setup is needed, just copy this repository to your Jupyter notebook file system and run the notebook.

### How to run a simulation on Keeling/Virtual Roger (HPC)
See [0. Replicating the manuscript experimental procedure.txt](DOCUMENTATION/0. Replicating the manuscript experimental procedure.txt) for steps taken to run simulations used for the manuscript

1. Make sure your directory structure is set up as expected (see [3. Expected base directory structure.txt](DOCUMENTATION/3. Expected base directory structure.txt))
1. Make sure you have the required python libraries installed (see [4. Required Python packages.txt](DOCUMENTATION/4. Required Python packages.txt))
1. If on Keeling or another HPC environment, set up a virtual environment (see [5. Creating a virtual environment for Osmnx.txt](DOCUMENTATION/5. Creating a virtual environment for Osmnx.txt))
1. If you want to set initial agent positions by population, get households data from the US Census  (see [6. Gathering household data.txt](DOCUMENTATION/6. Gathering household data.txt))
1. If you want to make your own wildfire simulation to use in the model, use FlamMap to generate a fire (see [7. Creating a simulated wildfire with FlamMap.txt](DOCUMENTATION/7. Creating a simulated wildfire with FlamMap.txt))
1. Determine which core script to use (see [8. Which script version to use - opt or opt_Keel.txt](DOCUMENTATION/8. Which script version to use - opt or opt_Keel.txt))
1. To run batch jobs determine which scripts you want to run (see [9. Running batch jobs.txt](DOCUMENTATION/9. Running batch jobs.txt))
1. Determine which simulation parameters to use (see [10. Simulation run parameters.txt](DOCUMENTATION/10. Simulation run parameters.txt))

### Simulation Output
Running the simulation will generate videos, such as the one at the beginning of this readme, as well as text results (see [11. Simulation output structure and explanation.txt](DOCUMENTATION/11. Simulation output structure and explanation.txt)).  [12. Example simulation run times.txt](DOCUMENTATION/12. Example simulation run times.txt) contains run times to help estimate how long a simulation might run. Finally, [13. Graph Files with names ending in orig.txt](DOCUMENTATION/13. Graph Files with names ending in orig.txt) discusses which results files are provided.

#### Important files
See [1. Index of provided files.txt](DOCUMENTATION/1. Index of provided files.txt) for short description the provided files in the repo

- [FireABM\_Demo_Notebook.ipynb](FireABM_Demo_Notebook.ipynb): the core notebook demonstrating how the simulation works
- [combined\_rslts\_8_13.txt](combined_rslts_8_13.txt): Combined results file containing results from all the sumulation runs that were run on HPC
- [Graph\_Results_Final.ipynb](Graph_Results_Final.ipynb): Use this notebook to create all of the figures used in the manuscript taking result data stored from [combined\_rslts\_8_13.txt](combined_rslts_8_13.txt)
- [Rebuild\_Road_Graphs.ipynb](Rebuild_Road_Graphs.ipynb): This notebook demonstrates all preprocessing steps taken to prepare the road network, households data, and wildfire perimeters for use in the simulation

