
#########################################################
Evaluating Implications of Problematic Routing Assumptions in Spatially Explicit Agent-Based Models of Wildfire Evacuation

By Rebecca Vandewalle, Jeon-Young Kang, and Shaowen Wang 
Fall 2020

Code Base Documentation

#########################################################
### Purpose 

This code directory contains code needed to replicate the experiments using a spatially-explicit agent-based model of wildfire evacuation in a forthcoming manuscript by Vandewalle, Kang, and Wang, as well as companion documentation and a code demonstration Jupyter notebook.

The main code base is flexible and can serve a variety of purposes; not all available parameters are used in the manuscript. Broadly, the code models the process of evacuation on a road network in which roads are progressively closed by wildfire spread. Individual households, represented by a vehicle, must navigate out of the danger zone and re-route if the road they are currently on becomes blocked by the wildfire. The forthcoming manuscript specifically looks at patterns in evacuation clearance and congestion that change based on how vehicle routing decisions are modeled. Specifically three driving strategies are compared, 2 based off of common modeling assumptions (quickest path and shortest path), and one that attempts to more closely model evacuee behavior (preference for major roads). These strategies are described in more detail in the manuscript and the demonstration notebook.

#########################################################
### Code and documentation contents

This package contains all files needed to run experiments and generate output used in the forthcoming manuscript. Additionally, a Jupyter Notebook, intended for use in CyberGISX, is provided to interactively demonstrate sections of the source code with more detailed commentary.

Targeted information to assist with running the source code can be found in the DOCUMENTATION folder and provided in individual help files for easy access. This includes required Python packages, a detailed description of provided files, the expected directory set-up required to run the code out of the box (files are already provided in this structure), and other similar added information.

