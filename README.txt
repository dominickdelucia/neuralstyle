############## README FOR STYLE TRANSFER PROJECT BY TEAM SDDD ############################


*** Instruction on how to run the code: 

The main functions are in helpers/image_objects.py and helpers/arch_objects.py
These are the Objects that do the heavy lifting in this system

First, have to initalize the Style Image and Content Image as image objects by giving a file path for where the source image is

Next, have to initialize an architecture with learning rate and epochs thru which those images will be run

Finally, just have to kickoff the process by running "arch_object.run_network(content_obj, style_obj)" 



*** Main Jupyter Notebook: 
For producing results, the main notebook is kickoff_grid.ipynb
However, I would suggest looking into the Reasearch.ipynb file because it shows an example iteration of our research process



*** Other Main Functions and where to find them: 

Loss function implementations can be found in helpers/loss_functions.py
Architecture settings relevant for the research process can be found in helpers/architecture.py
Some important functions for loading and visualizing the images can be found in helpers/viz_funcs.py


*** Location of Image Content
All images can be found in the imgs/ folder 

Base content images we utilized are in the imgs/content folder
Base style images we utilized are in the imgs/style folder 

All other folders in imgs/ are outputs of our research loops *some are nice, some are terrible*
