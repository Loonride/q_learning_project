# q_learning_project
## Q-learning algorithm
Executing the Q-learning algorithm 
Determining when the Q-matrix has converged: 
Once the Q-matrix has converged, how to determine which actions the robot should take to maximize expected reward: 

## Robot perception

Determining the identities and locations of the three colored dumbbells: We will use a color recognition algorithm (e.g. from https://data-flair.training/blogs/project-in-python-colour-detection/) to identify the color of dumbells in the robot's RGB camera feed. This algorithm will calculate the "distance" between the given rgb values and the rgb values corresponding to each color, and will return the color with the minimum distance. We will use the rgb values along the robot's line of sight, so that if it identifies green, blue, or red, the robot will know both the identity and location of the block. We will test this component by writing a function that returns the colors and locations that the robot identifies as containing a dumbbell and comparing them to the true values. 

Determining the identities and locations of the three numbered blocks: We will use the /scan ROS topic to identify the directions of nearby objects. We will then use the pre-trained keras_ocr model to identify the numbers and positions of the blocks observed by the robot's RGB camera (accessed using the /camera/rgb/image_raw ROS topic) when facing these directions. We will test this component using a similar approach to the one described above: we will write a function that returns the number identifiers and locations of the blocks identified by the robot, and compare them to the true values. 

## Robot manipulation & movement
Picking up and putting down the dumbbells with the OpenMANIPULATOR arm: 
Navigating to the appropriate locations to pick up and put down the dumbbells: 

## Timeline