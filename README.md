# Q-Learning Project

Team members: Kir Nagaitsev and Sydney Jenkins

gif, at 4x speed for fast demo:

![Robot](https://github.com/Loonride/q_learning_project/blob/master/gifs/q_learning.gif)

# Writeup

## Objectives Description

The goal of this project can be divided into two components: having the robot learn where each colored dumbbell should be placed and having the robot use this information to succesfully move the dumbbells. To accomplish the former goal, the robot will use a Q-learning algorithm to learn which numbered block should be associated with each colored dumbbell. To accomplish the latter goal, the robot will use the learned Q-matrix to correctly position the dumbbells. 

## High-level description

We used reinforcement learning to determine how to position dumbbells in the training phase. To accomplish this, we first initialize a Q-matrix, which has rows corresponding to possible world states and columns representing possible actions to be taken. Each action corresponds to moving a dumbbell to a specific block. Our robot then iterates over different possible actions using an action matrix, which stores possible actions for each  state. For each action, the robot receives a reward (from /q_learning/reward) depending on whether it was correct or not, and the Q-matrix is updated accordingly. Once the Q-matrix converges, we determine that our robot's training is complete. The Q matrix is then saved for use in the action phase. 

## Q-learning algorithm description

For Q-Learning we used a discount factor of 0.8, and we found a learning rate of 1 to be sufficient in order to accomplish the task. The training algorithm is implemented in the q_learning.py and actions.py files, and uses the virtual reset world to handle rewards.

Actions are selected and published in the `do_next_action` function of the actions.py file. This function calls `choose_action` which will randomly select a valid action from the action matrix. When the reward is received from the subscription, we update the current state, update and publish the Q-Matrix using the values described above, then call `do_next_action` again.

We check the current value of the Q-Matrix each time we are going to update it, and if the Q-Matrix remains unchanged for 500 iterations we say that it has converged, saving it to a CSV file.

We choose actions to take based on the valid action of the highest weight in our Q-Matrix at the current state. We continue this until each dumbbell is at a block which should give a reward. This also happens in actions.py, when it loads the CSV we saved back in and selects the action in `choose_action` based on the loaded matrix, rather than choosing randomly. This difference in behavior is due to an important property of the class `use_saved_matrix`, which determines whether we are doing training or doing actions based on the saved matrix.

## Robot perception description

We identify the locations and identities of each of the dumbbells in `compelete_action` (in robot_movement.py) using data from the robot's RGB camera. We use OpenCV to apply a mask to the image, as in the in-class line-follower program. This allows us to identify the center of the colored dumbbell and to calculate the corresponding error in the robot's position. The robot turns until this error falls below an experimentally-determined threshold, at which point it moves forward until it is .21 away from the dumbbell. This distance is determined in the function `get_distance`, which sets the variable front_distance according to the data in the Laser Scanner. 

We identify the locations and identities of the blocks using a pre-trained keras_ocr model. In the function `find_number` (in robot_movement.py), we input the data from the RGB camera into the model and, if the relevant number is identified in the image, we return the center of the number's location in the image. In `complete_action`, we use this output to either decide to look in another direction (if the number was not found) or to turn the robot towards the center outputted by `find_number`. The robot turns until it is facing the center (within some error). 

## Robot manipulation and movement

We use the method described in previous paragraphs to identify the location of the desired dumbbell. The robot then moves toward the dumbbell, adjusting its angular velocity according to the calculated error (as determined in `complete_action`) in its position. It then stops once it is .21 away from the dumbbell. This action is performed in `complete_action`. The distance to the dumbbell is determined in `get_distance` by taking the first element of the Laser Scanner data array. 

We first intialize the robot arm's position in `initialize_arm`, which opens the arm's grippers and positions it at the correct height to grasp the dumbbells. As the robot approaches the dumbbell (according to the code in `complete_action`), the grippers go around the dumbbell. The dumbbell is then picked up in `pickup_db` by raising the robot's arm. 

We determine the location of the desired block using the methods described in previous paragraphs. The robot spins until the pre-trained keras model identifies the relevant number in the inputted RGB image in the function `find_number` (in robot_movement.py). Once the block has been identified, we calculate the error in the robot's position in `complete_action` and change the robot's angular velocity accordingly. Once the error falls below a threshold, we have the robot move in a straight line until it is .75 away from the block. The distance is calculated in `get_distance`, which uses the data from the Laser Scanner to get the distances to the nearest objects. 

Once the robot is in front of the block, as determined by `get_distance`, we lower the arm in `set_db` (in robot_movement.py). This sets the robot back down. We then move the robot away from the dumbbell by setting the linear velocity to -.5. This ensures that the grippers are no longer aroung the dumbbell. 

## Challenges

We faced several challenges in this assignment. For instance, once our robot picked up the dumbbell, we initially struggled with identifying the corresponding numbered block. We found that this was because the RGB camera was supplying images facing the wrong direction. We had to correct for this by iterating through the image sensor readings until we got to the most recent one. We also struggled with getting the robot to face the correct position once it had the dumbbell. We ended up calculating the center of the number identified by the keras model and calculating an error that was then used to adjust the robot's angular velocity. This approach largely worked, as the robot then approached the correct block. However, it doesn't always stop at exactly the correct position to place the dumbbell directly in front of the block. Additionally, it was difficult to coordinate work remotely, though class times dedicated to working on the project were helpful for touching base. 

## Future work

If we had more time, we would improve how our robot identifies the locations of the numbered blocks. This is because, even though our robot is able to successfully identify the locations of the blocks, it doesn't always approach the blocks at the correct angle to position the dumbbell directly in front of the blocks. This could be improved by re-calculating (using the keras model) the location of the numbers as the robot approaches the block. This would, however, intoduce an additional lag to the robot's movements, as the keras model takes a while to identify objects in the input image. To avoid this, we could also find some way to improve how we initially position the robot before it begins to move toward the block. Additionally, once we identify the numbered block, we might try to position the robot without any subsequent calls to the keras model, as this would make the robot quicker. 

## Takeaways

- One take-away is that remote work on GitHub requires a lot of coordination. We tried to avoid working on the project at the same time because we didn't want to have to merge our different copies of the code. Communication was therefore critical for determining who would work on what when, and this had the benefit that we were always up-to-date on what the other person was working on. 

- An additional take-away is that you can use tools such as keras to add functionality to your rospy code. Using the pre-trained keras model was relatively simple, and added more complex behavior to our robot. Though we didn't use such a tool for color identification, it would have been interesting to explore some of the available tools for color recognition as well. 

# Implementation Plan

## Q-learning algorithm

We will execute the Q-learning algorithm by saving our initial state 0, randomly choosing a valid action based on the current state and submitting this action, then computing the new value of the state/action location in our Q-matrix when the reward is sent back. We will then call our action function to continue the process, resetting back to state 0 whenever there are no valid actions. We can test this using the provided phantom robot code, then placing a large delay between each action to see that actions are being done correctly and weights are being assigned correctly through printing of the Q-matrix.

We will determine when the Q-matrix has converged by comparing the Q-matrix on each iteration, and if it does not change for some large set number of iterations like 100, we will say it has converged. This constant may need to be larger, but 100 is likely enough to confirm convergence. We will test this by running the algorithm twice and checking that the 2 resulting Q-matrices are identical, indicating that they both converged.

Once the Q-matrix has converged, given the current state of the robot, we will select the action at that state with the highest value in the Q-matrix. We will repeat this until no more actions can be made, and this should lead to the robot likely receiving a reward. Testing this requires first confirming that the robot actually gets a good reward, then confirming that the action sequence matches up with the ideal action choices in the Q-matrix.

## Robot perception

Determining the identities and locations of the three colored dumbbells: We will use a color recognition algorithm (e.g. from https://data-flair.training/blogs/project-in-python-colour-detection/) to identify the color of dumbells in the robot's RGB camera feed. This algorithm will calculate the "distance" between the given rgb values and the rgb values corresponding to each color, and will return the color with the minimum distance. We will use the rgb values along the robot's line of sight, so that if it identifies green, blue, or red, the robot will know both the identity and location of the block. We will test this component by writing a function that returns the colors and locations that the robot identifies as containing a dumbbell and comparing them to the true values. 

Determining the identities and locations of the three numbered blocks: We will use the /scan ROS topic to identify the directions of nearby objects. We will then use the pre-trained keras_ocr model to identify the numbers and positions of the blocks observed by the robot's RGB camera (accessed using the /camera/rgb/image_raw ROS topic) when facing these directions. We will test this component using a similar approach to the one described above: we will write a function that returns the number identifiers and locations of the blocks identified by the robot, and compare them to the true values. 

## Robot manipulation & movement
Picking up and putting down the dumbbells with the OpenMANIPULATOR arm: We will use move_group_gripper.go() to ensure that the distance between the grippers is less than the maximum dumbbel width and larger than the minimum dumbbell width.  To pick up the dumbbell, we will then calculate what height the gripper should be at, use move_grou_arm.go() to move the arm to the goal position, have the robot approach the dumbbell until it is close enough to grasp the dumbbell, and lift the arm in order to lift the dumbbell. To put down the dumbbell, we will then lower the arm to the original height and have the arm pull away. We will test this component by placing a dumbbell in front of the robot and having it attempt to pick it up and put it down. 

Navigating to the appropriate locations to pick up and put down the dumbbells: Once the location of the dumbbell/block relative to the robot has been determined using the methods described in Robot Perception, we will have the robot turn in the given direction and move forward until it reaches some distance x from the object. We can test this component by placing the robot in a world with dumbbells and blocks and having it navigate toward one of them. 

## Timeline

- May 2 - Have a single or a few Q-learning iterations set up.
- May 3 - Complete Q-learning and test it. 
- May 5 - Have robot perception set up.
- May 7 - Set up robot manipulation for basic actions.
- May 10 - Put everything together to run actions with robot perception and the Q-matrix.
- May 12 - Finish writeup
