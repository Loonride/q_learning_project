# Q-Learning Project

Team members: Kir Nagaitsev and Sydney Jenkins

gif, at 4x speed for fast demo:

![Robot](https://github.com/Loonride/q_learning_project/blob/master/gifs/q_learning.gif)

# Writeup

## Objectives Description

## High-level description

## Q-learning algorithm description

For Q-Learning we used a discount factor of 0.8, and we found a learning rate of 1 to be sufficient in order to accomplish the task.

## Robot perception description

## Robot manipulation and movement

## Challenges

## Future work

## Takeaways



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
