import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
from niapy.task import Task
import numpy as np
from niapy.algorithms.basic import ParticleSwarmAlgorithm, GeneticAlgorithm, BatAlgorithm
from niapy.problems import Problem
from niapy.task import OptimizationType
import matplotlib.pyplot as plt

# Define the robot UR5 with Robotiq 85 gripper
class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

    # Load the robot URDF into simulation
    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    # Parse joint information
    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    # Setup mimic joints for the gripper
    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    # Move arm to a target pose using inverse kinematics
    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    # Control the gripper to open or close
    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    # Get current end-effector position
    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state

# Step the simulation forward
def update_simulation(steps, sleep_time=0.01):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(sleep_time)

# Setup PyBullet simulation environment
def setup_simulation():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
    cube_id2 = p.loadURDF("cube.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.6, useFixedBase=True)

    tray_pos = [0.5, 0.9, 0.6]
    tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    tray_id = p.loadURDF("tray/tray.urdf", tray_pos, tray_orn)
    return tray_pos, tray_orn

# Randomly color the cube
def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)

# Define the grasp optimization problem
class GraspOptimizationProblem(Problem):
    def __init__(self, robot, cube_pos, base_z):
        self.robot = robot
        self.cube_pos = cube_pos
        self.base_z = 0.78

        # Set dimension=2, optimize x and y only
        super().__init__(dimension=2,  
                         lower=[cube_pos[0] - 0.3, cube_pos[1] - 0.3],
                         upper=[cube_pos[0] + 0.3, cube_pos[1] + 0.3],
                         )

        # Initialize evaluation tracking and real-time plotting
        self.evaluations = []
        self.distances = []
        self.xy_history = [] 
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'blue', label='PSO')
        self.ax.set_xlabel('Evaluation')
        self.ax.set_ylabel('Fitness Value')
        self.ax.set_title('Learning Curve')
        self.ax.grid(True)
        self.data_file = "optimization_data_pso.npy"
        self.ax.legend(loc='upper right')

    # Evaluate a candidate solution
    def _evaluate(self, x):
        p.resetBasePositionAndOrientation(self.robot.cube_id, self.cube_pos, p.getQuaternionFromEuler([0, 0, 0]))
        tx, ty = x
        tz = self.base_z

        eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        eef_position = eef_state[0]
        eef_orientation = eef_state[1]

        distance = math.sqrt((tx - self.cube_pos[0])**2 + (ty - self.cube_pos[1])**2)

        self.evaluations.append(self.evaluations[-1] + 1 if self.evaluations else 1)

        if self.distances:
            last_distance = self.distances[-1]
            distance = min(distance, last_distance)
            if distance == last_distance:
                tx, ty = self.xy_history[-1]

            self.robot.move_arm_ik([tx, ty, tz+0.05], eef_orientation)
            update_simulation(40)
            self.robot.move_arm_ik([tx, ty, tz], eef_orientation)
            update_simulation(40)
            self.robot.move_gripper(0.01)
            update_simulation(20)
            self.robot.move_arm_ik([tx, ty, tz + 0.3], eef_orientation)
            update_simulation(40)
            self.robot.move_gripper(0.085)
            update_simulation(20)
        else:
            self.robot.move_arm_ik([tx, ty, tz+0.05], eef_orientation)
            update_simulation(40)
            self.robot.move_arm_ik([tx, ty, tz], eef_orientation)
            update_simulation(40)
            self.robot.move_gripper(0.01)
            update_simulation(20)
            self.robot.move_arm_ik([tx, ty, tz + 0.3], eef_orientation)
            update_simulation(40)
            self.robot.move_gripper(0.085)
            update_simulation(20)

        self.distances.append(distance)
        self.xy_history.append((tx, ty))

        self.line.set_data(self.evaluations, self.distances)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Reset arm to initial position after each evaluation
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        update_simulation(100)

        np.save(self.data_file, [self.evaluations, self.distances])

        return distance

    # Plot final progress with best solution
    def plot_progress(self, best_solution, best_fitness):
        plt.ioff()
        plt.figure(self.fig.number)

        text = f"Best Solution: x={best_solution[0]:.3f}, y={best_solution[1]:.3f} | Minimum Distance: {best_fitness:.4f}"

        self.fig.text(0.5, -0.02, text, ha='center', va='bottom', color='blue', fontsize=12)
        self.fig.savefig("learning_curve_pso.png", bbox_inches='tight')
        plt.close()

# Main function
def main():
    tray_pos, tray_orn = setup_simulation()

    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()

    cube_pos = [0.5, 0.0, 0.65]
    cube_id = p.loadURDF("cube_small.urdf", cube_pos, p.getQuaternionFromEuler([0, 0, 0]))
    random_color_cube(cube_id)
    robot.cube_id = cube_id 

    target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
    update_simulation(200)

    algo = "PSO"

    problem = GraspOptimizationProblem(robot, cube_pos, 0.78)
    task = Task(problem=problem, max_evals=100)

    if algo == 'GA':
        algo = GeneticAlgorithm(population_size=10)
    elif algo == 'PSO':
        algo = ParticleSwarmAlgorithm(population_size=10)
    elif algo == 'BAT':
        algo = BatAlgorithm(population_size=10)

    best_solution, best_fitness = algo.run(task)
    problem.plot_progress(best_solution, best_fitness)

    eef_state = p.getLinkState(robot.id, robot.eef_id)
    eef_position = eef_state[0]
    eef_orientation = eef_state[1]
    
    x, y = best_solution
    print(f"Best solution: x={x:.3f}, y={y:.3f}")
    print(f"Minimum Distance: {best_fitness:.4f}")

    for i in range(5):
        p.addUserDebugText(f"Test Phase", textColorRGB=[255, 0, 0], textPosition=[0.5, 0.5, tray_pos[2] + 0.3],
                            textSize=2, lifeTime=10)
        p.resetBasePositionAndOrientation(robot.cube_id, cube_pos, p.getQuaternionFromEuler([0, 0, 0]))
        robot.move_arm_ik([x, y, 0.78], eef_orientation)
        update_simulation(100)
        robot.move_gripper(0.01)
        update_simulation(30)
        robot.move_arm_ik([x, y, 0.98], eef_orientation)
        update_simulation(100)
        robot.move_gripper(0.085)
        update_simulation(30)
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        update_simulation(100)

# Entry point
if __name__ == "__main__":
    main()
