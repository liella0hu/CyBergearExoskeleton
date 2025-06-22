import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath as sm



def text():

    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    print("Tep", Tep)
    robot = rtb.models.Panda()
    sol = robot.ik_LM(Tep)         # solve IK
    print("robot", robot)
    print("sol", sol)
    q_pickup = sol[0]
    print("robot.fkine(q_pickup)", robot.fkine(q_pickup))
    qt = rtb.jtraj(robot.qr, q_pickup, 50)
    robot.plot(qt.q, backend='pyplot', movie='panda1.gif')

def y_arm():
    rtb.backends.set_backend(rtb.backends.pythreejs.PyThreeJSBackend())

    # 创建一个简单的两连杆机械臂
    L1 = 1.0  # 第一连杆的长度
    L2 = 1.0  # 第二连杆的长度

    # 定义关节和链接
    joint1 = rtb.Revolute('j1', 'z')  # 绕z轴的旋转关节
    link1 = rtb.links.RigidBody('link1', joint1, sm.SE3(0, 0, 0), L1 * sm.Vector(0, 0, 1))  # 第一连杆

    joint2 = rtb.Revolute('j2', 'z')  # 绕z轴的旋转关节（连接到第一连杆的末端）
    link2 = rtb.links.RigidBody('link2', joint2, sm.SE3(0, 0, L1), L2 * sm.Vector(0, 0, 1))  # 第二连杆

    # 将链接组合成机械臂
    robot = rtb.SerialLink([link1, link2], name='simple_arm')

    # 设置关节角度
    q = [0.5, 1.0]  # 关节1和关节2的角度（以弧度为单位）

    # 计算机械臂的正向运动学
    T = robot.fkine(q)

    # 绘制机械臂
    robot.plot(q)

def draw_CyberArm():
    import numpy as np
    from spatialmath import SE3
    puma = rtb.models.Puma560()
    puma.plot([0, 0, 0, 0, 0, 0])
    # # puma.teaching()
    # puma.plot([np.pi / 4, 0, 0, 0, 0, 0], block=True, ax=None, fig=None, backend='matplotlib', viewer=None,
    #           robotcolor='b', basecolor='g', linkcolor=None, jointcolor='k', jointwidth=3, linkwidth=2, drawframes=True,
    #           drawcom=True, drawbase=True, drawlinks=True, elevation=30, azimuth=30, zoom=1.0, pause=0.1)

def learn_spatialmath():
    import matplotlib.pyplot as plt
    from spatialmath.base import trplot

    pose = SE3.Rx(90, unit='deg') * SE3(1, 2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trplot(pose.A, frame='A', color='r', ax=ax)  # 绘制红色坐标系
    plt.show()

def rtpy_tongyi():
    import roboticstoolbox as rtb
    from spatialmath import SE3
    import matplotlib.pyplot as plt
    import numpy

    # 创建 Puma560 机器人模型
    puma = rtb.models.DH.Puma560()

    # 打印机器人信息
    print(puma)

    # 设置关节角度为 [pi/4, pi/4, pi/4, pi/4, pi/4, pi/4]
    q = [numpy.pi / 4] * puma.n

    # 计算末端执行器的位置
    T = puma.fkine(q)
    print("End-effector pose:\n", T)

    # 创建一个新的图形窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制机器人
    puma.plot(q, backend='pyplot')
    plt.title("Puma560 Robot Model with Joint Angles [pi/4, pi/4, pi/4, pi/4, pi/4, pi/4]")
    plt.show()

    # 目标位置
    Tt = SE3.Trans(0.8, -0.2, 0.1) * SE3.RPY([0, 0, -numpy.pi / 4])

    # 初始关节配置
    q0 = [0, 0, 0, 0, 0, 0]

    # 使用 IKCCS 求解逆运动学
    sol = puma.ikine_LM(Tt, q0=q0)

    if sol.success:
        print("Solution found:", sol.q)
    else:
        print("No solution found")

    # 计算机器人在给定关节角度下的雅可比矩阵
    J = puma.jacob0(sol.q)
    print("Jacobian matrix at the solution:\n", J)

    # 使用多项式轨迹生成方法
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = rtb.ctraj(SE3(), SE3.Tz(1), 50)
    # 绘制路径中的每个姿态
    for t in path:
        # 使用逆运动学求解关节角度
        q_path = puma.ikine_LM(t).q
        if q_path is not None:
            puma.plot(q_path, backend='pyplot')

    plt.title("Trajectory of Puma560 Robot")
    plt.show()

    # 创建一个简单的仿真实验
    env = rtb.backends.Swift()
    env.launch()
    env.add(puma)
    puma.q = sol.q
    env.step()


def rtpy_tongyi_move():
    import roboticstoolbox as rtb
    from spatialmath import SE3
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置 Matplotlib 后端为 TkAgg
    plt.switch_backend('TkAgg')
    # 创建 Puma560 机器人模型
    puma = rtb.models.DH.Puma560()

    # 打印机器人信息
    print(puma)

    # 设置初始关节角度为 [pi/4, pi/4, pi/4, pi/4, pi/4, pi/4]
    q_initial = [np.pi / 4] * puma.n

    # 计算初始末端执行器的位置
    T_initial = puma.fkine(q_initial)
    print("Initial end-effector pose:\n", T_initial)

    # 目标位置

    T_target = SE3.Trans(0.5, -0.2, 0.3) * SE3.RPY([0, 0, -np.pi / 4])
    # 使用 IKCCS 求解逆运动学得到目标关节角度
    sol = puma.ikine_LM(T_target, q0=q_initial)

    if sol.success:
        print("Solution found for target position:", sol.q)
    else:
        raise ValueError("No solution found for target position")

    q_target = sol.q

    # 使用多项式轨迹生成方法
    path = rtb.ctraj(SE3(), T_target, 50)

    # 创建一个新的图形窗口来绘制轨迹
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置绘图范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Puma560 Robot Trajectory")

    # 绘制路径中的每个姿态
    for i, t in enumerate(path):
        # 使用逆运动学求解关节角度
        sol_path = puma.ikine_LM(t, q0=q_initial)
        if sol_path.success:
            q_path = sol_path.q
            # 清除之前的绘图
            ax.cla()
            # 设置绘图范围
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Puma560 Robot Trajectory (Step {i + 1}/50)")

            # 绘制机器人当前姿态
            puma.plot(q_path, backend='pyplot', block=False)

            # 绘制起始点和终点
            ax.scatter(T_initial.t[0], T_initial.t[1], T_initial.t[2], color='green', label='Start')
            ax.scatter(T_target.t[0], T_target.t[1], T_target.t[2], color='red', label='Target')

            # 添加图例
            ax.legend()

            # 更新图形
            # plt.pause(0.01)

    plt.show()

def rtpy_wenxin_move():
    import roboticstoolbox as rtb
    from spatialmath import SE3
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建 Puma560 机器人模型
    puma = rtb.models.DH.Puma560()

    # 设置初始关节角度
    q_initial = [np.pi / 4] * puma.n

    # 计算初始末端执行器的位置
    T_initial = puma.fkine(q_initial)
    print("Initial end-effector pose:\n", T_initial)

    # 目标位置
    T_target = SE3.Trans(0.5, -0.2, 0.3) * SE3.RPY([0, 0, -np.pi / 4])

    # 使用 IKCCS 求解逆运动学得到目标关节角度
    sol = puma.ikine_LM(T_target, q0=q_initial)

    if sol.success:
        print("Solution found for target position:", sol.q)
        q_target = sol.q

        # 使用多项式轨迹生成方法（这里仅用于说明，实际并不用于绘图）
        # path = rtb.ctraj(SE3(), T_target, 50)

        # 创建一个新的图形窗口来绘制轨迹
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 设置绘图范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Puma560 Robot Trajectory")

        # 绘制起始点和终点
        ax.scatter(T_initial.t[0], T_initial.t[1], T_initial.t[2], color='green', label='Start')
        ax.scatter(T_target.t[0], T_target.t[1], T_target.t[2], color='red', label='Target')

        # 绘制目标关节角度对应的机器人姿态
        puma.plot(q_target, backend='pyplot', block=False, ax=ax)

        # 添加图例
        ax.legend()

        plt.show()
    else:
        raise ValueError("No solution found for target position")


def rtpy_csdn_move():
    import roboticstoolbox as rtb
    robot = rtb.models.DH.Panda
    qt = rtb.tools.trajectory.jtraj(robot.qz, robot.qr, 50)
    robot.plot(qt.q)

def load_urdf():
    import matplotlib
    matplotlib.use('Qt5Agg')  # 在 import roboticstoolbox 前调用
    import roboticstoolbox as rtb
    from spatialmath import SE3
    from spatialmath.base import trplot
    import swift
    urdf_path = r"E:\fight_for_py\CyBergearExoskeleton\robotics-toolbox-python-master\learning_robot_arm\cyberarm_urdf.SLDASM.urdf"
    robot = rtb.ERobot.URDF(
        urdf_path,
        # gripper="link3"  # 可选：指定末端执行器连杆
    )
    robot.teach(q=robot.q, backend='pyplot')
    robot.plot(q=robot.q, backend='pyplot')
    print(robot)
    # print(robot.joints)  # 输出关节信息
    print(robot.dh)  # 输出关节信息


if __name__ == '__main__':
    load_urdf()

