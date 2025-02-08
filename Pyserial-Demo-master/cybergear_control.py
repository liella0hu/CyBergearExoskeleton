import sys
import time
import numpy as np
import serial
import serial.tools.list_ports
import threading
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from cybergear_control_ui import Ui_Form
from serial_control_interface import SerialControllerInterface, CmdModes, RunModes


class CybergearControl(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(CybergearControl, self).__init__()
        self.setupUi(self)
        self.init()
        self.init_timer()
        self.init_QdoubleBox()
        self.init_matplotlib_canvas()
        self.init_threading()
        self.setWindowTitle("Cybergear")

        # self.ser = serial.Serial()


    def init(self):
        self.checkbox.clicked.connect(self.port_check)
        self.open_button.clicked.connect(self.cybergear_open_1)
        self.close_button.clicked.connect(self.cybergear_stop_1)

        self.open_button_2.clicked.connect(self.cybergear_open_2)
        self.close_button_2.clicked.connect(self.cybergear_stop_2)

        self.s3__send_button.clicked.connect(self.cyberger_execute_1)
        self.s3__send_button_2.clicked.connect(self.cyberger_execute_2)

        # self.timer_send_cb.stateChanged.connect(self.data_send_timer_1)
        # self.timer_send_cb_2.stateChanged.connect(self.data_send_timer_1)
        # 定时器接收数据

        self.close_button_EMG.clicked.connect(self.EMG_close)
    def init_timer(self):
        self.timer_draw = QTimer(self)  # 绘制图像
        self.timer_draw.timeout.connect(self.data_receive_draw)
        self.timer_draw.start(100)

        self.timer_send = QTimer(self)  # 连续动态参数运动
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer_1)
        self.timer_send_cb_2.stateChanged.connect(self.data_send_timer_1)

        self.timer_send_2 = QTimer(self)  # 连续动态参数运动
        self.timer_send_2.timeout.connect(self.data_send)


        self.timer_EMG = QTimer(self)
        self.timer_EMG.timeout.connect(self.EMG_read)
        self.open_button_EMG.stateChanged.connect(self.open_EMG)


    def init_QdoubleBox(self):
        self.speed_doubleSpinBox.setMinimum(0.0)
        self.speed_doubleSpinBox.setMaximum(10.0)
        self.speed_doubleSpinBox.setSingleStep(0.1)

        self.posision_doubleSpinBox_1.setMinimum(-10.0)
        self.posision_doubleSpinBox_1.setMaximum(10.0)
        self.posision_doubleSpinBox_1.setSingleStep(0.1)

        self.torque_doubleSpinBox_3.setMinimum(-10.0)
        self.torque_doubleSpinBox_3.setMaximum(10.0)
        self.torque_doubleSpinBox_3.setSingleStep(0.1)

        self.speed_doubleSpinBox_2.setMinimum(0.0)
        self.speed_doubleSpinBox_2.setMaximum(10.0)
        self.speed_doubleSpinBox_2.setSingleStep(0.1)

        self.position_doubleSpinBox_2.setMinimum(-10.0)
        self.position_doubleSpinBox_2.setMaximum(10.0)
        self.position_doubleSpinBox_2.setSingleStep(0.1)

        self.torque_doubleSpinBox_2.setMinimum(-10.0)
        self.torque_doubleSpinBox_2.setMaximum(10.0)
        self.torque_doubleSpinBox_2.setSingleStep(0.1)

    def init_matplotlib_canvas(self):
        self.figure_1, self.ax_1 = plt.subplots()
        self.ax_1.set_xlim(0, 60)
        self.motor1_speed_list = []
        self.motor1_position_list = []
        self.motor1_torque_list = []
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.display_verticalLayout_1.addWidget(self.canvas_1)

        self.figure_2, self.ax_2 = plt.subplots()
        self.ax_2.set_xlim(0, 60)
        self.motor2_speed_list = []
        self.motor2_position_list = []
        self.motor2_torque_list = []
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.display_verticalLayout_2.addWidget(self.canvas_2)

        self.figure_EMG_1, self.ax_EMG_1 = plt.subplots()
        self.ax_EMG_1.set_xlim(0, 240)
        self.ax_EMG_1.set_ylim(-10, 700)
        self.EMG_1_list = []
        self.EMG_canvas_1 = FigureCanvas(self.figure_EMG_1)
        self.display_verticalLayout_3.addWidget(self.EMG_canvas_1)

        self.figure_EMG_2, self.ax_EMG_2 = plt.subplots()
        self.ax_EMG_2.set_xlim(0, 240)
        self.ax_EMG_2.set_ylim(-10, 700)
        self.EMG_2_list = []
        self.EMG_canvas_2 = FigureCanvas(self.figure_EMG_2)
        self.display_verticalLayout_4.addWidget(self.EMG_canvas_2)
    def init_threading(self):
        self.data_send_flag_1 = False
        self.continuous_motion_1 = QThread()
        self.continuous_motion_1.start()
        self.data_send_flag_2 = False
        self.continuous_motion_2 = QThread()
        self.continuous_motion_2.start()


    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())[::-1]
        self.serial_selection_box.clear()
        self.serial_box.clear()
        self.serial_selection_box_EMG.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.serial_selection_box.addItem(port[0])
            self.serial_box.addItem(port[0])
            self.serial_selection_box_EMG.addItem(port[0])
        if len(self.Com_Dict) == 0:
            print(" 无串口")

    def port_imf(self):
        imf_s = self.serial_selection_box.currentText()
        if imf_s != "":
            print("serial_open", imf_s)

    def cybergear_open_1(self):
        motor_id = self.canID_box.currentText()
        port = self.serial_selection_box.currentText()
        print("motor_id, port", motor_id, port)
        self.motor1 = SerialControllerInterface(motor_id=int(motor_id),
                                                port=port)
        # try:
        #     self.motor1.enable_motor()
        # except:
        #     QMessageBox.critical(self, "cybergear Error", "此电机不能被打开！")
        #     return None
        self.motor1.enable_motor()
        self.open_button.setEnabled(False)
        self.close_button.setEnabled(True)
        # self.formGroupBox1.setTitle("串口状态（已开启）")

    def cyberger_execute_1(self):
        self.motor1_mode = self.mode_box.currentText()
        self.motor1_speed = self.speed_doubleSpinBox.value()
        self.motor1_position = self.posision_doubleSpinBox_1.value()
        self.motor1_torque = self.torque_doubleSpinBox_3.value()
        if self.motor1_mode == "运控模式":
            self.motor1.set_run_mode(RunModes.CONTROL_MODE)
            self.motor1.send_motor_control_command(torque=self.motor1_torque,
                                                   target_angle=self.motor1_position,
                                                   target_velocity=self.motor1_speed,
                                                   Kp=10, Kd=2)

        elif self.motor1_mode == "位置模式":
            self.motor1.set_run_mode(RunModes.POSITION_MODE)
            self.motor1.set_motor_position_control(limit_spd=self.motor1_speed,
                                                   loc_ref=self.motor1_position)
        elif self.motor1_mode == "速度模式":
            self.motor1.set_run_mode(RunModes.SPEED_MODE)
            self.motor1.write_single_param(param_name="limit_spd", value=self.motor1_speed)

        elif self.motor1_mode == "电流模式":
            self.motor1.set_run_mode(RunModes.CURRENT_MODE)

    def cybergear_stop_1(self):
        # self.timer.stop()
        try:
            self.motor1.disable_motor()
        except:
            QMessageBox.critical(self, "gear Error", "此电机不能被关闭！")
            return
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        print("gear状态（已关闭）")

    def data_send_timer_1(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
            self.data_send_flag_1 = True

        if self.timer_send_cb_2.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
            # self.data_send_flag_2 = True
        if not self.timer_send_cb_2.isChecked() and not self.timer_send_cb.isChecked():
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)



    def data_send(self):
        if self.timer_send_cb.isChecked():
            try:
                print("motor1_mode running")
                self.motor1_mode = self.mode_box.currentText()
                self.motor1_speed = self.speed_doubleSpinBox.value()
                self.motor1_position = self.posision_doubleSpinBox_1.value()
                self.motor1_torque = self.torque_doubleSpinBox_3.value()
                if self.motor1_mode == "运控模式":
                    # self.motor1.set_run_mode(RunModes.CONTROL_MODE)
                    self.motor1.send_motor_control_command(torque=self.motor1_torque,
                                                           target_angle=self.motor1_position,
                                                           target_velocity=self.motor1_speed,
                                                           Kp=10, Kd=2)
                elif self.motor1_mode == "位置模式":
                    # self.motor1.set_run_mode(RunModes.POSITION_MODE)
                    self.motor1.set_motor_position_control(limit_spd=self.motor1_speed,
                                                           loc_ref=self.motor1_position)
                elif self.motor1_mode == "速度模式":
                    # self.motor1.set_run_mode(RunModes.SPEED_MODE)
                    self.motor1.write_single_param(param_name="limit_spd", value=self.motor1_speed)

                elif self.motor1_mode == "电流模式":
                    ...
                    # self.motor1.set_run_mode(RunModes.CURRENT_MODE)
                else:
                    ...
            except:
                print("data_send 1 fail ")

        if self.timer_send_cb_2.isChecked():
            try:
                print("motor2_mode running")
                self.motor2_mode = self.mode_box_2.currentText()
                self.motor2_speed = self.speed_doubleSpinBox_2.value()
                self.motor2_position = self.position_doubleSpinBox_2.value()
                self.motor2_torque = self.torque_doubleSpinBox_2.value()
                if self.motor2_mode == "运控模式":
                    # self.motor1.set_run_mode(RunModes.CONTROL_MODE)
                    self.motor2.send_motor_control_command(torque=self.motor2_torque,
                                                           target_angle=self.motor2_position,
                                                           target_velocity=self.motor2_speed,
                                                           Kp=10, Kd=2)
                elif self.motor2_mode == "位置模式":
                    # self.motor1.set_run_mode(RunModes.POSITION_MODE)
                    self.motor2.set_motor_position_control(limit_spd=self.motor2_speed,
                                                           loc_ref=self.motor2_position)
                elif self.motor2_mode == "速度模式":
                    # self.motor1.set_run_mode(RunModes.SPEED_MODE)
                    self.motor2.write_single_param(param_name="limit_spd", value=self.motor2_speed)

                elif self.motor2_mode == "电流模式":
                    ...
                    # self.motor1.set_run_mode(RunModes.CURRENT_MODE)
                else:
                    ...
            except:
                print("data_send 2 fail ")

    def data_receive(self):
        try:
            self.motor1.result

        except:
            ...

    """
    二号电机功能设置
    """

    def cybergear_open_2(self):
        motor_id = self.canID_box_2.currentText()
        port = self.serial_box.currentText()
        print("motor_id, port", motor_id, port)
        self.motor2 = SerialControllerInterface(motor_id=int(motor_id),
                                                port=port)
        # try:
        #     self.motor1.enable_motor()
        # except:
        #     QMessageBox.critical(self, "cybergear Error", "此电机不能被打开！")
        #     return None
        self.motor2.enable_motor()
        self.open_button_2.setEnabled(False)
        self.close_button_2.setEnabled(True)
        # self.formGroupBox1.setTitle("串口状态（已开启）")

    def cyberger_execute_2(self):
        self.motor2_mode = self.mode_box_2.currentText()
        self.motor2_speed = self.speed_doubleSpinBox_2.value()
        self.motor2_position = self.position_doubleSpinBox_2.value()
        self.motor2_torque = self.torque_doubleSpinBox_2.value()
        if self.motor2_mode == "运控模式":
            self.motor2.set_run_mode(RunModes.CONTROL_MODE)
            self.motor2.send_motor_control_command(torque=self.motor2_torque,
                                                   target_angle=self.motor2_position,
                                                   target_velocity=self.motor2_speed,
                                                   Kp=10, Kd=2)

        elif self.motor2_mode == "位置模式":
            self.motor2.set_run_mode(RunModes.POSITION_MODE)
            self.motor2.set_motor_position_control(limit_spd=self.motor2_speed,
                                                   loc_ref=self.motor2_position)
        elif self.motor2_mode == "速度模式":
            self.motor2.set_run_mode(RunModes.SPEED_MODE)
            self.motor2.write_single_param(param_name="limit_spd", value=self.motor2_speed)

        elif self.motor2_mode == "电流模式":
            self.motor2.set_run_mode(RunModes.CURRENT_MODE)

    def cybergear_stop_2(self):
        try:
            self.motor2.disable_motor()
        except:
            QMessageBox.critical(self, "gear Error", "此电机不能被关闭！")
            return
        self.open_button_2.setEnabled(True)
        self.close_button_2.setEnabled(False)
        self.lineEdit_6.setEnabled(True)
        self.data_num_received_2 = 0
        self.lineEdit_4.setText(str(self.data_num_received_2))
        self.data_num_sended_2 = 0
        self.lineEdit_5.setText(str(self.data_num_sended_2))
        print("gear状态（已关闭）")


    def data_receive_draw(self):
        if self.timer_send_cb.isChecked():
            try:
                self.motor1_speed_list.append(self.motor1.result["vel"])
                self.motor1_position_list.append(self.motor1.result["pos"])
                self.motor1_torque_list.append(self.motor1.result["torque"])
                self.ax_1.clear()
                self.ax_1.plot(self.motor1_speed_list, color='#800080', linewidth=3.0, label="speed")
                self.ax_1.plot(self.motor1_position_list, color='#FFC0CB', linewidth=3.0, label="position")
                self.ax_1.plot(self.motor1_torque_list, color='#00FFFF', linewidth=3.0)
                if len(self.motor1_speed_list) >= 60:
                    self.motor1_speed_list.pop(0)
                    self.motor1_position_list.pop(0)
                    self.motor1_torque_list.pop(0)
                self.canvas_1.draw()
            except:
                print("data_receive_draw 1 fail")

        if self.timer_send_cb_2.isChecked():
            try:
                self.motor2_speed_list.append(self.motor2.result["vel"])
                self.motor2_position_list.append(self.motor2.result["pos"])
                self.motor2_torque_list.append(self.motor2.result["torque"])
                self.motor2_speed_list.append(self.motor2.result["vel"])
                self.motor2_position_list.append(self.motor2.result["pos"])
                self.motor2_torque_list.append(self.motor2.result["torque"])
                self.ax_2.clear()
                self.ax_2.plot(self.motor2_speed_list, color='#800080', linewidth=3.0)
                self.ax_2.plot(self.motor2_position_list, color='#FFC0CB', linewidth=3.0)
                self.ax_2.plot(self.motor2_torque_list, color='#00FFFF', linewidth=3.0)
                if len(self.motor2_speed_list) >= 60:
                    self.motor2_speed_list.pop(0)
                    self.motor2_position_list.pop(0)
                    self.motor2_torque_list.pop(0)
                # if len(self.motor2_position_list) >= 60:
                #
                # if len(self.motor2_torque_list) >= 60:
                #
                # self.canvas_2.draw()

            except:
                print("data_receive_draw 2 fail")

    def open_EMG(self):
        if self.open_button_EMG.isChecked():

            port = self.serial_selection_box_EMG.currentText()
            baudrate = int(self.canID_box_3.currentText())
            print("open_button_EMG", port, baudrate)
            try:
                self.ser_EMG = serial.Serial(port=self.serial_selection_box_EMG.currentText(),
                                         baudrate=int(self.canID_box_3.currentText()), timeout=1)
                self.timer_EMG.start(10)
                self.forearm_EMG = []
                self.upperarm_EMG = []
            except:
                QMessageBox.information(self, "error", "串口选错了哥们")
                self.open_button_EMG.setChecked(False)
        else:

            self.ser_EMG.close()
            self.timer_EMG.stop()

    def EMG_read(self):
        self.EMG_signal = self.ser_EMG.inWaiting()
        if self.EMG_signal>0:
            self.EMG_data = self.ser_EMG.readline().decode('utf-8').strip()
            self.EMG_forearm_data, self.EMG_upperarm_data = map(int, self.EMG_data.split(','))
            # print("EMG_forearm_data, EMG_upperarm_data", self.EMG_forearm_data, self.EMG_upperarm_data)
            self.forearm_EMG.append(self.EMG_forearm_data)
            self.upperarm_EMG.append(self.EMG_upperarm_data)
            len_upperarm_EMG = len(self.upperarm_EMG)
            if len_upperarm_EMG >= 240:
                self.upperarm_EMG.pop(0)
                self.forearm_EMG.pop(0)
            if len_upperarm_EMG >= 60:
                data_np = np.array(self.forearm_EMG[-60:])
                std_dev = np.std(data_np)
                self.EMG_forearm_activity.setText(str(std_dev))
            try:
                self.ax_EMG_1.clear()
                self.ax_EMG_2.clear()
                # self.ax_EMG_1.set_xlim(0, 240)
                # self.ax_EMG_1.set_ylim(-10, 2400)
                self.ax_EMG_1.plot(self.forearm_EMG, color='#800080', linewidth=1.0, label="speed")
                self.ax_EMG_2.plot(self.upperarm_EMG, color='#00FFFF', linewidth=1.0, label="speed")
                self.EMG_forearm_cuurent.setText(str(int(self.EMG_forearm_data)))
                self.EMG_upperarm_current.setText(str(int(self.EMG_upperarm_data)))
                self.EMG_canvas_1.draw()
                self.EMG_canvas_2.draw()
            except:
                print("pass")
    def EMG_close(self):
        if self.open_button_EMG.isChecked():
            try:
                self.ser_EMG.close()
                self.open_button_EMG.setChecked(False)
            except:
                print("did not open ser")
        else:
            QMessageBox.information(self, "error", "没开串口点什么关闭")




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = CybergearControl()
    myshow.show()
    sys.exit(app.exec_())
