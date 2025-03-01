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
import pyqtgraph

class EMG_serial_reader(QThread):
    data_ready = pyqtSignal(str)  # 定义一个信号，用于当数据准备好时发送

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None  # 串口对象，将在run方法中初始化
        self.running = False  # 线程运行标志

    def run(self):
        self.running = True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            while self.running:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8').strip()
                    self.data_ready.emit(data)  # 发出信号，传递数据
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        finally:
            if self.ser:
                self.ser.close()

    def stop(self):
        self.running = False  # 设置运行标志为False，以退出循环
class CybergearControl(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(CybergearControl, self).__init__()
        self.setupUi(self)
        self.init()
        self.init_timer()
        self.init_QdoubleBox()
        self.init_matplotlib_canvas()
        self.init_StyleSheet()
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

        self.help_forearm_box.stateChanged.connect(self.help_forearm)
        self.help_upperarm_box.stateChanged.connect(self.help_upperarm)
        # self.timer_send_cb.stateChanged.connect(self.data_send_timer_1)
        # self.timer_send_cb_2.stateChanged.connect(self.data_send_timer_1)
        # 定时器接收数据

        self.close_button_EMG.clicked.connect(self.EMG_close)

        self.listWidget_2.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)
    def init_timer(self):
        self.timer_draw = QTimer(self)  # 绘制图像
        self.timer_draw.timeout.connect(self.data_receive_draw)
        self.timer_draw.start(100)

        self.timer_send = QTimer(self)  # 连续动态参数运动
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer_1)
        self.timer_send_cb_2.stateChanged.connect(self.data_send_timer_1)

        self.timer_help_mode = QTimer(self)  # 连续动态参数运动
        self.timer_help_mode.timeout.connect(self.help_mode_Time)


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
        self.pen_red = pyqtgraph.mkPen(color="#c91315")
        self.pen_blue_line = pyqtgraph.mkPen(color="#a4def4")
        self.pen_SlateBlue_line = pyqtgraph.mkPen(color="#ac80ef")
        self.pen_OrangeRed_line = pyqtgraph.mkPen(color="#f36b2b")

        self.pen_red.setWidth(2)
        self.pen_blue_line.setWidth(2)
        self.pen_SlateBlue_line.setWidth(2)
        self.pen_OrangeRed_line.setWidth(2)

        self.motor1_speed_list = []
        self.motor1_position_list = []
        self.motor1_torque_list = []
        self.forearm_gear = pyqtgraph.plot()
        self.forearm_gear.addLegend()
        self.forearm_gear.setBackground("#fbeaea")
        self.display_verticalLayout_1.addWidget(self.forearm_gear)


        self.motor2_speed_list = []
        self.motor2_position_list = []
        self.motor2_torque_list = []
        # self.upperarm_gear = pyqtgraph.PlotWidget()
        self.upperarm_gear = pyqtgraph.plot()
        self.upperarm_gear.addLegend()
        self.upperarm_gear.setBackground("#fbeaea")

        # self.upperarm_gear.setLa
        self.display_verticalLayout_2.addWidget(self.upperarm_gear)


        self.EMG_1_list = []
        self.EMG1_plot_widget = pyqtgraph.PlotWidget()
        self.EMG1_plot_widget.setBackground("#fbeaea")
        self.display_verticalLayout_3.addWidget(self.EMG1_plot_widget)

        self.figure_EMG_2, self.ax_EMG_2 = plt.subplots()

        self.EMG_2_list = []
        self.EMG2_plot_widget = pyqtgraph.PlotWidget()
        self.EMG2_plot_widget.setBackground("#fbeaea")
        self.display_verticalLayout_4.addWidget(self.EMG2_plot_widget)

    def init_StyleSheet(self):
        self.stackedWidget.setStyleSheet("""QWidget {background-color: #f0eaeb;}""")
        self.checkbox.setStyleSheet(""" QPushButton{border: 2px solid #ae45b0;border-radius: 10px;}
                                        QPushButton:hover {background-color: #4549b0;}
                                        QPushButton:pressed {background-color: #9fa1d6;}""")
        self.open_button.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                           QPushButton:hover {background-color: #d69f9f;}
                                           QPushButton:pressed {background-color: #d69f9f;}""")
        self.close_button.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.open_button_2.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                           QPushButton:hover {background-color: #d69f9f;}
                                           QPushButton:pressed {background-color: #d69f9f;}""")
        self.close_button_2.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.s3__send_button.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.s3__send_button_2.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.s3__clear_button.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.s3__clear_button_2.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        self.close_button_EMG.setStyleSheet(""" QPushButton{border: 2px solid #f48080;border-radius: 10px;}
                                            QPushButton:hover {background-color: #d69f9f;}
                                            QPushButton:pressed {background-color: #d69f9f;}""")
        # self.serial_selection_box.setStyleSheet("""QComboBox {background-color: #ee9bb1;border: 2px solid #c0c0c0;border-radius: 5px;""")



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
        # self.data_num_received = 0
        # self.lineEdit.setText(str(self.data_num_received))
        # self.data_num_sended = 0
        # self.lineEdit_2.setText(str(self.data_num_sended))
        print("gear状态（已关闭）")

    def data_send_timer_1(self):
        if self.timer_send_cb.isChecked():
            self.motor1_mode = self.mode_box.currentText()
            if self.motor1_mode == "运控模式":
                self.motor1.set_run_mode(RunModes.CONTROL_MODE)
            elif self.motor1_mode == "位置模式":
                self.motor1.set_run_mode(RunModes.POSITION_MODE)
            elif self.motor1_mode == "速度模式":
                self.motor1.set_run_mode(RunModes.SPEED_MODE)
            elif self.motor1_mode == "电流模式":
                ...
                # self.motor1.set_run_mode(RunModes.CURRENT_MODE)
            else:
                ...
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)


        if self.timer_send_cb_2.isChecked():
            self.motor2_mode = self.mode_box.currentText()
            if self.motor2_mode == "运控模式":
                self.motor2.set_run_mode(RunModes.CONTROL_MODE)
            elif self.motor2_mode == "位置模式":
                self.motor2.set_run_mode(RunModes.POSITION_MODE)
            elif self.motor2_mode == "速度模式":
                self.motor2.set_run_mode(RunModes.SPEED_MODE)
            elif self.motor2_mode == "电流模式":
                ...
                # self.motor1.set_run_mode(RunModes.CURRENT_MODE)
            else:
                ...
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
            # self.data_send_flag_2 = True
        if not self.timer_send_cb_2.isChecked() and not self.timer_send_cb.isChecked():
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)



    def data_send(self):
        if self.timer_send_cb.isChecked():
            try:
                # print("motor1_mode running")
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
        # self.lineEdit_6.setEnabled(True)
        # self.data_num_received_2 = 0
        # self.lineEdit_4.setText(str(self.data_num_received_2))
        # self.data_num_sended_2 = 0
        # self.lineEdit_5.setText(str(self.data_num_sended_2))
        print("gear状态（已关闭）")


    def data_receive_draw(self):
        """
        定时器开启
        绘制电机反馈的数据
        :return:
        """
        if self.timer_send_cb.isChecked():
            self.motor1_speed_list.append(self.motor1.result["vel"])
            self.motor1_position_list.append(self.motor1.result["pos"])
            self.motor1_torque_list.append(self.motor1.result["torque"])
            self.forearm_cuurent_torque.setText(str(self.motor1.result["torque"]))
            self.EMG_forearm_cuurent_3.setText(str(self.motor1.result["temperature_celsius"]))
            self.forearm_gear.clear()
            self.forearm_gear.plot(self.motor1_speed_list, pen=self.pen_blue_line, name="速度")
            self.forearm_gear.plot(self.motor1_position_list, pen=self.pen_OrangeRed_line, name="位置")
            self.forearm_gear.plot(self.motor1_torque_list, pen=self.pen_SlateBlue_line, name="力矩")
            if len(self.motor1_speed_list) >= 60:
                self.motor1_speed_list.pop(0)
                self.motor1_position_list.pop(0)
                self.motor1_torque_list.pop(0)


        if self.timer_send_cb_2.isChecked():
            self.motor2_speed_list.append(self.motor2.result["vel"])
            self.motor2_position_list.append(self.motor2.result["pos"])
            self.motor2_torque_list.append(self.motor2.result["torque"])
            self.upperarm_cuurent_torque.setText(str(self.motor2.result["torque"]))
            self.upperarm_current_tem.setText(str(self.motor2.result["temperature_celsius"]))
            self.upperarm_gear.clear()
            self.upperarm_gear.plot(self.motor2_speed_list, pen=self.pen_blue_line, name="速度")
            self.upperarm_gear.plot(self.motor2_position_list, pen=self.pen_OrangeRed_line, name="位置")
            self.upperarm_gear.plot(self.motor2_torque_list, pen=self.pen_SlateBlue_line, name="力矩")
            if len(self.motor2_speed_list) >= 60:
                self.motor2_speed_list.pop(0)
                self.motor2_position_list.pop(0)
                self.motor2_torque_list.pop(0)
                # self.canvas_2.draw()

            # except:
            #     print("data_receive_draw 2 fail")

    def open_EMG(self):
        """
        开启读取肌电信号的线程
        打开单片机的串口
        :return:
        """
        if self.open_button_EMG.isChecked():
            port = self.serial_selection_box_EMG.currentText()
            baudrate = int(self.canID_box_3.currentText())
            self.forearm_std_dev = 0
            print("open_button_EMG", port, baudrate)
            try:
                self.serial_thread = EMG_serial_reader(port=port, baudrate=baudrate)  # 根据实际情况设置串口和波特率
                self.serial_thread.data_ready.connect(self.update_data)  # 连接信号与槽
                self.serial_thread.start()  # 启动线程
                self.EMG_data = None
                self.timer_EMG.start(10)
                self.forearm_EMG = []
                self.upperarm_EMG = []
            except:
                QMessageBox.information(self, "error", "串口选错了哥们")
                self.open_button_EMG.setChecked(False)
        else:
            self.timer_EMG.stop()
            self.serial_thread.stop()  # 停止线程
            self.serial_thread.wait()  # 等待线程结束（可选，但通常建议等待以避免资源泄露）

    def update_data(self, data):
        """
        线程回调函数
        线程读取到的串口数据回传
        :param data:
        :return:
        """
        self.EMG_data = data  # 更新标签显示接收
    def EMG_read(self):
        """
        定时器开启
        绘制读取到的数据
        :return:
        """
        if self.EMG_data is not None:
            self.EMG_forearm_data, self.EMG_upperarm_data = map(int, self.EMG_data.split(','))
            # print("EMG_forearm_data, EMG_upperarm_data", self.EMG_forearm_data, self.EMG_upperarm_data)
            self.forearm_EMG.append(self.EMG_forearm_data)
            self.upperarm_EMG.append(self.EMG_upperarm_data)
            len_upperarm_EMG = len(self.upperarm_EMG)
            if len_upperarm_EMG >= 240:
                self.EMG_forearm_cuurent.setText(str(self.forearm_EMG.pop(0)))
                self.EMG_upperarm_current.setText(str(self.upperarm_EMG.pop(0)))

            if len_upperarm_EMG >= 40:
                forearm_data_np = np.array(self.forearm_EMG[-30:])
                upperarm_data_np = np.array(self.upperarm_EMG[-30:])
                self.forearm_std_dev = np.std(forearm_data_np)
                self.upperarm_std_dev = np.std(upperarm_data_np)
                self.EMG_forearm_activity.setText(str(self.forearm_std_dev))
                self.EMG_upperarm_activity.setText(str(self.upperarm_std_dev))
            # try:
            self.EMG1_plot_widget.clear()
            self.EMG2_plot_widget.clear()

            self.EMG1_plot_widget.plot(self.forearm_EMG, pen=self.pen_red, name="EMG1")
            self.EMG2_plot_widget.plot(self.upperarm_EMG, pen=self.pen_red, name="EMG2")

                # self.EMG_canvas_2.draw()
            # except:
            #     print("pass")
    def EMG_close(self):
        """
        关闭读取肌电信号的线程
        关闭单片机的串口
        :return:
        """
        if self.open_button_EMG.isChecked():
            try:
                self.ser_EMG.close()
                self.open_button_EMG.setChecked(False)
            except:
                print("did not open ser")
        else:
            QMessageBox.information(self, "error", "没开串口点什么关闭")

    def help_forearm(self):
        """
        前臂助力模式
        打开助力模式定时器
        """
        self.set_position=self.motor1.result["pos"]
        if self.help_forearm_box.isChecked():
            self.timer_send_cb.setChecked(True)
            self.timer_help_mode.start(200)
        if not self.help_forearm_box.isChecked():
            self.timer_send_cb.setChecked(False)
        if not self.help_forearm_box.isChecked() and not self.help_upperarm_box.isChecked():
            self.timer_help_mode.stop()
    def help_upperarm(self):
        """
        后臂助力模式
        打开助力模式定时器
        """
        if self.help_upperarm_box.isChecked():
            self.timer_send_cb_2.setChecked(True)
            self.timer_help_mode.start(200)
        if not self.help_upperarm_box.isChecked():
            self.timer_send_cb_2.setChecked(False)
        if not self.help_upperarm_box.isChecked() and not self.help_forearm_box.isChecked():
            self.timer_help_mode.stop()

    def help_mode_Time(self):
        """
        助力模式主函数
        """
        if self.help_forearm_box.isChecked():
            if self.open_button_EMG.isChecked():
                curren_position_motor1 = self.motor1.result["pos"]
                curren_torque_motor1 = self.motor1.result["torque"]
                # curren_position_motor2 = self.motor2.result["pos"]
                # curren_torque_motor2 = self.motor2.result["torque"]
                print("curren_position", curren_position_motor1)
                """
                前臂肘关节助力
                """
                if self.forearm_std_dev >= 60 and curren_torque_motor1 <= -0.7:
                    self.speed_doubleSpinBox.setValue(0.5)
                    self.set_position = curren_position_motor1 + 0.05
                    if self.set_position > 2.5:
                        self.set_position = 2.5
                    self.posision_doubleSpinBox_1.setValue(self.set_position)
                elif self.forearm_std_dev >= 60 and curren_torque_motor1 >= 0.7:
                    self.speed_doubleSpinBox.setValue(0.5)
                    self.set_position = curren_position_motor1 - 0.05
                    if self.set_position < 0.5:
                        self.set_position = 0.5
                    self.posision_doubleSpinBox_1.setValue(self.set_position)
                # if self.set_position > 2.5:
                #     self.set_position = 2.5
                # if self.set_position < 0.5:
                #     self.set_position = 0.5
                # self.posision_doubleSpinBox_1.setValue(self.set_position)
                #
                # """
                # 大臂肩关节助力
                # """
                # if self.upperarm_std_dev >= 200 and curren_torque_motor2 <= -0.7:
                #     self.speed_doubleSpinBox_2.setValue(0.23)
                #     set_position = curren_torque_motor1 + 0.05
                #     if set_position > 0.9:
                #         set_position = 0.9
                #     self.position_doubleSpinBox_2.setValue(set_position)
                # elif self.forearm_std_dev >= 150 and curren_torque_motor1 >= 0.7:
                #     self.speed_doubleSpinBox_2.setValue(0.23)
                #     set_position = curren_torque_motor1 - 0.02
                #     if set_position < -0.2:
                #         set_position = -0.2
                #     self.position_doubleSpinBox_2.setValue(set_position)
        print("help_mode_Time is open", self.posision_doubleSpinBox_1.value())

    def switchPage(self, index):
        self.stackedWidget.setCurrentIndex(index)
    def depth_vision(self):
        ...




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = CybergearControl()
    myshow.show()
    sys.exit(app.exec_())
# 写到这真累啊
#。。。