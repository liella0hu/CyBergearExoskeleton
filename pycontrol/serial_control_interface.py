import struct
import logging
import enum
import math
import serial
import time
import tqdm
import numpy

 # 通信类型
class CmdModes(enum.Enum):
    GET_DEVICE_ID = 0  # 获取设备ID的命令
    MOTOR_CONTROL = 1  # 电机控制命令
    MOTOR_FEEDBACK = 2  # 电机反馈命令
    MOTOR_ENABLE = 3  # 电机启用命令
    MOTOR_STOP = 4  # 电机停止命令
    SET_MECHANICAL_ZERO = 6  # 设置机械零点的命令
    SET_MOTOR_CAN_ID = 7  # 设置电机CAN ID的命令
    PARAM_TABLE_WRITE = 8  # 参数表写入命令
    SINGLE_PARAM_READ = 17  # 单参数读取命令
    SINGLE_PARAM_WRITE = 18  # 单参数写入命令
    FAULT_FEEDBACK = 21  # 故障反馈命令

# 控制模式
class RunModes(enum.Enum):
    CONTROL_MODE = 0  # 运控模式
    POSITION_MODE = 1  # 位置模式
    SPEED_MODE = 2  # 速度模式
    CURRENT_MODE = 3  # 电流模式



class SerialControllerInterface:
    
    PARAM_TABLE = {
        "motorOverTemp": {"feature_code": 0x200D, "type": "int16"},
        "overTempTime": {"feature_code": 0x200E, "type": "int32"},
        "limit_torque": {"feature_code": 0x2007, "type": "float"},
        "cur_kp": {"feature_code": 0x2012, "type": "float"},
        "cur_ki": {"feature_code": 0x2013, "type": "float"},
        "spd_kp": {"feature_code": 0x2014, "type": "float"},
        "spd_ki": {"feature_code": 0x2015, "type": "float"},
        "loc_kp": {"feature_code": 0x2016, "type": "float"},
        "spd_filt_gain": {"feature_code": 0x2017, "type": "float"},
        "limit_spd": {"feature_code": 0x2018, "type": "float"},
        "limit_cur": {"feature_code": 0x2019, "type": "float"},
    }

    PARAMETERS = {
        "run_mode": {"index": 0x7005, "format": "u8", "min": 0, "max": 3},
        "iq_ref": {"index": 0x7006, "format": "f", "min": -23.0, "max": 23.0},
        "spd_ref": {"index": 0x700A, "format": "f", "min": -30.0, "max": 30.0},
        "limit_torque": {"index": 0x700B, "format": "f", "min": 0.0, "max": 12.0},
        "cur_kp": {"index": 0x7010, "format": "f", "min": 0.0, "max": 500.0},
        "cur_ki": {"index": 0x7011, "format": "f", "min": 0.0, "max": 5.0},
        "cur_filt_gain": {"index": 0x7014, "format": "f", "min": 0.0, "max": 1.0},
        "loc_ref": {"index": 0x7016, "format": "f", "min": -4 * math.pi, "max": 4 * math.pi},
        "limit_spd": {"index": 0x7017, "format": "f", "min": 0.0, "max": 30.0},
        "limit_cur": {"index": 0x7018, "format": "f", "min": 0.0, "max": 23.0}
    }
    TWO_BYTES_BITS = 16
    def __init__(self, motor_id=1, main_can_id=253, port = 'COM16', baudrate = 921600, timeout=0.1):
        """
        初始化CAN电机控制器。

        参数:
        bus: CAN总线对象。
        motor_id: 电机的CAN ID。
        main_can_id: 主CAN ID。
        """
        self.MOTOR_ID = motor_id
        self.MAIN_CAN_ID = main_can_id
        self.P_MIN = -4 * math.pi
        self.P_MAX = 4 * math.pi
        self.V_MIN = -30.0
        self.V_MAX = 30.0
        self.T_MIN = -12.0
        self.T_MAX = 12.0
        self.KP_MIN, self.KP_MAX = 0.0, 500.0  # 0.0 ~ 500.0
        self.KD_MIN, self.KD_MAX = 0.0, 5.0  # 0.0 ~ 5.0
        self.serial = None
        self.port = port
        
    # 定义类的析构函数，关闭serialport
    def __del__(self):
        print(f"Motor {self.MOTOR_ID} 对象已销毁")
        if self.serial is not None:
            self.serial.close()
        
    def serial_wrapper(func):  # 自动管理串行通信的打开、关闭以及AT模式的进入
        def wrapper(self, *args, **kwargs):
            if self.serial is None:
                self.serial = serial.Serial(self.port, baudrate=921600, timeout=0.01)
                time.sleep(0.01)
            self.serial.readlines(); # 清空之前残留的buffer    
            self._enter_AT_mode()
            time.sleep(0.1)
            try:
                func(self, *args, **kwargs)
                time.sleep(0.1)
            finally:
                if self.serial is not None:
                    self.serial.close()
                    self.serial = None
                    time.sleep(0.1)
        return wrapper
    
    
    def _enter_AT_mode(self):
        """
        进入AT模式。
        """
        self.serial.write(bytes.fromhex('41 54 2b 41 54 0d 0a'))
        
    @serial_wrapper
    def enable_motor(self):
        """
        启用电机。
        
        返回:
        应答电机反馈帧
        """
        data_frame = self._encode_data(CmdModes.MOTOR_ENABLE.value)
        # 发送数据帧
        self.serial.write(data_frame)
        # 接收数据帧
        data = self.serial.readlines() 
        length = len(data)
        if length > 1 and  len(data[length-1].hex()) == 34:
            print("接收到enable应答数据帧:", data[length-1].hex())
            self._parse_received_msg(data[length-1],  format = "ba")
    
    @serial_wrapper        
    def disable_motor(self):
        """
        禁用电机。
        
        返回:
        应答电机反馈帧
        """
        data_frame = self._encode_data(CmdModes.MOTOR_STOP.value)
        # 发送数据帧
        self.serial.write(data_frame)
        time.sleep(0.1)
        # 接收数据帧
        data = self.serial.readlines() 
        length = len(data)
        if length > 1 and  len(data[length-1].hex()) == 34:
            print("接收到的disable应答数据帧:", data[length-1].hex())
            self._parse_received_msg(data[length-1],  format = "ba")
            
    @serial_wrapper        
    def set_motor_0position(self):
        """
        设置电机0位置。
        
        返回:
        应答电机反馈帧
        """
        index = bytearray(2)
        index[0] = 0x00
        index[1] = 0x01
        
        data_frame = self._encode_data(CmdModes.SET_MECHANICAL_ZERO.value, index= int.from_bytes(index, byteorder='big'))
        # 发送数据帧
        self.serial.write(data_frame)
        # 接收数据帧
        data = self.serial.readlines() 
        length = len(data)
        if length > 1 and  len(data[length-1].hex()) == 34:
            print("接收到的set0位置应答数据帧:", data[length-1].hex())
            return self._parse_received_msg(data[length-1], format = "ba")
        else:
            print(f"Failed to set motor 0 position. No response received.")
            return None
    
    @serial_wrapper        
    def write_single_param(self, param_name, value):
        """
        通过参数名称写入单个参数。

        参数:
        param_name: 参数名称。
        value: 要设置的值。

        返回:
        写入操作的结果。
        """
        if param_name in self.PARAMETERS:
            index = self.PARAMETERS[param_name]["index"]
            format = self.PARAMETERS[param_name]["format"]
            min = self.PARAMETERS[param_name]["min"]
            max = self.PARAMETERS[param_name]["max"]
            # 如果value的type是enum, 取其value
            if isinstance(value, enum.Enum):
                value = value.value
            data_frame = self._encode_data(CmdModes.SINGLE_PARAM_WRITE.value, index, format, value, min, max)
            # 发送数据帧
            self.serial.write(data_frame)
            # 接收数据帧
            data = self.serial.readlines() 
            length = len(data)
            if length > 1 and  len(data[length-1].hex()) == 34:
                print("接收到的数据帧:", data[length-1].hex())
                return self._parse_received_msg(data[length-1], "ba")
        else:
            print(f"Parameter {param_name} not found in parameters list.")
        
    @serial_wrapper
    def read_single_parameter(self, parameter_name):
        """
        读取单个参数的值。

        参数:
        parameter_name: 参数名称。

        返回:
        参数值。

        """
        if parameter_name in self.PARAMETERS:
            index = self.PARAMETERS[parameter_name]["index"]
            format = self.PARAMETERS[parameter_name]["format"]
            data_frame = self._encode_data(CmdModes.SINGLE_PARAM_READ.value, index, format)
            # 发送数据帧
            self.serial.write(data_frame)
            # 接收数据帧
            data = self.serial.readlines() 
            length = len(data)
            if length > 0 and  len(data[length-1].hex()) == 34:
                # print("接收到read_param的数据帧:", data[length-1].hex())
                return self._parse_received_msg(data[length-1], format)
        else:
            print(f"Parameter {parameter_name} not found in parameters list.")
            
    # 解析收到的17位bytes消息
    def _parse_received_msg(self, received_msg_data, format = 'f'):
        # 将数据转位字节
        received_msg_data_bytes = received_msg_data.hex()
        # print("recv msg is :", received_msg_data_bytes)
        # print("recv msg type is :", type(received_msg_data_bytes))
        # 将数据放到bytearray中
        received_msg_data_bytes = bytearray.fromhex(received_msg_data_bytes)
        # print("recv msg type is :", type(received_msg_data_bytes))
        # 扩展can id
        ex_can_id = self._decode_canid(received_msg_data_bytes[2:6])
        print("ex_can_id is :", int(ex_can_id[2]))
        # 数据帧
        data = received_msg_data_bytes[7:15]
        result = self._decode_8_bytes_data(data, format)
        
        if type(result) == dict:
            print("pos", result["pos"])
            print("vel", result["vel"])
            print("torque", result["torque"])
            print("temperature_celsius", result["temperature_celsius"])
        else:
            print("reuslt is :", result)
        return  result
          
        
    # 组合Serial 17字节数据帧
    def _encode_data(self, cmd_mode,  index = None, format=None, value = None, min = None, max = None, bit29 = None):
        data_frame = bytearray(17)
        # 帧头两个字节
        data_frame[0] = 0x41
        data_frame[1] = 0x54
        if bit29 == None:
            # 计算4个字节长度的ex_can_id
            ex_can_id = self._encode_canid(cmd_mode)
            # 将ex_can_id转换为字节，并填充到字节数组中
            data_frame[2:6] = ex_can_id.to_bytes(4, byteorder='big')
        else:
            data_frame[2:6] = bit29
        # 数据位个数放到第六位上
        data_frame[6] = 0x08
        # 组合数据帧
        if index == None and value != None:
            data_frame[7:15] = value
            
        else:
            data_frame[7:15] = self._encode_8_bytes_data(index, format, value, min, max)
        
        # 帧尾两个字节
        data_frame[15] = 0x0d
        data_frame[16] = 0x0a
        # print("发送的字节数组:", data_frame.hex())
        return data_frame
    
    # 组合Serial 4字节扩展帧canid
    def _encode_canid(self, cmd_mode):
        # 0. 声明一个字节数组，长度为4
        canid = bytearray(4)
        # 1. 将通讯模式cmd_mode转换为字节
        canid[0] = cmd_mode
        # 2. 填充0x00到第二个字节
        canid[1] = 0x00
        # 3. 将主机canid转换为字节，并填充到字节数组中
        canid[2] = self.MAIN_CAN_ID
        # 4. 将电机canid转换为字节，并填充到字节数组中
        canid[3] = self.MOTOR_ID
        # 5. 用16进制形式打印组合后的字节数组
        # print("组合扩展帧canid后的字节数组:", canid.hex())
        # 6. 将字节数组左移三位并把第二位置为1
        canid = int.from_bytes(canid, byteorder='big')
        canid = canid << 3 | 0x04
        # print("最终的扩展帧canid字节数组:", hex(canid))
        return canid
    
    def _decode_canid(self, ex_can_id):
        # 判断ex_can_id的type是否是bytearray
        if type(ex_can_id) == bytearray:
            ex_can_id = int.from_bytes(ex_can_id, byteorder='big')
        canid = ex_can_id >> 3
        # print("解析后的canid字节数组: ", hex(canid))
        # 将int转换为bytearray
        return canid.to_bytes(4, byteorder='big')
        
    # 组合Serial 8字节数据帧
    def _encode_8_bytes_data(self, index, format="f", value = None, x_min = None, x_max = None):
        data_frame = bytearray(8)
        # 将index转换为字节，并填充到字节数组中,
        if(index != None):
            data_frame[0:2] = index.to_bytes(2, byteorder='little')
        data_frame[2] = 0x00
        data_frame[3] = 0x00
        
        # 将value转换为字节，并填充到字节数组中
        if(value != None and format != "ba"):
            value = max(min(value, x_max), x_min)
            if format == "u8": 
                data_frame[4] = value
            if format == "f":       
                data_frame[4:8] = struct.pack('f', value)
        elif (value != None and format == "ba"):
            data_frame[4:8] = value
        else:
            print("value is None")
        return data_frame
    
    def _decode_8_bytes_data(self, data : bytearray, format="f"): 
        if format == "u8":
            # 将字节转位uint8
            return data[4]
        if format == "f":       
            # 将字节数组转换为浮点数
            return struct.unpack('f', data[4:8])[0]
        if format == "ba":
            # 将前两个字节[0-65535]映射到[-4pi , 4pi]
            pos =  self._uint_to_float(
                (data[0] << 8) + data[1], self.P_MIN, self.P_MAX, self.TWO_BYTES_BITS
            )
            vel = self._uint_to_float(
                (data[2] << 8) + data[3], self.V_MIN, self.V_MAX, self.TWO_BYTES_BITS
            )
            torque = self._uint_to_float(
                (data[4] << 8) + data[5], self.T_MIN, self.T_MAX, self.TWO_BYTES_BITS
            )
            temperature_raw = (data[6] << 8) + data[7]
            temperature_celsius = temperature_raw / 10.0
            # 返回字典型数据
            return {
                "pos": pos,
                "vel": vel,
                "torque": torque,
                "temperature_celsius": temperature_celsius,
            }

    def _uint_to_float(self, x, x_min, x_max, bits):
        """
        将无符号整数转换为浮点数。

        参数:
        x: 输入的无符号整数。
        x_min: 可接受的最小浮点数。
        x_max: 可接受的最大浮点数。
        bits: 输入无符号整数的位数。

        返回:
        转换后的浮点数。
        """
        span = (1 << bits) - 1
        offset = x_max - x_min
        x = max(min(x, span), 0)  # Clamp x to the range [0, span]
        return offset * x / span + x_min

    def _linear_mapping(
        self, value, value_min, value_max, target_min=0, target_max=65535
    ):
        """
        对输入值进行线性映射。

        参数:
        value: 输入值。
        value_min: 输入值的最小界限。
        value_max: 输入值的最大界限。
        target_min: 输出值的最小界限。
        target_max: 输出值的最大界限。

        返回:
        映射后的值。
        """
        return int(
            (value - value_min) / (value_max -
                                   value_min) * (target_max - target_min)
            + target_min
        )

    

    def _pack_conrol_8bytes(self, target_angle, target_velocity, Kp, Kd):
        """
        定义打包data1函数, 将控制参数打包为8字节的数据。

        参数:
        target_angle: 目标角度。
        target_velocity: 目标速度。
        Kp: 比例增益。
        Kd: 微分增益。

        返回:
        8字节的数据。
        """
        # 对输入变量进行线性映射
        target_angle_mapped = self._linear_mapping(
            target_angle, self.P_MIN, self.P_MAX)
        target_velocity_mapped = self._linear_mapping(
            target_velocity, self.V_MIN, self.V_MAX
        )
        Kp_mapped = self._linear_mapping(Kp, self.KP_MIN, self.KP_MAX)
        Kd_mapped = self._linear_mapping(Kd, self.KD_MIN, self.KD_MAX)

        # 使用Python的struct库进行打包
        # 使用H表示无符号短整数(2字节), 共需要8字节
        data1_bytes = struct.pack(
            "HHHH", target_angle_mapped, target_velocity_mapped, Kp_mapped, Kd_mapped
        )
        # print(type(data1_bytes))
        # print(data1_bytes.hex())
        # bytes转为bytearray
        data1 = bytearray(8)
        # for i in range(8):
        #     data1[i] = data1_bytes[i]
        data1[0] = data1_bytes[1]
        data1[1] = data1_bytes[0]
        data1[2] = data1_bytes[3]
        data1[3] = data1_bytes[2]
        data1[4] = data1_bytes[5]
        data1[5] = data1_bytes[4]
        data1[6] = data1_bytes[7]
        data1[7] = data1_bytes[6]
        
        return data1

    @serial_wrapper
    def set_run_mode(self, mode):
        """
        设置运行模式。

        参数:
        mode: 运行模式，应为 RunModes 枚举的一个实例。

        返回:
        写入操作的结果。
        """
        if not isinstance(mode, RunModes):
            raise ValueError(
                f"Invalid mode: {mode}. Must be an instance of RunModes enum.")
        return self.write_single_param("run_mode", value=mode.value)

    @serial_wrapper
    def set_motor_position_control(self, limit_spd, loc_ref, torque=5):
        """
        位置模式下设置电机的位置控制参数。

        参数:
        limit_spd: 电机的最大速度。
        loc_ref: 电机的目标位置。

        返回:
        None。
        """
        # 设置电机最大速度
        self.write_single_param(param_name="limit_spd", value=limit_spd)
        # 设置电机目标位置
        self.write_single_param(param_name="loc_ref", value=loc_ref)
        # 设置电机最大扭矩
        self.write_single_param(param_name="")

    @serial_wrapper
    def send_motor_control_command(
        self, torque, target_angle, target_velocity, Kp, Kd
    ):
        """
        运控模式下发送电机控制指令。

        参数:
        torque: 扭矩。
        target_angle: 目标角度。
        target_velocity: 目标速度。
        Kp: 比例增益。
        Kd: 导数增益。

        返回:
        解析后的接收消息。
        """

        # 生成29位的仲裁ID的组成部分
        cmd_mode = CmdModes.MOTOR_CONTROL
        torque_mapped = self._linear_mapping(
            torque, self.T_MIN, self.T_MAX, target_min=0, target_max=65535)
        # 0. 声明一个字节数组，长度为4
        canid = bytearray(4)
        # 1. 将通讯模式cmd_mode转换为字节
        canid[0] = cmd_mode.value
        canid[1:3] = bytearray(struct.pack(">H",torque_mapped))
        canid[3] = self.MOTOR_ID
        # 将字节数组左移三位并把第二位置为1
        canid = int.from_bytes(canid, byteorder='big')
        canid = canid << 3 | 0x04
        canid = canid.to_bytes(4, byteorder='big')
        # 2. 将目标角度和目标速度转换为字节数组
        value = self._pack_conrol_8bytes(target_angle, target_velocity, Kp, Kd)
        # print("发送的电机控制数据帧:", canid.hex(), value.hex())
        # 3. 构造数据帧
        data_frame = self._encode_data(cmd_mode = cmd_mode.value, bit29 = canid, value = value)
        # 发送数据帧
        self.serial.write(data_frame)
        # 接收数据帧
        data = self.serial.readlines() 
        length = len(data)
        if length > 0:
            # print("接收到的set0位置应答数据帧:", data[length-1].hex())
            return self._parse_received_msg(data[length-1], format = "ba")
        else:
            # print(f"Failed to set motor 0 position. No response received.")
            return None
        

  
        

if __name__ == "__main__":
    motor1 = SerialControllerInterface(1, port="COM16")
    motor2 = SerialControllerInterface(2, port="COM16")
    motor1.enable_motor()
    motor2.enable_motor()
    # motor1.set_motor_0position()
    # motor2.set_motor_0position()
    motor2.set_run_mode( RunModes.POSITION_MODE)
    motor1.set_run_mode( RunModes.POSITION_MODE)
    # time.sleep(1)
    #
    # motor2.set_motor_position_control(3.0, -9)
    # motor1.set_motor_position_control(3.0, 9)
    # time.sleep(1.5)
    motor1.set_motor_position_control(2.0, 4)
    # motor2.write_single_param(param_name="limit_spd", value=10)
    motor2.set_motor_position_control(2.0, 7)
    #
    time.sleep(2)
    # motor1.set_run_mode(RunModes.CONTROL_MODE)
    # motor1.send_motor_control_command(torque=0.1, target_angle=1, target_velocity=2, Kp=0.001, Kd=0.00001)
    # time.sleep(30)
    # motor2.set_motor_0position()
    # limit_torque = motor2.read_single_parameter(parameter_name="limit_torque")
    # print("limit_torque", limit_torque)
    # time.sleep(1)
    motor1.disable_motor()
    motor2.disable_motor()

    
    del motor1
    del motor2
