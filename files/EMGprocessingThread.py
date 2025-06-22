
class EMG_serial_reader(QThread):
    # data_ready = pyqtSignal(str)  # 定义一个信号，用于当数据准备好时发送
    data_ready = pyqtSignal(dict)
    data_record = pyqtSignal(list)

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None  # 串口对象，将在run方法中初始化
        self.running = False  # 线程运行标志
        self.recording = False  # 记录标志
        self.record_angle = None  # 记录角度
        SAMPLE_RATE = 1000  # 采样率(Hz)
        LOWCUT = 20  # 带通下限频率
        HIGHCUT = 400  # 带通上限频率
        ORDER = 4  # 滤波器阶数
        self.WINDOW_SIZE = 200  # 分析窗口大小(ms)
        THRESHOLD = 0.2  # 肌肉激活阈值

        self.forearm_data_np = []
        self.upperarm_data_np = []  # deque(maxlen=int(SAMPLE_RATE * 1.5))  # 1.5秒缓存
        self.b, self.a = self.butter_bandpass(LOWCUT, HIGHCUT, SAMPLE_RATE, ORDER)
        self.forearm_filtered = {
            "filtered": None,
            "forearm_EMG": None,
            "std_dev": None
        }
        self.upperarm_filtered = {
            "filtered": None,
            "upperarm_EMG": None,
            "std_dev": None
        }
        self.EMG_data = {
            "forearm": None,
            "upperarm": None,
        }
        self.data_buffer = []
        self.csv_header = ["timestamp", "forearm_sEMG", "upperarm_sEMG", "joint_angle"]
        self.receive_msg = None

    def run(self):
        self.running = True

        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        while self.running:
            if self.ser.in_waiting > 0:
                try:
                    data = self.ser.readline().decode('utf-8').strip()

                    self.EMG_forearm_data, self.EMG_upperarm_data, self.EMG_shoulder = map(int, data.split(','))
                    self.forearm_data_np.append(self.EMG_forearm_data)
                    self.upperarm_data_np.append(self.EMG_upperarm_data)
                    self.forearm_filtered["forearm_EMG"] = self.EMG_forearm_data
                    self.upperarm_filtered["upperarm_EMG"] = self.EMG_upperarm_data
                    len_upperarm_EMG = len(self.upperarm_data_np)
                    if len_upperarm_EMG >= 60:
                        forearm_data_np = np.array(self.forearm_data_np[-50:])
                        upperarm_data_np = np.array(self.upperarm_data_np[-50:])
                        self.forearm_std_dev = np.std(forearm_data_np)
                        self.upperarm_std_dev = np.std(upperarm_data_np)
                        self.forearm_filtered["std_dev"] = self.forearm_std_dev
                        self.upperarm_filtered["std_dev"] = self.upperarm_std_dev
                        self.upperarm_data_np.pop()
                        self.forearm_data_np.pop()

                    self.EMG_data["forearm"] = self.forearm_filtered
                    self.EMG_data["upperarm"] = self.upperarm_filtered
                    self.data_ready.emit(self.EMG_data)  # 发出信号，传递数据
                    if self.recording is True:  # 记录数据
                        self.data_buffer.append([time.time(), 
                                                self.EMG_forearm_data, 
                                                self.EMG_upperarm_data, 
                                                self.receive_msg["record_angle"]
                                                ])
                    if self.recording is False and len(self.data_buffer) >= 100:
                        df = pd.DataFrame(self.data_buffer, columns=self.csv_header)
                        df.to_csv(self.receive_msg["file_path"], mode='a', index=False)
                        self.data_buffer.clear()
                        print("saved")
                except:
                    print("except: data, len(data)")
        else:
            self.ser.close()

    def stop(self):
        self.running = False  # 设置运行标志为False，以退出循环

    def handle_cmd(self, cmd):
        if cmd["flag"] is True:
            self.recording = True
        elif cmd["flag"] is False:
            self.recording = False
        self.receive_msg = cmd
        print("cmd, self.recording\n", cmd, self.recording)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def compute_features(self, signal):
        features = {}

        # 时域特征
        features['MAV'] = np.mean(np.abs(signal))  # 平均绝对值
        features['VAR'] = np.var(signal)  # 方差
        features['RMS'] = np.sqrt(np.mean(signal ** 2))  # 均方根值
        features['IEMG'] = np.sum(np.abs(signal))  # 积分肌电值

        # 过零率计算
        zero_cross = np.where(np.diff(np.sign(signal)))[0]
        features['ZC'] = len(zero_cross) / (len(signal) / 1000)  # 每秒过零次数

        return features