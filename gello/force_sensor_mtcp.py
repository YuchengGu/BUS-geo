import time
from pymodbus.client import ModbusTcpClient

class ForceSensorMTCP:
    def __init__(self, ip, port=502, device_id=1):
        """
        初始化力传感器客户端
        :param ip: 变送器IP地址
        :param port: Modbus端口 (默认502)
        :param slave: 从站ID (默认1)
        """
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.client = ModbusTcpClient(self.ip, port=self.port)
        
        # 默认小数点位数 (防止初始化读取失败时无法计算)
        self.decimals = {'force': 2, 'torque': 2}
        self.is_connected = False

    def connect(self):
        """建立连接并读取一次参数配置"""
        if self.client.connect():
            self.is_connected = True
            print(f"[系统] 已连接到力传感器 {self.ip}")
            self._update_decimals() # 连接成功后自动同步一次小数点
            return True
        else:
            self.is_connected = False
            print(f"[错误] 无法连接到 {self.ip}")
            return False

    def disconnect(self):
        """断开连接"""
        self.client.close()
        self.is_connected = False
        print("[系统] 连接已关闭")

    def _update_decimals(self):
        """(内部方法) 读取力和小数点的位参数"""
        try:
            # 地址 0x616 (1558) 读取4个寄存器 (包含力小数点和力矩小数点)
            result = self.client.read_holding_registers(address=0x616, count=4, device_id=self.device_id)
            if not result.isError():
                # 使用 pymodbus v3.x+ 新标准
                raw_params = self.client.convert_from_registers(
                    result.registers, 
                    data_type=self.client.DATATYPE.INT32, 
                    word_order="big"
                )
                self.decimals['force'] = raw_params[0]
                self.decimals['torque'] = raw_params[1]
                print(f"[配置] 小数点同步成功: 力={self.decimals['force']}, 力矩={self.decimals['torque']}")
            else:
                print("[警告] 读取小数点参数失败，将使用默认值 (2)")
        except Exception as e:
            print(f"[异常] 更新参数时出错: {e}")

    def read_values(self):
        """
        读取当前6维力/力矩数据
        :return: 包含6个浮点数的列表 [Fx, Fy, Fz, Mx, My, Mz] 或 None (如果失败)
        """
        if not self.is_connected:
            print("[错误] 未连接设备")
            return None

        try:
            # 读取 0xA00 (2560) 开始的12个寄存器 (6个INT32)
            result = self.client.read_holding_registers(address=0xA00, count=12, device_id=self.device_id)
            
            if result.isError():
                print(f"[通信错误] Modbus返回错误")
                return None

            # 转换数据
            raw_values = self.client.convert_from_registers(
                result.registers,
                data_type=self.client.DATATYPE.INT32,
                word_order="big"
            )

            # 应用缩放系数
            real_values = []
            f_scale = 10 ** self.decimals['force']
            t_scale = 10 ** self.decimals['torque']

            for i, val in enumerate(raw_values):
                if i < 3: # 前3个是力 (Fx, Fy, Fz)
                    real_values.append(round(val / f_scale, 3))
                else:     # 后3个是力矩 (Mx, My, Mz)
                    real_values.append(round(val / t_scale, 3))
            
            return real_values

        except Exception as e:
            print(f"[异常] 读取数据出错: {e}")
            return None