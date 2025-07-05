import numpy as np
import heapq
from collections import defaultdict


class Business:
    def __init__(self, id, compute_time, data_matrix, communication_time=None):
        """
        初始化业务
        :param id: 业务ID
        :param compute_time: 计算时间 (秒)
        :param data_matrix: 数据矩阵 (总pod数×总pod数，对角线为0)
        :param communication_time: 预先计算好的通信时间
        """
        self.id = id
        self.compute_time = compute_time
        self.data_matrix = data_matrix
        self.iteration_count = 0
        self.total_iteration_time = 0.0
        self.current_state = 'idle'  # 'idle', 'computing', 'communicating'
        self.remaining_time = 0.0
        # 预先计算通信时间（如果未提供）
        self.communication_time = communication_time if communication_time is not None \
            else 0

    def calculate_communication_time(self, num_ports_per_pod, link_capacity):
        """
        简化的连接分配算法：
        1. 初始为所有有流量的pod对分配一条连接
        2. 计算各连接的通信时间矩阵
        3. 找到时间最长的连接，尝试增加一条连接
        4. 重复直到无法添加连接
        """
        # 确保对角线为0
        # np.fill_diagonal(self.data_matrix, 0)
        num_pods = self.data_matrix.shape[0]

        # 初始化连接矩阵：有流量的地方分配1条连接
        connection_matrix = (self.data_matrix > 0).astype(int)

        # 计算端口使用情况
        port_usage = np.sum(connection_matrix, axis=1) + np.sum(connection_matrix, axis=0)

        # 迭代优化连接分配
        while True:
            # 计算当前各连接的通信时间
            with np.errstate(divide='ignore', invalid='ignore'):
                comm_times = np.where(connection_matrix > 0,
                                      self.data_matrix / (connection_matrix * link_capacity),
                                      0)

            # 找到最耗时的连接
            max_time = np.max(comm_times)
            if max_time <= 0:
                break  # 没有通信需求

            i, j = np.unravel_index(np.argmax(comm_times), comm_times.shape)

            # 尝试增加一条连接
            if port_usage[i] < num_ports_per_pod and port_usage[j] < num_ports_per_pod:
                connection_matrix[i, j] += 1
                port_usage[i] += 1
                port_usage[j] += 1
            else:
                break  # 无法再添加连接

        # 最终通信时间由最长的连接决定
        final_max_time = np.max(comm_times) if np.any(comm_times > 0) else 0
        return final_max_time

    def start_iteration(self, current_time):
        """开始一次新的迭代"""
        self.iteration_count += 1
        self.current_state = 'computing'
        self.remaining_time = self.compute_time
        return f"Business {self.id} started computing at {current_time:.3f}s"

    def update(self, delta_time):
        """
        更新业务状态
        :return: 如果状态改变，返回状态改变信息，否则返回None
        """
        if self.current_state == 'idle':
            return None

        self.remaining_time -= delta_time
        if self.remaining_time <= 0:
            if self.current_state == 'computing':
                # 计算完成，转为通信状态
                self.current_state = 'communicating'
                self.remaining_time = self.communication_time
                return 'switch_to_communicating'
            elif self.current_state == 'communicating':
                # 通信完成，迭代结束
                self.current_state = 'idle'
                return 'iteration_complete'
        return None


class OpticalNetwork:
    def __init__(self, num_pods, num_ports, recon_time):
        """
        初始化光网络
        :param num_pods: pod总数
        :param num_ports: 端口数
        :param recon_time: 重构时间 (秒)
        """
        self.num_pods = num_pods
        self.num_ports = num_ports
        self.recon_time = recon_time
        self.reconfiguring = False
        self.reconfig_remaining = 0.0
        self.active_business = None  # 当前正在通信的业务

    def can_start_communication(self, business):
        """检查是否可以开始通信（非抢占式，一次只能一个业务通信）"""
        return self.active_business is None

    def start_reconfiguration(self, business):
        """开始网络重构"""
        if not self.reconfiguring and self.active_business is None:
            self.reconfiguring = True
            self.reconfig_remaining = self.recon_time
            self.active_business = business
            return True
        return False

    def start_communication(self, business):
        """开始通信（不经过重构）"""
        if self.active_business is None:
            self.active_business = business
            return True
        return False

    def complete_communication(self):
        """完成通信"""
        self.active_business = None

    def update_reconfiguration(self, delta_time):
        """更新重构状态"""
        if self.reconfiguring:
            self.reconfig_remaining -= delta_time
            if self.reconfig_remaining <= 0:
                self.reconfiguring = False
                return True  # 重构完成
        return False


def simulate(businesses, network, total_simulation_time):
    """
    运行仿真
    :param businesses: 业务列表
    :param network: 光网络
    :param total_simulation_time: 总仿真时间 (秒)
    :return: 平均迭代时间
    """
    current_time = 0.0
    event_queue = []

    # 初始化所有业务
    for business in businesses:
        heapq.heappush(event_queue, (0.0, 'start_iteration', business.id))

    while current_time < total_simulation_time and event_queue:
        event_time, event_type, business_id = heapq.heappop(event_queue)
        current_time = event_time

        business = next(b for b in businesses if b.id == business_id)

        if event_type == 'start_iteration':
            # 开始计算
            # print(business.start_iteration(current_time))
            # 安排计算完成事件
            heapq.heappush(event_queue,
                           (current_time + business.compute_time,
                            'compute_complete',
                            business.id))
            network.active_business = None
        elif event_type == 'compute_complete':
            # 计算完成，尝试开始通信
            if network.can_start_communication(business):
                # 在通信开始前进行网络重构
                # if network.start_reconfiguration(business):
                # print(f"Network reconfiguration started at {current_time:.3f}s for business {business.id}")
                # 安排重构完成事件
                heapq.heappush(event_queue,
                               (current_time + network.recon_time + business.communication_time,
                                'start_iteration',
                                business.id))
                network.active_business = True
                business.iteration_count += 1
                # else:
                #     # 直接开始通信
                #     network.start_communication(business)
                #     print(f"Business {business.id} started communication at {current_time:.3f}s")
                #     # 安排通信完成事件
                #     heapq.heappush(event_queue,
                #                    (current_time + business.communication_time,
                #                     'communication_complete',
                #                     business.id))
            else:
                # 无法立即开始通信，稍后重试
                heapq.heappush(event_queue,
                               (current_time + 0.01,  # 0.1秒后重试
                                'start_iteration',
                                business.id))

        # elif event_type == 'comm_complete':
        #     # 重构完成，开始通信
        #     # print(f"Network reconfiguration completed at {current_time:.3f}s for business {business.id}")
        #     # print(f"Business {business.id} started communication at {current_time:.3f}s")
        #     heapq.heappush(event_queue,
        #                    (current_time + business.communication_time,
        #                     'start_iteration',
        #                     business.id))

        # elif event_type == 'communication_complete':
        #     # 通信完成，迭代结束
        #     iteration_time = business.compute_time + business.communication_time
        #     # if network.reconfiguring:
        #     iteration_time += network.recon_time
        #
        #     business.total_iteration_time += iteration_time
        #     network.complete_communication()
        #     business.iteration_count += 1
        #     # print(f"Business {business.id} completed iteration at {current_time:.3f}s, "
        #     #       f"iteration time: {iteration_time:.3f}s")
        #
        #     # 安排下一次迭代
        #     heapq.heappush(event_queue,
        #                    (current_time,  # 微小延迟后开始下一次迭代
        #                     'start_iteration',
        #                     business.id))

    # 计算平均迭代时间
    avg_iteration_times = []
    for business in businesses:
        if business.iteration_count > 0:
            avg_time = total_simulation_time / business.iteration_count
            avg_iteration_times.append(avg_time)
            print(f"Business {business.id}: {business.iteration_count} iterations, "
                  f"avg iteration time: {avg_time:.3f}s")

    if avg_iteration_times:
        overall_avg = sum(avg_iteration_times) / len(avg_iteration_times)
        print(f"\nOverall average iteration time: {overall_avg:.3f}s")
        return overall_avg
    else:
        print("No iterations completed during simulation")
        return 0.0


def create_data_matrix(num_pods, density=0.3):
    """
    创建数据矩阵（总pod数×总pod数，对角线为0）
    :param num_pods: pod总数
    :param density: 非零元素密度
    :return: 数据矩阵
    """
    matrix = np.random.rand(num_pods, num_pods) * 10
    np.fill_diagonal(matrix, 0)  # 对角线置为0
    # 随机将部分元素置为0
    mask = np.random.rand(num_pods, num_pods) > density
    matrix[mask] = 0
    return matrix


def precompute_communication_times(businesses, num_ports_per_pod, link_capacity):
    """预先为所有业务计算通信时间"""
    for business in businesses:
        business.communication_time = business.calculate_communication_time(
            num_ports_per_pod=num_ports_per_pod,
            link_capacity=link_capacity
        )


# 示例用法
if __name__ == "__main__":
    # 网络参数
    NUM_PODS = 8
    NUM_PORTS = 16
    RECON_TIME = 0.15  # 150ms重构时间
    LINK_CAPACITY = 10.0  # 单条连接带宽

    # 创建网络
    network = OpticalNetwork(num_pods=NUM_PODS, num_ports=NUM_PORTS, recon_time=RECON_TIME)

    # 创建业务列表
    businesses = []
    np.random.seed(42)

    # 创建5个业务
    for i in range(5):
        compute_time = 0.2 + np.random.rand() * 0.6  # 200-800ms计算时间
        data_matrix = create_data_matrix(NUM_PODS, density=0.4)
        businesses.append(Business(i, compute_time, data_matrix))

    # 预先计算所有业务的通信时间，考虑端口限制和连接带宽
    precompute_communication_times(businesses,
                                   num_ports_per_pod=NUM_PORTS,
                                   link_capacity=LINK_CAPACITY)

    # 打印各业务的预计算通信时间
    for business in businesses:
        print(f"Business {business.id}: precomputed communication time = {business.communication_time:.3f}s")

    # 运行仿真
    simulate(businesses, network, total_simulation_time=100.0)
