import numpy as np
import heapq
from collections import defaultdict

class Business:
    def __init__(self, id, compute_time, data_matrix):
        self.id = id
        self.compute_time = compute_time
        self.data_matrix = data_matrix
        self.communication_time = 0
        self.iteration_count = 0
        self.last_comm_time = 0
        self.total_wait_time = 0
        self.total_service_time = 0  # 累计服务时间（用于计算响应比的分母）

    def calculate_communication_time(self, link_matrix):
        """与FIFO示例完全相同的通信时间计算逻辑"""
        num_pods = self.data_matrix.shape[0]
        # connection_matrix = (self.data_matrix > 0).astype(int)
        # port_usage = np.sum(connection_matrix, axis=1) + np.sum(connection_matrix, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            comm_times = np.where(
                self.data_matrix > 0,
                self.data_matrix / link_matrix,
                0
            )

        # while True:
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         comm_times = np.where(
        #             connection_matrix > 0,
        #             self.data_matrix / (connection_matrix * link_capacity),
        #             0
        #         )
        #     max_time = np.max(comm_times)
        #     if max_time <= 0:
        #         break
        #
        #     i, j = np.unravel_index(np.argmax(comm_times), comm_times.shape)
        #     if port_usage[i] < num_ports_per_pod and port_usage[j] < num_ports_per_pod:
        #         connection_matrix[i, j] += 1
        #         port_usage[i] += 1
        #         port_usage[j] += 1
        #     else:
        #         break

        self.communication_time = np.max(comm_times) if np.any(comm_times > 0) else 0
        return self.communication_time

    @property
    def total_iteration_time(self):
        return self.compute_time + self.communication_time

    def response_ratio(self, current_time):
        """计算响应比：(等待时间 + 预计服务时间) / 预计服务时间"""
        wait_time = current_time - self.last_comm_time
        expected_service_time = self.communication_time + self.compute_time
        return (wait_time + expected_service_time) / expected_service_time if expected_service_time > 0 else 0


class OpticalNetwork:
    def __init__(self, link_matrix, recon_time):
        self.recon_time = recon_time
        self.active_business = None


def simulate_hrrn_with_starvation(
        businesses,
        network,
        total_simulation_time,
        starvation_threshold=5.0
):
    """HRRN调度+饥饿补偿（非抢占式）"""
    current_time = 0.0
    event_queue = []
    waiting_businesses = []  # 业务列表（非堆结构，需动态计算响应比）
    starving_businesses = []  # 饥饿业务列表

    # 初始化：所有业务加入等待队列
    waiting_businesses = businesses.copy()

    while current_time < total_simulation_time and (event_queue or waiting_businesses or starving_businesses):
        # 检测饥饿业务（等待时间超过阈值）
        current_starving = [
            biz for biz in waiting_businesses
            if current_time - biz.last_comm_time >= starvation_threshold
        ]
        for biz in current_starving:
            starving_businesses.append(biz)
            waiting_businesses.remove(biz)

        # 网络空闲时的调度决策
        if network.active_business is None:
            # 选择优先级最高的业务（HRRN核心逻辑）
            candidate_biz = None
            max_ratio = -1
            is_starving = False

            # 优先检查饥饿业务
            if starving_businesses:
                for biz in starving_businesses:
                    ratio = biz.response_ratio(current_time)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        candidate_biz = biz
                is_starving = True
            # 检查普通等待队列
            elif waiting_businesses:
                for biz in waiting_businesses:
                    ratio = biz.response_ratio(current_time)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        candidate_biz = biz

            if candidate_biz:
                # 从原队列移除
                if is_starving:
                    starving_businesses.remove(candidate_biz)
                else:
                    waiting_businesses.remove(candidate_biz)

                # 执行业务调度
                candidate_biz.total_wait_time += max(0, current_time - candidate_biz.last_comm_time)
                candidate_biz.last_comm_time = current_time

                # 通信事件（含重构时间）
                comm_end_time = current_time + network.recon_time + candidate_biz.communication_time
                heapq.heappush(event_queue, (comm_end_time, 'comm_complete', candidate_biz.id))
                network.active_business = candidate_biz

                # print(f"{'⚠️ 饥饿' if is_starving else '⏳ HRRN'}调度 biz_{candidate_biz.id} | "
                #       f"响应比={max_ratio:.2f} | "
                #       f"等待={current_time - candidate_biz.last_comm_time:.2f}s")

        # 处理事件（必须推进时间）
        if not event_queue:
            current_time += 0.001
            continue

        event_time, event_type, biz_id = heapq.heappop(event_queue)
        current_time = event_time
        biz = next(b for b in businesses if b.id == biz_id)

        if event_type == 'comm_complete':
            network.active_business = None
            compute_end_time = current_time + biz.compute_time
            heapq.heappush(event_queue, (compute_end_time, 'compute_complete', biz.id))

        elif event_type == 'compute_complete':
            biz.iteration_count += 1
            biz.total_service_time += biz.communication_time + biz.compute_time
            # 重新加入等待队列
            waiting_businesses.append(biz)

    # 统计输出
    avg_time_all = []
    for biz in businesses:
        avg_time = total_simulation_time / max(biz.iteration_count, 1)
        avg_time_all.append(avg_time)
    return avg_time_all


# 示例用法
if __name__ == "__main__":
    NUM_PODS = 8
    NUM_PORTS = 16
    RECON_TIME = 0.15
    LINK_CAPACITY = 10.0

    def create_data_matrix(num_pods, density=0.3):
        matrix = np.random.rand(num_pods, num_pods) * 10
        np.fill_diagonal(matrix, 0)
        matrix[np.random.rand(num_pods, num_pods) > density] = 0
        return matrix

    # 创建业务
    businesses = []
    np.random.seed(42)
    for i in range(5):
        compute_time = 0.2 + np.random.rand() * 0.6  # 200-800ms
        data_matrix = create_data_matrix(NUM_PODS, density=0.4)
        biz = Business(i, compute_time, data_matrix)
        biz.calculate_communication_time(num_ports_per_pod=NUM_PORTS, link_capacity=LINK_CAPACITY)
        businesses.append(biz)
        print(f"业务{biz.id}: 计算={biz.compute_time:.2f}s | 通信={biz.communication_time:.2f}s")

    network = OpticalNetwork(num_pods=NUM_PODS, num_ports=NUM_PORTS, recon_time=RECON_TIME)
    simulate_hrrn_with_starvation(businesses, network, total_simulation_time=100.0)