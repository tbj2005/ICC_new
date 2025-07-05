import numpy as np
import heapq
from collections import defaultdict


class Business:
    def __init__(self, id, compute_time, data_matrix):
        self.id = id
        self.compute_time = compute_time
        self.data_matrix = data_matrix
        self.communication_time = 0  # 通过calculate_communication_time计算
        self.iteration_count = 0
        self.last_comm_time = 0  # 饥饿检测用
        self.total_wait_time = 0  # 累计等待时间

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
        """单次迭代总时间（用于SJF排序）"""
        return self.compute_time + self.communication_time


class OpticalNetwork:
    def __init__(self, link_matrix, recon_time):
        self.recon_time = recon_time
        self.active_business = None


def simulate_sjf_with_starvation(
        businesses,
        network,
        total_simulation_time,
        starvation_threshold=5.0
):
    """改进后的SJF调度+饥饿补偿（非抢占式，等待当前业务完成）"""
    current_time = 0.0
    event_queue = []
    waiting_businesses = []  # 最小堆：(总迭代时间, 业务ID, 业务对象)
    starving_businesses = []  # 新增：饥饿业务专用堆

    # 初始化：预计算通信时间并加入等待队列
    for biz in businesses:
        heapq.heappush(waiting_businesses, (biz.total_iteration_time, biz.id, biz))

    while current_time < total_simulation_time and (event_queue or waiting_businesses or starving_businesses):
        # 检测饥饿业务（但不立即调度）
        current_starving = [
            biz for _, _, biz in waiting_businesses
            if current_time - biz.last_comm_time >= starvation_threshold
        ]

        # 将饥饿业务移到专用队列并重新堆化等待队列
        for biz in current_starving:
            heapq.heappush(starving_businesses, (biz.total_iteration_time, biz.id, biz))
            waiting_businesses = [(t, id, b) for t, id, b in waiting_businesses if b.id != biz.id]
            heapq.heapify(waiting_businesses)

        # 网络空闲时的调度决策
        if network.active_business is None:
            # 优先调度饥饿业务中的SJF
            if starving_businesses:
                _, _, target_biz = heapq.heappop(starving_businesses)
                is_starving = True
            elif waiting_businesses:
                _, _, target_biz = heapq.heappop(waiting_businesses)
                is_starving = False
            else:
                # 没有可调度的业务，推进时间
                if event_queue:
                    current_time = event_queue[0][0]
                else:
                    current_time += 0.001
                continue

            # 执行业务调度
            target_biz.total_wait_time += max(0, current_time - target_biz.last_comm_time)
            target_biz.last_comm_time = current_time

            # 通信事件（含重构时间）
            comm_end_time = current_time + network.recon_time + target_biz.communication_time
            heapq.heappush(event_queue, (comm_end_time, 'comm_complete', target_biz.id))
            network.active_business = target_biz

            # print(f"{'⚠️ 饥饿' if is_starving else '⏳ SJF'}调度 biz_{target_biz.id} | "
            #       f"等待{current_time:.4f}s")

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
            # 完成后的业务重新加入等待队列（而非饥饿队列）
            heapq.heappush(waiting_businesses, (biz.total_iteration_time, biz.id, biz))

    # 统计输出（保持不变）
    # print("\n业务统计：")
    avg_time_all = []
    for biz in businesses:
        avg_time_all.append(total_simulation_time / max(biz.iteration_count, 1))
    return avg_time_all
    #     print(f"业务{biz.id}: "
    #           f"迭代={biz.iteration_count}次 | "
    #           f"通信时间={biz.communication_time:.4f}s | "
    #           f"平均耗时={avg_time:.2f}s")


# 示例用法（完全复制FIFO的数据生成逻辑）
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


    # 创建业务（与FIFO示例完全相同）
    businesses = []
    np.random.seed(42)
    for i in range(5):
        compute_time = 0.2 + np.random.rand() * 0.6  # 200-800ms
        data_matrix = create_data_matrix(NUM_PODS, density=0.4)
        biz = Business(i, compute_time, data_matrix)
        biz.calculate_communication_time(num_ports_per_pod=NUM_PORTS, link_capacity=LINK_CAPACITY)
        businesses.append(biz)
        print(f"业务{biz.id}: 计算时间={biz.compute_time:.4f}s | 通信时间={biz.communication_time:.4f}s")

    network = OpticalNetwork(num_pods=NUM_PODS, num_ports=NUM_PORTS, recon_time=RECON_TIME)
    simulate_sjf_with_starvation(businesses, network, total_simulation_time=100.0)
