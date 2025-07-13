import numpy as np
from math import gcd
from functools import reduce
from collections import defaultdict
import heapq


class CassiniSimulator:
    def __init__(self, num_servers, link_capacity, link_matrix):
        """
        Initialize the Cassini simulator for optical interconnected data centers.

        Args:
            num_servers: Number of servers in the data center
            link_capacity: Capacity of each link in Gbps (default 50Gbps)
        """
        self.num_servers = num_servers
        self.link_capacity = link_capacity
        self.link_matrix = link_matrix

        # Create a diagonal connectivity matrix for optical interconnection
        self.connectivity = np.eye(num_servers)
        # Add some cross-connections (optical links between non-adjacent servers)
        for i in range(num_servers):
            for j in range(i + 1, num_servers, 2):  # Connect every other server
                self.connectivity[i, j] = 1
                self.connectivity[j, i] = 1

        # Data structures to track current jobs
        self.active_jobs = []

        # For tracking performance metrics
        self.iteration_times = []

    def add_job(self, job_id, iteration_time, compute_time, bandwidth_matrix):
        """
        Add a new job to the cluster with its traffic matrix.

        Args:
            job_id: Unique identifier for the job
            iteration_time: Total time for one training iteration (ms)
            compute_time: Duration of compute phase (ms)
            :param bandwidth_matrix:

        """
        comm_time = iteration_time - compute_time
        job = {
            'id': job_id,
            'iteration_time': iteration_time,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'bandwidth_matrix': bandwidth_matrix,
            'time_shift': 0  # Will be set by Cassini scheduler
        }
        self.active_jobs.append(job)

    def find_bottleneck_links(self):
        """
        Find all links (including unidirectional) that have more than one job using them.
        Returns a dict: {(src, dst): [list of jobs sharing this link]}
        """
        # First build link usage: {(src, dst): set(job_ids)}
        link_usage = defaultdict(set)

        for job in self.active_jobs:
            # Get all directed links used by this job (src -> dst)
            srcs, dsts = np.where(job['bandwidth_matrix'] > 0)
            for src, dst in zip(srcs, dsts):
                link_usage[(src, dst)].add(job['id'])

        # Only keep links shared by multiple jobs
        bottleneck_links = {link: list(jobs) for link, jobs in link_usage.items() if len(jobs) > 1}
        return bottleneck_links

    def lcm(self, numbers):
        """Compute LCM of a list of numbers"""

        def lcm_two(a, b):
            return a * b // gcd(a, b)

        return reduce(lcm_two, numbers, 1)

    def compute_compatibility(self, jobs, link):
        """
        Compute compatibility score and optimal time shifts for jobs sharing a link.

        Args:
            jobs: List of jobs sharing the link
            link: Tuple (server1, server2) identifying the link

        Returns:
            (compatibility_score, time_shifts) where time_shifts is a dict {job_id: shift}
        """
        # Create unified circle with perimeter = LCM of iteration times
        iteration_times = [job['iteration_time'] for job in jobs]
        # perimeter = self.lcm(iteration_times)
        perimeter = max(iteration_times)
        degree = int(np.ceil(perimeter))
        # print(degree)

        # For each job, create its bandwidth demand pattern on the unified circle
        job_patterns = []
        for job in jobs:
            r = job['iteration_time'] / perimeter

            # Create bandwidth demand pattern (0 during compute, bw during comm)
            pattern = np.zeros(degree)  # 1-degree precision (360 points)

            # 计算该作业在LCM周期内完整的迭代次数
            num_repeats = int(np.ceil(perimeter / job['iteration_time']))
            # num_repeats = 1

            # 计算通信阶段在统一圆环上的总角度跨度
            # total_comm_angle = int(degree * job['comm_time'] / perimeter)

            # 每次迭代在圆环上的角度跨度
            iter_angle = int(degree * job['iteration_time'] / perimeter)

            # 每次迭代的通信角度跨度（保持通信/计算比例）
            comm_angle_per_iter = int(degree * job['comm_time'] / perimeter)
            # pattern[0:total_comm_angle] = job['bandwidth_matrix'][link]

            for rep in range(num_repeats):
                iter_start = rep * iter_angle
                comm_start = int(iter_start)
                comm_end = int(iter_start + comm_angle_per_iter)
                pattern[comm_start:comm_end] = job['bandwidth_matrix'][link]

                # # 处理圆环边界
                # if comm_end > degree:
                #     pattern[comm_start:degree] = job['bandwidth_matrix'][link]
                #     pattern[0:(comm_end % degree)] = job['bandwidth_matrix'][link]
                # else:
                #     pattern[comm_start:comm_end] = job['bandwidth_matrix'][link]

            job_patterns.append((job, pattern))

        # Try all possible rotations to find best compatibility
        best_score = -np.inf
        best_shifts = {}

        # We'll use a greedy approach to find good shifts (not exhaustive for performance)
        for base_job, base_pattern in job_patterns:
            # Try aligning other jobs to this base job
            shifts = {base_job['id']: 0}
            total_demand = np.copy(base_pattern)

            for other_job, other_pattern in job_patterns:
                if other_job['id'] == base_job['id']:
                    continue

                # Find shift that minimizes overlap
                min_overlap = np.inf
                best_shift = 0

                # 只需遍历当前作业的迭代角度范围
                for shift in range(int(degree * other_job['iteration_time'] / perimeter)):
                    shifted_pattern = np.roll(other_pattern, shift)
                    overlap = np.sum(np.maximum(total_demand + shifted_pattern - self.link_capacity, 0))

                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_shift = shift

                # 转换最优偏移为时间
                time_shift = (best_shift / degree) * perimeter

                # Convert angle shift to time shift
                # time_shift = (best_shift / 360) * perimeter % other_job['iteration_time']
                shifts[other_job['id']] = time_shift

                # Update total demand
                shifted_pattern = np.roll(other_pattern, best_shift)
                total_demand += shifted_pattern

            # Calculate compatibility score
            excess = np.maximum(total_demand - self.link_capacity, 0)
            score = 1 - np.mean(excess) / self.link_capacity

            if score > best_score:
                best_score = score
                best_shifts = shifts

        return best_score, best_shifts

    def build_affinity_graph(self):
        """Build the affinity graph and compute time shifts for all jobs."""
        # Find all bottleneck links and their shared jobs
        bottleneck_links = self.find_bottleneck_links()

        # Build affinity graph: {job: {link: time_shift}}
        affinity_graph = defaultdict(dict)

        # First compute time shifts for each link independently
        for link, job_ids in bottleneck_links.items():
            job_objects = [j for j in self.active_jobs if j['id'] in job_ids]
            score, shifts = self.compute_compatibility(job_objects, link)

            for job_id, shift in shifts.items():
                affinity_graph[job_id][link] = shift

        # Now traverse the graph to assign unique time shifts
        # We'll use a simple BFS approach as described in the paper
        if not affinity_graph:
            return {}

        # Select a random job as root
        root_job = next(iter(affinity_graph.keys()))
        time_shifts = {root_job: 0}
        queue = [root_job]
        visited = set([root_job])

        while queue:
            current_job = queue.pop(0)

            # Get all links connected to this job
            links = affinity_graph[current_job].keys()

            for link in links:
                # Get all jobs sharing this link
                sharing_jobs = [j['id'] for j in self.active_jobs if
                                j['id'] in affinity_graph and link in affinity_graph[j['id']]]

                for neighbor_job in sharing_jobs:
                    if neighbor_job == current_job:
                        continue

                    if neighbor_job not in visited:
                        # Compute time shift relative to current job
                        t_current = time_shifts[current_job]
                        t_link_current = affinity_graph[current_job][link]
                        t_link_neighbor = affinity_graph[neighbor_job][link]

                        # Find the neighbor job object to get its iteration time
                        neighbor_obj = next(j for j in self.active_jobs if j['id'] == neighbor_job)

                        # The unique time shift formula from the paper
                        t_neighbor = (t_current - t_link_current + t_link_neighbor) % neighbor_obj['iteration_time']

                        time_shifts[neighbor_job] = t_neighbor
                        visited.add(neighbor_job)
                        queue.append(neighbor_job)

        return time_shifts

    def simulate_iteration(self):
        """基于全局时间线的精确冲突检测与惩罚计算"""
        # 1. 应用Cassini时间偏移
        time_shifts = self.build_affinity_graph()
        for job in self.active_jobs:
            job['time_shift'] = time_shifts.get(job['id'], 0)

        # 2. 生成全局事件时间线（所有作业的所有通信时段）
        events = []
        for job in self.active_jobs:
            # 获取该作业在所有链路上的通信窗口
            srcs, dsts = np.where(job['bandwidth_matrix'] > 0)
            for src, dst in zip(srcs, dsts):
                link = (src, dst)
                bw = job['bandwidth_matrix'][src, dst]
                windows = self.get_periodic_windows(
                    job['time_shift'],
                    job['comm_time'],
                    job['iteration_time'],
                    max([j['iteration_time'] for j in self.active_jobs])
                )
                for start, end in windows:
                    events.append((start, 'start', job['id'], link, bw))
                    events.append((end, 'end', job['id'], link, bw))

        # 3. 按时间排序所有事件
        events.sort(key=lambda x: x[0])

        active_links = defaultdict(float)  # {link: 当前总带宽}
        active_jobs = []
        total_penalty = 0.0
        prev_time = 0.0
        iter_num = np.zeros(len(self.active_jobs))
        job_penalty = np.zeros(len(self.active_jobs))

        # 5. 处理每个时间区间
        for time, typ, job_id, link, bw in events:
            # 计算上一时间段的惩罚
            if prev_time < time:
                # 计算所有链路的过载量
                link_overloads = {
                    link: max(0, (load - self.link_capacity * self.link_matrix[link]) / (self.link_capacity * self.link_matrix[link]))
                    for link, load in active_links.items()
                }
                # 取最大过载作为该时段惩罚
                current_penalty = max(link_overloads.values(), default=0.0)
                # 乘以持续时间
                total_penalty += current_penalty * (time - prev_time)
                for job_con in active_jobs:
                    (bw_id, idx) = job_con
                    job_penalty[idx] += current_penalty * (time - prev_time) / len(active_jobs)

            # 更新链路状态
            if typ == 'start':
                active_links[link] += bw
                active_jobs.append((bw, job_id))
            else:
                active_links[link] -= bw
                active_links[link] = max(0.0, active_links[link])  # 防止负值
                active_jobs = [ele for ele in active_jobs if ele[1] != job_id]

            prev_time = time

        # 6. 返回总惩罚（单位：Gbps·ms）
        max_time = (max([j['iteration_time'] for j in self.active_jobs]) + total_penalty) / 1000
        job_time = []
        for job in self.active_jobs:
            job_time.append((max([j['iteration_time'] for j in self.active_jobs]) / int((max([j['iteration_time'] for j in self.active_jobs]) / job['iteration_time'])) + job_penalty[job['id']]) / 1000)
        return max_time, job_time
        # 4. 全局冲突检测
        # active_communications = defaultdict(dict)  # {link: {job_id: bw}}
        # current_penalties = defaultdict(float)  # {job_id: penalty}
        # prev_time = 0
        #
        # for time, typ, job_id, link, bw in events:
        #     # 处理上一个时段的冲突
        #     time_elapsed = time - prev_time
        #     if time_elapsed > 0:
        #         print(1)
        #         self._calculate_instant_penalties(
        #             active_communications,
        #             time_elapsed,
        #             current_penalties
        #         )
        #
        #     # 更新当前活动通信
        #     if typ == 'start':
        #         active_communications[link][job_id] = bw
        #     else:
        #         active_communications[link].pop(job_id, None)
        #
        #     prev_time = time
        #
        # # 5. 应用惩罚（限制最大惩罚）
        # iteration_times = []
        # for job in self.active_jobs:
        #     penalty = min(
        #         current_penalties.get(job['id'], 0),
        #         job['comm_time'] * 2  # 最多2倍通信时间
        #     )
        #     iteration_times.append(job['iteration_time'] + penalty)
        #
        # self.iteration_times.append(iteration_times)
        # return np.mean(iteration_times)

    def _calculate_instant_penalties(self, active_comms, duration, penalties):
        """计算当前瞬时各链路的冲突惩罚"""
        job_link_penalties = defaultdict(list)

        # 第一步：计算每个作业在各链路的瞬时惩罚
        for link, jobs in active_comms.items():
            total_bw = sum(jobs.values())
            overload = max(0, total_bw - self.link_capacity * self.link_matrix[link])

            if overload > 0:
                # 分配当前链路的惩罚给各作业
                for job_id, bw in jobs.items():
                    penalty = duration * (bw / total_bw) * (overload / (self.link_capacity * self.link_matrix[link]))
                    job_link_penalties[job_id].append(penalty)

        # 第二步：对每个作业取最大链路惩罚
        for job_id, link_penalties in job_link_penalties.items():
            if link_penalties:
                penalties[job_id] += max(link_penalties)

    def get_periodic_windows(self, start, duration, period, lcm_period):
        """生成周期性通信窗口（支持跨周期边界）"""
        windows = []
        num_repeats = int(lcm_period / period)
        # num_repeats = 1
        for k in range(num_repeats):
            window_start = (start + k * period) % lcm_period
            window_end = (window_start + duration) % lcm_period
            if window_end < window_start:
                windows.append((window_start, lcm_period))
                if window_end != 0:
                    windows.append((0, window_end))
            else:
                windows.append((window_start, window_end))
        return windows

    """
    def calculate_windows_overlap(self, windows1, windows2):
        # Calculate total overlap time between two sets of windows.
        overlap = 0

        for (s1, e1) in windows1:
            for (s2, e2) in windows2:
                # Calculate overlap between two individual windows
                overlap += max(0, min(e1, e2) - max(s1, s2))

        return overlap

    def simulate_iteration(self):
        # Simulate one training iteration with current placements and time shifts.
        # Get time shifts from affinity graph
        time_shifts = self.build_affinity_graph()

        # Apply time shifts to jobs
        for job in self.active_jobs:
            job['time_shift'] = time_shifts.get(job['id'], 0)

        # Simulate the iteration for each job
        iteration_times = []

        for job in self.active_jobs:
            # Baseline iteration time (without network congestion)
            base_time = job['iteration_time']

            # Find all bottleneck links this job uses
            bottleneck_links = self.find_bottleneck_links()
            job_links = []

            for link, job_ids in bottleneck_links.items():
                if job['id'] in job_ids:
                    job_links.append(link)

            # Calculate congestion penalty
            congestion_penalty = 0

            for link in job_links:
                # Get all jobs sharing this link
                sharing_jobs = [j for j in self.active_jobs if j['id'] in bottleneck_links[link]]

                # Check if communication phases overlap
                overlaps = 0
                for other_job in sharing_jobs:
                    if other_job['id'] == job['id']:
                        continue

                    # Check if communication phases overlap considering time shifts
                    job1_start = job['time_shift']
                    job1_end = job1_start + job['comm_time']

                    job2_start = other_job['time_shift']
                    job2_end = job2_start + other_job['comm_time']

                    # Check for overlap (simplified)
                    if not (job1_end <= job2_start or job2_end <= job1_start):
                        overlaps += 1

                if overlaps > 0:
                    # Get the actual bandwidth demand for this link
                    i, j = link
                    bw_demand = job['traffic_matrix'][i, j]
                    # Penalty proportional to demand and overlaps
                    congestion_penalty += (bw_demand / self.link_capacity) * job['comm_time'] * overlaps

            # Final iteration time is base time plus congestion penalty
            iteration_time = base_time + congestion_penalty
            iteration_times.append(iteration_time)

        # Record metrics
        self.iteration_times.append(iteration_times)

        return np.mean(iteration_times)
    """
    def run_simulation(self):
        """Run the simulation for multiple iterations."""
        avg_time, all_time = self.simulate_iteration()
        return avg_time, all_time


# Example usage
if __name__ == "__main__":
    # Create a simulator with 24 servers (like in the paper)
    simulator = CassiniSimulator(num_servers=4, link_capacity=10)

    # Example traffic matrices (in reality these would come from your external source)
    # For VGG16 job: using servers 0-3 with ring allreduce pattern
    vgg16_band = np.zeros((4, 4))
    vgg16_band[0, 1] = vgg16_band[1, 0] = 20  # 20Gbps between server 0-1
    vgg16_band[1, 2] = vgg16_band[2, 1] = 20  # 20Gbps between server 1-2
    vgg16_band[2, 3] = vgg16_band[3, 2] = 20  # 20Gbps between server 2-3
    vgg16_band[3, 0] = vgg16_band[0, 3] = 20  # 20Gbps between server 3-0

    # For ResNet50 job: using servers 4-7 with all-to-all pattern
    resnet_band = np.zeros((4, 4))
    for i in range(0, 3):
        for j in range(i + 1, 4):
            resnet_band[i, j] = resnet_band[j, i] = 45  # 15Gbps between all pairs

    # Add jobs with their traffic matrices
    simulator.add_job("VGG16", 200, 50, vgg16_band)
    simulator.add_job("ResNet50", 200, 100, resnet_band)

    # Run simulation for 10 iterations
    avg_times = simulator.run_simulation()

    print(avg_times)
