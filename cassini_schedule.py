import numpy as np
from math import gcd
from functools import reduce
from collections import defaultdict
import heapq


class CassiniSimulator:
    def __init__(self, num_servers, link_capacity):
        """
        Initialize the Cassini simulator for optical interconnected data centers.

        Args:
            num_servers: Number of servers in the data center
            link_capacity: Capacity of each link in Gbps (default 50Gbps)
        """
        self.num_servers = num_servers
        self.link_capacity = link_capacity

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
        """Simulate one iteration with dynamic per-link overload calculation and fair penalty allocation."""
        # Phase 1: Apply time shifts
        time_shifts = self.build_affinity_graph()
        for job in self.active_jobs:
            job['time_shift'] = time_shifts.get(job['id'], 0)

        # Phase 2: Precompute LCM period
        # lcm_period = self.lcm([job['iteration_time'] for job in self.active_jobs])
        lcm_period = max([job['iteration_time'] for job in self.active_jobs])

        # Phase 3: Calculate penalties per job
        job_penalties = defaultdict(float)
        bottleneck_links = self.find_bottleneck_links()

        for job in self.active_jobs:
            job_penalties[job['id']] = 0
        for link, job_ids in bottleneck_links.items():
            # Only process links with actual contention
            if len(job_ids) < 2:
                continue

            # Get all jobs sharing this link with their bandwidths
            sharing_jobs = [j for j in self.active_jobs if j['id'] in job_ids]

            # Calculate dynamic penalties for this link
            link_penalties = self.calculate_link_penalties(
                jobs=sharing_jobs,
                link=link,
                lcm_period=lcm_period
            )

            # Accumulate penalties to jobs
            for job_id, penalty in link_penalties.items():
                job_penalties[job_id] = max(job_penalties[job_id], penalty)

        # Phase 4: Apply penalties and record results
        iteration_time = lcm_period
        for job in self.active_jobs:
            total_penalty = job_penalties.get(job['id'], 0)

            # Ensure penalty doesn't exceed physical limits
            # clamped_penalty = min(total_penalty, job['comm_time'] * 3)  # Max 3x base comm time
            # iteration_time = job['iteration_time'] + clamped_penalty
            # iter_num = int(lcm_period / job["iteration_time"])
            iteration_time += total_penalty

        # self.iteration_times.append(iteration_times)
        iter_num = [int(lcm_period / job['iteration_time']) for job in self.active_jobs]
        avg = [iteration_time / int(lcm_period / job['iteration_time']) for job in self.active_jobs]
        return avg, iter_num

    def calculate_link_penalties(self, jobs, link, lcm_period):
        # 生成带作业ID和带宽的事件
        events = []
        for job in jobs:
            bw = job['bandwidth_matrix'][link]
            windows = self.get_periodic_windows(
                job['time_shift'],
                job['comm_time'],
                job['iteration_time'],
                lcm_period
            )
            for start, end in windows:
                events.append((start, 'a_start', bw, job['id']))
                events.append((end, 'b_end', bw, job['id']))

        # 按时间排序事件
        events.sort()

        # 动态计算惩罚
        active_jobs = {}  # {job_id: bandwidth}
        current_demand = 0
        job_penalties = defaultdict(float)
        prev_time = 0
        for time, typ, bw, job_id in events:
            # 处理上一时段的过载分配
            if len(active_jobs) >= 2:
                overload = max(0, current_demand - self.link_capacity)
                if overload > 0:
                    duration = time - prev_time
                    total_bw = sum(active_jobs.values())
                    for jid, jbw in active_jobs.items():
                        job_penalties[jid] += (jbw / total_bw) * overload * duration / self.link_capacity

            # 更新当前活跃作业
            if typ == 'a_start':
                active_jobs[job_id] = bw
                current_demand += bw
            else:
                del active_jobs[job_id]
                current_demand -= bw

            prev_time = time

        return job_penalties

    def get_periodic_windows(self, start, duration, period, lcm_period):
        """生成周期性通信窗口（支持跨周期边界）"""
        windows = []
        windows.append((start, start + duration))
        # num_repeats = lcm_period // period
        # for k in range(num_repeats):
        #     window_start = (start + k * period) % lcm_period
        #     window_end = (window_start + duration) % lcm_period
        #     if window_end < window_start:
        #         windows.append((window_start, lcm_period))
        #         if window_end != 0:
        #             windows.append((0, window_end))
        #     else:
        #         windows.append((window_start, window_end))
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
        avg_time, iter_num = self.simulate_iteration()
        return avg_time, iter_num


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