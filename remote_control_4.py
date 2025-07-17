# remote_control.py
import paramiko
import time

# 服务器配置
servers = [
    {
        'host': '192.168.104.242',
        'ipv6': '2001:da8:d800:338:ae1f:6bff:fe81:62c6',
        'username': 'tbj',
        'password': 'TBJtbj20041204'
    },
    {
        'host': '192.168.104.243',
        'ipv6': '2001:da8:d800:338:ae1f:6bff:fe81:666e',
        'username': 'tbj',
        'password': 'TBJtbj20041204'
    }
]

# RDMA RoCEv2 配置
RDMA_DEVICE = 'ens10f1np1'

def start_training():
    processes = []
    
    for i, server in enumerate(servers):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server['host'], username=server['username'], password=server['password'])
        
        # 启动训练命令
        master_addr = servers[0]['ipv6']  # 使用第一个服务器的IPv6作为master地址
        cmd = f"""
        cd /home/tbj/dml_code && 
        NCCL_IB_HCA=mlx5_1 
        NCCL_SOCKET_IFNAME={RDMA_DEVICE} 
        NCCL_IB_GID_INDEX=3 
        NCCL_DEBUG=INFO 
        python dml4.py 
        --world-size 2 
        --rank {i} 
        --master-addr {master_addr} 
        --master-port 29500 
        --batch-size 256 
        --data-path /home/tbj/data/cifar-10-batches-py/
        """
        
        # 使用nohup在后台运行
        full_cmd = f"nohup {cmd} > /home/tbj/dml_code/train_{i}.log 2>&1 &"
        stdin, stdout, stderr = ssh.exec_command(full_cmd)
        print(f"Started training on {server['host']}")
        
        processes.append(ssh)
    
    print("Training started on both servers. Waiting for completion...")
    
    # 保持连接打开 (在实际使用中可能需要更复杂的监控)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping training...")
        for ssh in processes:
            ssh.close()

if __name__ == "__main__":
    start_training()
