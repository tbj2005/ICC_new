import paramiko
import time
import select
import sys

# Server configurations
servers = [
    {
        'host': '192.168.104.242',
        'port': 22,
        'username': 'tbj',
        'password': 'TBJtbj20041204',
        'env': '/home/tbj/.pyenv/versions/3.10.12/bin/python',
        'code_dir': '/home/tbj/dml_code',
        'network_interface': 'ens10f1np1',  # RDMA网卡名
        'ib_device': 'mlx5_1'  # IB设备名
    },
    {
        'host': '192.168.104.243',
        'port': 22,
        'username': 'tbj',
        'password': 'TBJtbj20041204',
        'env': 'python3.10',
        'code_dir': '/home/tbj/dml_code',
        'network_interface': 'ens10f1np1',  # RDMA网卡名
        'ib_device': 'mlx5_1'  # IB设备名
    }
]

# Training parameters
world_size = len(servers)
master_addr = servers[0]['host']
master_port = '29500'
num_epochs = 3  # GPT-2训练通常需要较少epoch
batch_size = 2   # 减小batch size以适应GPT-2内存需求
seq_len = 2   # 序列长度
learning_rate = 5e-5  # GPT-2常用学习率

def execute_remote_command(ssh, command):
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.exec_command(command)
    return channel

def monitor_channels(servers):
    print("\nTraining started on all servers. Monitoring progress...\n")
    while True:
        for server in servers:
            channel = server['channel']
            host = server['host']
            
            if channel.exit_status_ready():
                continue
            
            rl, _, _ = select.select([channel], [], [], 0.1)
            if channel in rl:
                if channel.recv_ready():
                    output = channel.recv(1024).decode('utf-8')
                    print(f"[{host}] {output}", end='')
                    sys.stdout.flush()
                if channel.recv_stderr_ready():
                    error = channel.recv_stderr(1024).decode('utf-8')
                    print(f"[{host} ERROR] {error}", end='')
                    sys.stdout.flush()
        
        # Check completion
        if all(server['channel'].exit_status_ready() for server in servers):
            print("\nAll training processes completed.")
            break
            
        time.sleep(0.1)

def main():
    ssh_connections = []
    
    try:
        # Connect to all servers
        for rank, server in enumerate(servers):
            print(f"Connecting to {server['host']}...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=server['host'],
                port=server['port'],
                username=server['username'],
                password=server['password'],
                timeout=10
            )
            ssh_connections.append(ssh)
            
            # Build the training command with RDMA optimization
            command = (
                f"export NCCL_SOCKET_IFNAME={server['network_interface']}; "  # 指定RDMA网卡
                f"export NCCL_IB_HCA={server['ib_device']}; "  # 指定IB设备
                f"export NCCL_IB_GID_INDEX=3; "       # RoCE v2通常使用GID索引3
                f"export NCCL_IB_TIMEOUT=23; "        # 增加IB超时时间
                f"export NCCL_IB_QPS_PER_CONNECTION=4; "  # 提高QP数量
                f"export NCCL_DEBUG=INFO; "           # 开启详细日志
                f"export NCCL_NET_GDR_LEVEL=2; "      # 强制使用GPUDirect RDMA
                f"export OMP_NUM_THREADS=8; "        # 优化OpenMP线程数
                f"cd {server['code_dir']} && "
                f"{server['env']} dml3.py "  # 修改为GPT-2训练脚本
                f"--world-size {world_size} "
                f"--rank {rank} "
                f"--master-addr {master_addr} "
                f"--master-port {master_port} "
                f"--batch-size {batch_size} "
                f"--epochs {num_epochs} "
                f"--seq-len {seq_len} "
                f"--lr {learning_rate} "
                f"--model-path /home/tbj/gpt "  # GPT-2模型路径
                f"--data-path /home/tbj/data/wikitext-103/wiki.train.tokens "
                f"2>&1"  # 将stderr重定向到stdout
            )
            
            print(f"\nStarting training on {server['host']} with command:")
            print("="*80)
            print(command)
            print("="*80 + "\n")
            
            # Execute command
            server['channel'] = execute_remote_command(ssh, command)
        
        # Monitor training progress
        monitor_channels(servers)
                
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Stopping training...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Clean up
        for ssh in ssh_connections:
            try:
                ssh.close()
            except:
                pass
        print("All SSH connections closed.")

if __name__ == '__main__':
    main()
    