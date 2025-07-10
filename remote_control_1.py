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
        'network_interface': 'eno1'  # 修改为实际网卡名
    },
    {
        'host': '192.168.104.243',
        'port': 22,
        'username': 'tbj',
        'password': 'TBJtbj20041204',
        'env': 'python3.10',
        'code_dir': '/home/tbj/dml_code',
        'network_interface': 'eno1'  # 修改为实际网卡名
    }
]

# Training parameters
world_size = len(servers)
master_addr = servers[0]['host']
master_port = '29500'
num_epochs = 10
batch_size = 32

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
            
            # Build the training command with TCP enforcement
            command = (
                f"export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1; "
                f"export NCCL_SOCKET_IFNAME={server['network_interface']}; "
                f"export GLOO_SOCKET_IFNAME={server['network_interface']}; "
                f"cd {server['code_dir']} && "
                f"{server['env']} dml1.py "
                f"--world-size {world_size} "
                f"--rank {rank} "
                f"--master-addr {master_addr} "
                f"--master-port {master_port} "
                f"--batch-size {batch_size} "
                f"--epochs {num_epochs} "
                f"--backend nccl "
                f"--dist-url tcp://{master_addr}:{master_port}"
            )
            
            print(f"\nStarting training on {server['host']} with command:")
            print("="*50)
            print(command)
            print("="*50 + "\n")
            
            # Execute command
            transport = ssh.get_transport()
            channel = transport.open_session()
            channel.exec_command(command)
            server['channel'] = channel
        
        # Monitor all channels
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
                
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Stopping training...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        for ssh in ssh_connections:
            try:
                ssh.close()
            except:
                pass
        print("All SSH connections closed.")

if __name__ == '__main__':
    main()
