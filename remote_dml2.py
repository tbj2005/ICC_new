import paramiko
import os
import time
import signal
from threading import Thread

servers = [
    {"host": "192.168.104.220", "rank": 0, "user": "tbj", "key_path": r"C:\Users\Smith Tang\.ssh\id_rsa"},
    {"host": "192.168.104.221", "rank": 1, "user": "tbj", "key_path": r"C:\Users\Smith Tang\.ssh\id_rsa"}
]

def connect_ssh(host, user, key_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        private_key = paramiko.RSAKey.from_private_key_file(key_path)
        ssh.connect(host, username=user, pkey=private_key, timeout=10)
        return ssh
    except Exception as e:
        print(f"连接 {host} 失败: {str(e)}")
        return None

def launch_remote(server):
    ssh = connect_ssh(server["host"], server["user"], server["key_path"])
    if ssh:
        cmd = f"""
        export NCCL_SOCKET_IFNAME=eth0;
        export CUDA_VISIBLE_DEVICES=0;
        export RANK={server["rank"]};
        export WORLD_SIZE=2;
        export MASTER_ADDR="192.168.104.220";
        export MASTER_PORT=23456;
        python /code/dml2.py > /code/rank_{server["rank"]}.log 2>&1 &
        echo $! > /code/rank_{server["rank"]}.pid
        """
        ssh.exec_command(cmd)
        print(f"成功在 {server['host']} 启动训练 (PID已保存)")
        ssh.close()

def cleanup():
    print("\n正在清理远程进程...")
    for server in servers:
        try:
            ssh = connect_ssh(server["host"], server["user"], server["key_path"])
            if ssh:
                stdin, stdout, stderr = ssh.exec_command(
                    f"kill $(cat /code/rank_{server['rank']}.pid 2>/dev/null || echo '') && "
                    f"rm -f /code/rank_{server['rank']}.pid"
                )
                print(f"已停止 {server['host']} 的训练进程")
                ssh.close()
        except Exception as e:
            print(f"清理 {server['host']} 失败: {str(e)}")

def signal_handler(sig, frame):
    cleanup()
    exit(0)

def monitor_logs():
    try:
        while True:
            for server in servers:
                try:
                    ssh = connect_ssh(server["host"], server["user"], server["key_path"])
                    if ssh:
                        stdin, stdout, stderr = ssh.exec_command(
                            "tail -n 5 /code/rank_*.log", timeout=10)
                        print(f"\n[{time.ctime()}] {server['host']} 日志:")
                        print(stdout.read().decode().strip())
                        ssh.close()
                except Exception as e:
                    print(f"监控错误 {server['host']}: {str(e)}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动所有节点
    for server in servers:
        Thread(target=launch_remote, args=(server,)).start()
    
    # 开始监控
    print("开始监控训练日志... (Ctrl+C 停止)")
    monitor_logs()
    cleanup()  # 额外确保清理
    