开始工作：
1. 创建 A6000 VM
2. 和 Volume 放在同一个 environment（现在是 default-CANADA-1）
3. Attach LLM-WS-Volume
4. 在新 VM 里：
    ```
    sudo mkdir -p /workspace
    sudo mount /dev/vdb /workspace
    sudo chown -R ubuntu:ubuntu /workspace
    ```
5. cd /workspace，继续昨天的工作



结束工作：
1. 确认程序跑完
2. Git commit/push 重要代码
3. 执行：
    ```
    sync
    ```
4. 最好先：
    ```
    sudo umount /workspace
    ```
5. sudo umount /workspace
6. 删除 VM
7. 不要删除 Volume