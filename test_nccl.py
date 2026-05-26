#!/usr/bin/env python3
"""
测试 NCCL 在当前环境下的工作状态
"""
import torch
import torch.distributed as dist
import os

def test_nccl():
    print("=== NCCL 兼容性测试 ===")
    
    # 检查 P2P 状态
    if torch.cuda.device_count() >= 2:
        p2p_01 = torch.cuda.can_device_access_peer(0, 1)
        p2p_10 = torch.cuda.can_device_access_peer(1, 0)
        print(f"P2P 0->1: {p2p_01}")
        print(f"P2P 1->0: {p2p_10}")
        
        if not (p2p_01 and p2p_10):
            print("⚠️  P2P 被禁用，NCCL 可能不稳定")
    
    # 设置单进程环境
    os.environ.update({
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29501',
        'RANK': '0',
        'WORLD_SIZE': '1'
    })
    
    # 测试 NCCL 初始化
    try:
        print("\n测试 NCCL 初始化...")
        dist.init_process_group(backend='nccl', timeout=dist.default_pg_timeout)
        print("✅ NCCL 初始化成功")
        
        # 测试基本通信
        print("测试 NCCL 通信...")
        tensor = torch.randn(1000).cuda()
        dist.all_reduce(tensor)
        print("✅ NCCL 通信成功")
        
        # 测试大张量（模拟模型参数）
        print("测试大张量通信...")
        large_tensor = torch.randn(10000, 100).cuda().half()
        dist.all_reduce(large_tensor)
        print("✅ NCCL 大张量通信成功")
        
        dist.destroy_process_group()
        print("\n🎉 NCCL 完全正常，可以使用 NCCL backend！")
        return True
        
    except Exception as e:
        print(f"\n❌ NCCL 测试失败: {e}")
        print("建议继续使用 Gloo backend")
        return False

def test_gloo():
    print("\n=== Gloo 性能测试 ===")
    
    os.environ.update({
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29502',
        'RANK': '0',
        'WORLD_SIZE': '1'
    })
    
    try:
        import time
        
        dist.init_process_group(backend='gloo')
        print("✅ Gloo 初始化成功")
        
        # 性能测试
        tensor = torch.randn(10000, 100).cuda().half()
        
        start_time = time.time()
        dist.all_reduce(tensor)
        gloo_time = time.time() - start_time
        
        print(f"✅ Gloo 通信耗时: {gloo_time:.4f}s")
        
        dist.destroy_process_group()
        return gloo_time
        
    except Exception as e:
        print(f"❌ Gloo 测试失败: {e}")
        return None

if __name__ == "__main__":
    # 测试 NCCL
    nccl_works = test_nccl()
    
    # 测试 Gloo 作为对比
    gloo_time = test_gloo()
    
    print("\n=== 建议 ===")
    if nccl_works:
        print("🚀 使用 NCCL backend 获得最佳性能")
        print("修改 train_ddp.py: dist.init_process_group(backend='nccl')")
    else:
        print("🐌 继续使用 Gloo backend")
        print("联系云服务商启用 GPU P2P 权限以使用 NCCL")
        if gloo_time:
            print(f"当前 Gloo 性能基准: {gloo_time:.4f}s")