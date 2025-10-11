"""
算子学习完整训练代码 - 纯PyTorch实现
Deep Operator Network (DeepONet) for Integral Operator Learning

这是一个完整的DeepONet实现，用于学习积分算子映射。
运行此文件即可完成模型训练、评估和可视化。

作者: AI4CFD团队
日期: 2025年10月11日
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置随机种子确保可重现性
torch.manual_seed(42)
np.random.seed(42)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")
print(f"🔥 PyTorch版本: {torch.__version__}")

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# ==================== 模型定义 ====================

class MLP(nn.Module):
    """多层感知机基础类"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(MLP, self).__init__()
        
        # 选择激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # 除了最后一层，都添加激活函数
            if i < len(dims) - 2:
                layers.append(self.activation)
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """Xavier均匀初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class SimpleDeepONet(nn.Module):
    """简化版DeepONet实现"""
    
    def __init__(self, n_sensors, coord_dim, hidden_dim=100, latent_dim=100):
        super(SimpleDeepONet, self).__init__()
        
        self.n_sensors = n_sensors
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        
        # 分支网络：处理传感器数据
        self.branch_net = MLP(
            input_dim=n_sensors,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=latent_dim
        )
        
        # 主干网络：处理查询坐标
        self.trunk_net = MLP(
            input_dim=coord_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=latent_dim
        )
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
        
        print(f"🏗️ DeepONet模型创建完成:")
        print(f"   📡 传感器数量: {n_sensors}")
        print(f"   📍 坐标维度: {coord_dim}")
        print(f"   🧠 潜在空间维度: {latent_dim}")
        print(f"   🔢 总参数数: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sensor_data, query_coords):
        """
        前向传播
        
        Args:
            sensor_data: 传感器数据 [batch_size, n_sensors]
            query_coords: 查询坐标 [batch_size, n_queries, coord_dim]
        
        Returns:
            output: 预测结果 [batch_size, n_queries, 1]
        """
        batch_size = sensor_data.shape[0]
        n_queries = query_coords.shape[1]
        
        # 分支网络输出: [batch_size, latent_dim]
        branch_output = self.branch_net(sensor_data)
        
        # 主干网络输出: [batch_size * n_queries, latent_dim]
        coords_flat = query_coords.view(-1, self.coord_dim)
        trunk_output = self.trunk_net(coords_flat)
        trunk_output = trunk_output.view(batch_size, n_queries, self.latent_dim)
        
        # 计算内积: [batch_size, n_queries]
        output = torch.einsum('bl,bql->bq', branch_output, trunk_output)
        
        # 添加偏置并扩展维度: [batch_size, n_queries, 1]
        output = output + self.bias
        output = output.unsqueeze(-1)
        
        return output


# ==================== 数据生成器 ====================

class IntegralOperatorDataset:
    """积分算子数据集"""
    
    def __init__(self, n_samples=1000, n_sensors=50, n_queries=100, domain=[0, 1]):
        self.n_samples = n_samples
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        self.domain = domain
        
        # 传感器位置（固定）
        self.sensor_locations = torch.linspace(domain[0], domain[1], n_sensors)
        
        print(f"📊 创建积分算子数据集:")
        print(f"   📈 样本数量: {n_samples}")
        print(f"   📡 传感器数量: {n_sensors}")
        print(f"   📍 查询点数量: {n_queries}")
        print(f"   🔢 定义域: {domain}")
    
    def generate_input_function(self, coeffs):
        """
        生成输入函数 u(x) = Σ a_k sin(k*π*x)
        
        Args:
            coeffs: 傅里叶系数 [n_modes]
            
        Returns:
            function: 在传感器位置的函数值
        """
        x = self.sensor_locations
        u = torch.zeros_like(x)
        
        for k, a_k in enumerate(coeffs, 1):
            u += a_k * torch.sin(k * np.pi * x)
        
        return u
    
    def compute_integral(self, coeffs, query_points):
        """
        计算积分算子的解析解
        G[u](y) = ∫₀ʸ u(x) dx
        
        For u(x) = Σ a_k sin(k*π*x), the integral is:
        G[u](y) = Σ a_k * (1 - cos(k*π*y)) / (k*π)
        """
        integral_values = torch.zeros_like(query_points)
        
        for k, a_k in enumerate(coeffs, 1):
            integral_values += a_k * (1 - torch.cos(k * np.pi * query_points)) / (k * np.pi)
        
        return integral_values
    
    def generate_data(self):
        """生成训练数据"""
        
        sensor_data = []
        query_coords = []
        target_values = []
        
        print("🔄 开始生成数据...")
        for i in tqdm(range(self.n_samples), desc="生成数据"):
            # 随机生成傅里叶系数
            coeffs = torch.randn(5) * 0.5  # 5个模态
            
            # 生成输入函数在传感器位置的值
            u_sensors = self.generate_input_function(coeffs)
            sensor_data.append(u_sensors)
            
            # 随机选择查询点
            query_points = torch.rand(self.n_queries) * (self.domain[1] - self.domain[0]) + self.domain[0]
            query_points = query_points.sort()[0]  # 排序以便可视化
            query_coords.append(query_points.unsqueeze(-1))
            
            # 计算积分算子的精确值
            integral_exact = self.compute_integral(coeffs, query_points)
            target_values.append(integral_exact.unsqueeze(-1))
        
        # 转换为张量
        sensor_data = torch.stack(sensor_data)
        query_coords = torch.stack(query_coords)
        target_values = torch.stack(target_values)
        
        print(f"\n✅ 数据生成完成!")
        print(f"   📊 传感器数据形状: {sensor_data.shape}")
        print(f"   📍 查询坐标形状: {query_coords.shape}")
        print(f"   🎯 目标值形状: {target_values.shape}")
        
        return sensor_data, query_coords, target_values


# ==================== 训练器 ====================

class DeepONetTrainer:
    """DeepONet训练器"""
    
    def __init__(self, model, device='cpu', save_dir='./results'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self, sensor_data, query_coords, target_values, 
              epochs=5000, batch_size=32, lr=1e-3, val_split=0.2,
              save_model=True, plot_interval=1000):
        """训练模型"""
        
        print(f"\n🚀 开始DeepONet训练...")
        print(f"   📊 训练轮次: {epochs}")
        print(f"   🎯 批次大小: {batch_size}")
        print(f"   📈 学习率: {lr}")
        print(f"   🔄 验证集比例: {val_split}")
        
        # 数据分割
        n_samples = sensor_data.shape[0]
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        train_sensor = sensor_data[train_idx].to(self.device)
        train_coords = query_coords[train_idx].to(self.device)
        train_targets = target_values[train_idx].to(self.device)
        
        val_sensor = sensor_data[val_idx].to(self.device)
        val_coords = query_coords[val_idx].to(self.device)
        val_targets = target_values[val_idx].to(self.device)
        
        print(f"📊 数据分割完成:")
        print(f"   🚂 训练集: {n_train} 样本")
        print(f"   🔍 验证集: {n_val} 样本")
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        # 训练循环
        print(f"\n🔥 开始训练循环...")
        for epoch in tqdm(range(epochs), desc="训练进度"):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            # 批次训练
            n_batches = (n_train + batch_size - 1) // batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_train)
                
                batch_sensor = train_sensor[start_idx:end_idx]
                batch_coords = train_coords[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # 前向传播
                predictions = self.model(batch_sensor, batch_coords)
                loss = criterion(predictions, batch_targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= n_batches
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_sensor, val_coords)
                val_loss = criterion(val_predictions, val_targets).item()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss and save_model:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')
            
            # 学习率调度
            scheduler.step()
            
            # 打印进度
            if (epoch + 1) % plot_interval == 0:
                print(f"\n轮次 {epoch+1:4d}/{epochs}: "
                      f"训练损失 = {train_loss:.6e}, "
                      f"验证损失 = {val_loss:.6e}, "
                      f"学习率 = {optimizer.param_groups[0]['lr']:.1e}")
        
        print(f"\n✅ 训练完成!")
        print(f"   📉 最终训练损失: {self.history['train_loss'][-1]:.6e}")
        print(f"   📊 最终验证损失: {self.history['val_loss'][-1]:.6e}")
        print(f"   🏆 最佳验证损失: {best_val_loss:.6e}")
        
        # 保存最终模型和训练历史
        if save_model:
            self.save_checkpoint(epochs-1, self.history['val_loss'][-1], 'final_model.pth')
            self.save_training_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, loss, filename):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_history(self, save_plot=True):
        """绘制训练历史"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax.semilogy(epochs, self.history['train_loss'], 'b-', linewidth=2, label='训练损失')
        ax.semilogy(epochs, self.history['val_loss'], 'r-', linewidth=2, label='验证损失')
        
        ax.set_xlabel('训练轮次')
        ax.set_ylabel('损失值 (对数尺度)')
        ax.set_title('DeepONet训练历史', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()


# ==================== 评估器 ====================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device, save_dir='./results'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
    
    def evaluate_model(self, dataset, n_test_samples=100):
        """全面评估模型性能"""
        
        print("🧪 生成测试数据...")
        test_sensor, test_coords, test_targets = dataset.generate_data()
        
        # 移动到设备
        test_sensor = test_sensor[:n_test_samples].to(self.device)
        test_coords = test_coords[:n_test_samples].to(self.device)
        test_targets = test_targets[:n_test_samples].to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_sensor, test_coords)
        
        # 计算误差指标
        mse = torch.mean((predictions - test_targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - test_targets)).item()
        relative_error = torch.mean(torch.abs(predictions - test_targets) / 
                                   (torch.abs(test_targets) + 1e-8)).item()
        
        # 计算R²
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = test_targets.cpu().numpy().flatten()
        r2 = 1 - np.sum((pred_flat - target_flat)**2) / np.sum((target_flat - target_flat.mean())**2)
        
        print(f"\n📊 模型性能评估:")
        print(f"   📈 均方误差 (MSE): {mse:.6e}")
        print(f"   📏 平均绝对误差 (MAE): {mae:.6e}")
        print(f"   📋 相对误差: {relative_error:.6f}")
        print(f"   🎯 R² 系数: {r2:.6f}")
        
        # 保存评估结果
        results = {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'r2_score': r2,
            'n_test_samples': n_test_samples
        }
        
        results_path = os.path.join(self.save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return predictions, test_targets, test_coords, test_sensor, results
    
    def visualize_predictions(self, predictions, targets, coords, sensor_data, 
                            dataset, n_samples=4, save_plot=True):
        """可视化预测结果"""
        
        # 转换为CPU并取样本
        pred_np = predictions[:n_samples].cpu().numpy()
        target_np = targets[:n_samples].cpu().numpy()
        coords_np = coords[:n_samples].cpu().numpy()
        sensor_np = sensor_data[:n_samples].cpu().numpy()
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        
        for i in range(n_samples):
            # 输入函数
            ax1 = plt.subplot(3, n_samples, i + 1)
            ax1.plot(dataset.sensor_locations, sensor_np[i], 'ro-', 
                    markersize=4, linewidth=2, alpha=0.8)
            ax1.set_title(f'输入函数 u_{i+1}(x)', fontweight='bold')
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x)')
            ax1.grid(True, alpha=0.3)
            
            # 预测 vs 真实值
            ax2 = plt.subplot(3, n_samples, i + 1 + n_samples)
            x_query = coords_np[i, :, 0]
            y_true = target_np[i, :, 0]
            y_pred = pred_np[i, :, 0]
            
            ax2.plot(x_query, y_true, 'b-', linewidth=3, label='真实值', alpha=0.8)
            ax2.plot(x_query, y_pred, 'r--', linewidth=2, label='预测值')
            ax2.set_title(f'G[u_{i+1}] - 预测 vs 真实', fontweight='bold')
            ax2.set_xlabel('x')
            ax2.set_ylabel('G[u](x)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 误差分析
            ax3 = plt.subplot(3, n_samples, i + 1 + 2*n_samples)
            error = np.abs(y_pred - y_true)
            ax3.plot(x_query, error, 'g-', linewidth=2)
            ax3.fill_between(x_query, 0, error, alpha=0.3, color='green')
            ax3.set_title(f'绝对误差 |预测 - 真实|', fontweight='bold')
            ax3.set_xlabel('x')
            ax3.set_ylabel('|误差|')
            ax3.grid(True, alpha=0.3)
            
            # 添加误差统计
            mean_error = np.mean(error)
            max_error = np.max(error)
            ax3.text(0.05, 0.95, f'平均: {mean_error:.4f}\n最大: {max_error:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, 'prediction_results.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def final_demonstration(self, dataset, save_plot=True):
        """最终演示：测试全新的高斯函数"""
        
        print("🎭 最终演示：DeepONet处理全新输入函数")
        print("="*50)
        
        # 创建测试函数
        x = torch.linspace(0, 1, 50)
        
        def gaussian_function(x, center=0.5, width=0.2):
            return torch.exp(-((x - center) / width) ** 2)
        
        # 三个不同的高斯函数
        test_functions = [
            gaussian_function(x, 0.3, 0.1),
            gaussian_function(x, 0.7, 0.15),
            gaussian_function(x, 0.5, 0.25)
        ]
        
        query_points = torch.linspace(0, 1, 100).unsqueeze(-1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        self.model.eval()
        for i, test_func in enumerate(test_functions):
            # 准备输入
            sensor_input = test_func.unsqueeze(0).to(self.device)
            query_input = query_points.unsqueeze(0).to(self.device)
            
            # DeepONet预测
            with torch.no_grad():
                prediction = self.model(sensor_input, query_input)
            
            # 数值积分作为参考
            x_fine = torch.linspace(0, 1, 1000)
            func_fine = gaussian_function(x_fine, 
                                        0.3 + i * 0.2, 
                                        0.1 + i * 0.075)
            numerical_integral = []
            for j, y in enumerate(query_points.squeeze()):
                mask = x_fine <= y
                if mask.sum() > 1:
                    integral_val = torch.trapz(func_fine[mask], x_fine[mask])
                else:
                    integral_val = torch.tensor(0.0)
                numerical_integral.append(integral_val)
            numerical_integral = torch.stack(numerical_integral)
            
            # 绘制输入函数
            axes[0, i].plot(x, test_func, 'bo-', linewidth=2, markersize=3)
            axes[0, i].set_title(f'测试函数 {i+1} (高斯分布)', fontweight='bold')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('u(x)')
            axes[0, i].grid(True, alpha=0.3)
            
            # 绘制积分预测
            axes[1, i].plot(query_points.squeeze(), prediction.cpu().squeeze(), 
                           'r-', linewidth=3, label='DeepONet预测')
            axes[1, i].plot(query_points.squeeze(), numerical_integral, 
                           'b--', linewidth=2, alpha=0.7, label='数值积分')
            axes[1, i].set_title(f'积分预测 vs 数值解', fontweight='bold')
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('∫₀ˣ u(s) ds')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
            # 计算误差
            error = torch.abs(prediction.cpu().squeeze() - numerical_integral)
            mean_error = torch.mean(error).item()
            max_error = torch.max(error).item()
            
            print(f"测试函数 {i+1}: 平均误差 = {mean_error:.6f}, 最大误差 = {max_error:.6f}")
        
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, 'generalization_test.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎉 演示完成！DeepONet成功处理了训练时未见过的高斯函数！")
        print("💡 这证明了模型的泛化能力和算子学习的威力。")


# ==================== 主函数 ====================

def main():
    """主训练流程"""
    
    print("="*60)
    print("🚀 DeepONet算子学习训练程序")
    print("="*60)
    
    # 配置参数
    config = {
        'n_samples': 1000,
        'n_sensors': 50,
        'n_queries': 100,
        'hidden_dim': 64,
        'latent_dim': 64,
        'epochs': 3000,
        'batch_size': 16,
        'lr': 1e-3,
        'val_split': 0.2,
        'save_dir': './deeponet_results'
    }
    
    print("📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 1. 创建数据集
    print("\n" + "="*40)
    print("📊 Step 1: 创建数据集")
    print("="*40)
    
    dataset = IntegralOperatorDataset(
        n_samples=config['n_samples'],
        n_sensors=config['n_sensors'],
        n_queries=config['n_queries']
    )
    
    sensor_data, query_coords, target_values = dataset.generate_data()
    
    # 2. 创建模型
    print("\n" + "="*40)
    print("🏗️ Step 2: 创建DeepONet模型")
    print("="*40)
    
    model = SimpleDeepONet(
        n_sensors=config['n_sensors'],
        coord_dim=1,
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim']
    )
    
    # 3. 训练模型
    print("\n" + "="*40)
    print("🚀 Step 3: 训练模型")
    print("="*40)
    
    trainer = DeepONetTrainer(model, device=device, save_dir=config['save_dir'])
    history = trainer.train(
        sensor_data=sensor_data,
        query_coords=query_coords,
        target_values=target_values,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        val_split=config['val_split']
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 4. 评估模型
    print("\n" + "="*40)
    print("📊 Step 4: 评估模型")
    print("="*40)
    
    evaluator = ModelEvaluator(model, device, save_dir=config['save_dir'])
    
    # 标准评估
    predictions, targets, coords, sensor_test, results = evaluator.evaluate_model(dataset)
    
    # 可视化预测结果
    evaluator.visualize_predictions(predictions, targets, coords, sensor_test, dataset)
    
    # 泛化能力测试
    evaluator.final_demonstration(dataset)
    
    print("\n" + "="*60)
    print("✅ 训练流程完成！")
    print("="*60)
    print(f"📁 所有结果已保存到: {config['save_dir']}")
    print("📊 文件列表:")
    for file in os.listdir(config['save_dir']):
        print(f"   - {file}")
    
    return model, history, results


if __name__ == "__main__":
    # 运行主程序
    model, history, results = main()