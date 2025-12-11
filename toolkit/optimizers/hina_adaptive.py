import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Optional
import math
import array
import random
import concurrent.futures
from threading import Thread
from collections import defaultdict
import time

class MemoryMonitor:
    """動態記憶體監控器"""

    def __init__(self, target_vram_gb: float = 16):
        self.target_vram = target_vram_gb * 1024**3
        self.current_usage = 0

    def check_memory_pressure(self) -> float:
        """檢查記憶體壓力比例"""
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            pressure_ratio = current_allocated / self.target_vram
            return pressure_ratio
        return 0.0

    def suggest_optimizations(self, pressure_ratio: float) -> Dict[str, bool]:
        """基於記憶體壓力建議優化策略"""
        if pressure_ratio > 0.9:
            return {
                'reduce_buffer_pool': True,
                'increase_gc_frequency': True,
                'use_checkpoint_offload': True,
                'reduce_precision': True
            }
        elif pressure_ratio > 0.7:
            return {
                'reduce_buffer_pool': True,
                'increase_gc_frequency': False,
                'use_checkpoint_offload': False,
                'reduce_precision': False
            }
        return {
            'reduce_buffer_pool': False,
            'increase_gc_frequency': False,
            'use_checkpoint_offload': False,
            'reduce_precision': False
        }


class EnhancedBufferPool:
    """增強型緩衝區池，智能記憶體管理"""

    def __init__(self, max_total_memory_mb: int = 500):
        self._buffer_pool = {}
        self._usage_stats = defaultdict(int)
        self._max_total_memory = max_total_memory_mb * 1024 * 1024
        self._current_memory = 0

    def get_buffer_with_priority(self, shape: Tuple, dtype: torch.dtype,
                               device: torch.device, priority: str = 'normal') -> torch.Tensor:
        """基於優先級獲取緩衝區"""
        key = (shape, dtype, device)
        self._usage_stats[key] += 1

        if key in self._buffer_pool and self._buffer_pool[key]:
            return self._buffer_pool[key].pop()

        # 檢查記憶體預算
        tensor_size = torch.prod(torch.tensor(shape)).item() * self._get_dtype_size(dtype)
        if self._current_memory + tensor_size > self._max_total_memory and priority != 'critical':
            # 記憶體不足且非關鍵優先級，返回新張量（不加入池）
            return torch.empty(shape, dtype=dtype, device=device)

        return torch.empty(shape, dtype=dtype, device=device)

    def return_buffer(self, tensor: torch.Tensor, priority: str = 'normal'):
        """歸還緩衝區到池中"""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)

        if key not in self._buffer_pool:
            self._buffer_pool[key] = []

        # 基於使用頻率決定是否保留
        usage_freq = self._usage_stats.get(key, 0)
        max_buffers = max(1, min(3, usage_freq // 10))  # 動態調整緩衝區數量

        if len(self._buffer_pool[key]) < max_buffers:
            tensor.zero_()
            self._buffer_pool[key].append(tensor)
            tensor_size = torch.prod(torch.tensor(tensor.shape)).item() * self._get_dtype_size(tensor.dtype)
            self._current_memory += tensor_size

    def smart_cleanup(self, memory_pressure: float):
        """智能清理緩衝區"""
        if memory_pressure > 0.8:
            # 清理使用頻率低的緩衝區
            keys_to_clean = sorted(self._usage_stats.keys(), key=lambda k: self._usage_stats[k])[:len(self._usage_stats)//2]
            for key in keys_to_clean:
                if key in self._buffer_pool:
                    del self._buffer_pool[key]
            self._current_memory = 0  # 重置計數

    @staticmethod
    def _get_dtype_size(dtype: torch.dtype) -> int:
        """獲取數據類型大小"""
        size_map = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int32: 4, torch.int16: 2, torch.int8: 1, torch.bool: 1
        }
        return size_map.get(dtype, 4)


class CompactStateDict:
    """緊湊化的狀態存儲"""
    __slots__ = ['tensor_states', 'scalar_states', 'bool_states', 'string_states']

    def __init__(self):
        self.tensor_states = {}
        self.scalar_states = {}
        self.bool_states = {}
        self.string_states = {}

    def set_tensor(self, key: str, value: torch.Tensor, use_half_precision: bool = False):
        """設置張量狀態，可選擇使用半精度"""
        if use_half_precision and value.dtype == torch.float32:
            value = value.to(torch.bfloat16)
        self.tensor_states[key] = value

    def get_tensor(self, key: str, target_dtype: torch.dtype = None, target_device: torch.device = None) -> torch.Tensor:
        """獲取張量狀態，可轉換到目標精度和設備"""
        tensor = self.tensor_states.get(key)
        if tensor is not None:
            if target_dtype is not None and tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            if target_device is not None and tensor.device != target_device:
                tensor = tensor.to(target_device)
        return tensor

    def set_scalar(self, key: str, value: float):
        """設置標量狀態"""
        self.scalar_states[key] = value

    def get_scalar(self, key: str, default: float = 0.0) -> float:
        """獲取標量狀態"""
        return self.scalar_states.get(key, default)


class CompressedRelationships:
    """壓縮的參數關係存儲"""

    def __init__(self):
        self.param_pairs = []
        self.compatibility_scores = torch.tensor([], dtype=torch.float16)
        self.interaction_types = []
        self._type_pool = {
            'matmul_12': 0, 'matmul_21': 1, 'matmul_12t': 2,
            'matmul_1t2': 3, 'norm_based': 4
        }
        self._reverse_type_pool = {v: k for k, v in self._type_pool.items()}

    def add_relationship(self, param1_id: int, param2_id: int,
                        compatibility: float, interaction_type: str):
        """添加參數關係"""
        self.param_pairs.append((param1_id, param2_id))

        # 擴展相容性分數張量
        new_score = torch.tensor([compatibility], dtype=torch.float16)
        self.compatibility_scores = torch.cat([self.compatibility_scores, new_score])

        # 使用類型池化
        type_id = self._type_pool.get(interaction_type, 4)
        self.interaction_types.append(type_id)

    def get_relationship(self, param1_id: int) -> Optional[Dict]:
        """獲取參數關係"""
        for i, (p1_id, p2_id) in enumerate(self.param_pairs):
            if p1_id == param1_id:
                return {
                    'partner_id': p2_id,
                    'compatibility': self.compatibility_scores[i].item(),
                    'interaction_type': self._reverse_type_pool[self.interaction_types[i]]
                }
        return None


@torch.jit.script
def quantize_importance_score(score: float) -> int:
    """將重要性分數量化為 int16"""
    return int(torch.clamp(torch.round(torch.tensor(score * 6553.5)), 0, 65535).item())


@torch.jit.script
def dequantize_importance_score(quantized: int) -> float:
    """將量化的重要性分數還原"""
    return float(quantized) / 6553.5


@torch.jit.script
def compute_lr_mask_update_core(lr_mask: torch.Tensor, sign_agree: torch.Tensor,
                              lr_bump: float, min_lr: float, max_lr: float) -> torch.Tensor:
    """JIT 編譯的 lr_mask 更新核心邏輯"""
    lr_adjustment = torch.where(sign_agree > 0, lr_bump, -lr_bump)
    new_lr_mask = lr_mask + lr_adjustment
    return torch.clamp(new_lr_mask, min=min_lr, max=max_lr)


@torch.jit.script
def orthogonal_gradient_core_optimized(grad_flat: torch.Tensor, param_flat: torch.Tensor,
                                     eps: float) -> torch.Tensor:
    """優化的正交梯度投影核心計算"""
    grad_norm = torch.norm(grad_flat, p=2)
    if grad_norm <= eps:
        return grad_flat

    dot_product = torch.dot(param_flat, grad_flat)
    param_norm_sq = torch.dot(param_flat, param_flat) + eps
    proj_coeff = dot_product / param_norm_sq

    orthogonal_grad_flat = grad_flat - proj_coeff * param_flat
    orth_norm = torch.norm(orthogonal_grad_flat, p=2) + eps
    scale_factor = grad_norm / orth_norm

    return orthogonal_grad_flat * scale_factor


class AsyncComputeManager:
    """異步計算管理器"""

    def __init__(self, max_workers: int = 2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []

    def submit_async_task(self, func, *args, **kwargs):
        """提交異步任務"""
        future = self.executor.submit(func, *args, **kwargs)
        self.pending_futures.append(future)
        return future

    def collect_completed_tasks(self, timeout: float = 0.001):
        """收集已完成的任務"""
        completed = []
        remaining = []

        for future in self.pending_futures:
            if future.done():
                try:
                    result = future.result(timeout=timeout)
                    completed.append(result)
                except concurrent.futures.TimeoutError:
                    remaining.append(future)
                except Exception as e:
                    print(f"[ERROR] 異步任務執行失敗: {e}")
                    remaining.append(future)
            else:
                remaining.append(future)

        self.pending_futures = remaining
        return completed

    def shutdown(self):
        """關閉執行器"""
        self.executor.shutdown(wait=True)


class Hina_Adaptive(torch.optim.Optimizer):
    """
    記憶體優化版本的自適應 Hina_Adaptive 優化器

    主要優化特性：
    1. 精度分級：關鍵狀態保持高精度，次要狀態使用低精度
    2. 智能緩衝區池：動態管理記憶體使用
    3. 壓縮狀態存儲：減少 Python 對象開銷
    4. 異步計算：非關鍵計算異步執行
    5. 自適應記憶體管理：根據記憶體壓力調整策略
    6. 邊緣過擬合控制：
       - 邊緣抑制：檢測並抑制邊緣梯度，防止邊緣過擬合
       - 頻率感知：控制高頻噪聲，保持訓練穩定性
       - 空間感知：基於空間變異數進行正則化
       - LoRA 低秩正則化：針對 LoRA 層的秩懲罰機制
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        optim_bits: int = 32,
        args: Any = None,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
        # Enhanced features configuration
        use_spd: bool = True,
        spd_lambda: float = 0.06,
        use_cautious: bool = True,
        use_orthogonal_grad: bool = False,
        use_adopt_stability: bool = True,
        use_grams: bool = True,
        use_agr: bool = True,
        use_tam: bool = True,
        tam_beta: float = 0.999,
        # 動態自適應學習率功能
        use_dynamic_adaptation: bool = True,
        adaptation_strength: float = 1.0,
        relationship_discovery_interval: int = 100,
        importance_decay: float = 0.95,
        compatibility_threshold: float = 0.3,
        # lr_mask 機制配置
        use_lr_mask: bool = True,
        lr_bump: float = 3e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        warmup_steps: int = 500,
        # Dynamic weight decay configuration
        dynamic_weight_decay: bool = True,
        wd_transition_steps: int = 1000,
        wd_decay_factor: float = 0.7,
        wd_min_ratio: float = 0.1,
        # 記憶體優化配置
        memory_efficient: bool = True,
        vram_budget_gb: float = 16.0,
        cpu_offload_states: bool = True,
        reduce_precision: bool = True,
        adaptive_features: bool = True,
        emergency_simplify: bool = True,
        max_buffer_memory_mb: int = 500,
        # 邊緣過擬合控制參數
        edge_suppression: bool = False,
        edge_penalty: float = 0.1,
        # 空間感知
        spatial_awareness: bool = False,
        frequency_penalty: float = 0.05,
        detail_preservation: float = 0.8,
        edge_threshold: float = 0.6,
        # LoRA 低秩正則化
        lora_rank_penalty: bool = False,
        rank_penalty_strength: float = 0.01,
        low_rank_emphasis: float = 1.2,
        **kwargs
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            optim_bits=optim_bits,
            args=args,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged
        )

        super().__init__(params, defaults)

        # 原有功能開關
        self.use_spd = use_spd
        self.spd_lambda = spd_lambda
        self.use_cautious = use_cautious
        self.use_orthogonal_grad = use_orthogonal_grad
        self.use_adopt_stability = use_adopt_stability
        self.use_grams = use_grams
        self.use_agr = use_agr
        self.use_tam = use_tam
        self.tam_beta = tam_beta

        # 動態自適應功能配置
        self.use_dynamic_adaptation = use_dynamic_adaptation
        self.adaptation_strength = adaptation_strength
        self.relationship_discovery_interval = relationship_discovery_interval
        self.importance_decay = importance_decay
        self.compatibility_threshold = compatibility_threshold

        # lr_mask 機制配置
        self.use_lr_mask = use_lr_mask
        self.lr_bump = lr_bump
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

        # 動態權重衰減配置
        self.dynamic_weight_decay = dynamic_weight_decay
        self.wd_transition_steps = wd_transition_steps
        self.wd_decay_factor = wd_decay_factor
        self.wd_min_ratio = wd_min_ratio

        # 記憶體優化配置
        self.memory_efficient = memory_efficient
        self.vram_budget_gb = vram_budget_gb
        self.cpu_offload_states = cpu_offload_states
        self.reduce_precision = reduce_precision
        self.adaptive_features = adaptive_features
        self.emergency_simplify = emergency_simplify

        # 邊緣過擬合控制配置
        self.edge_suppression = edge_suppression
        self.edge_penalty = edge_penalty
        self.spatial_awareness = spatial_awareness
        self.frequency_penalty = frequency_penalty
        self.detail_preservation = detail_preservation
        self.edge_threshold = edge_threshold
        self.lora_rank_penalty = lora_rank_penalty
        self.rank_penalty_strength = rank_penalty_strength
        self.low_rank_emphasis = low_rank_emphasis

        # 初始化記憶體管理組件
        self.memory_monitor = MemoryMonitor(vram_budget_gb)
        self.buffer_pool = EnhancedBufferPool(max_buffer_memory_mb)
        self.async_manager = AsyncComputeManager()

        # 初始化邊緣過擬合控制組件
        if self.edge_suppression:
            self.edge_cache = {}  # 邊緣計算緩存

        # 壓縮狀態存儲
        self.compressed_relationships = CompressedRelationships()
        self.quantized_importance_scores = {}  # param_id -> int16
        self.last_relationship_update = 0

        # 初始化參數組的元數據
        self._initialize_adaptive_metadata()

        # 存儲初始參數（用於 SPD）
        if self.use_spd:
            self._store_initial_parameters()

        # 啟用 PyTorch 優化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        print(f"[INFO] HinaAdaptive 初始化完成，記憶體預算：{vram_budget_gb}GB")

    def _initialize_adaptive_metadata(self):
        """初始化自適應版本的元數據結構（記憶體優化版本）"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            self.param_groups_metadata[group_idx] = {
                'param_count': len(group['params']),
                'param_list': list(group['params']),
                'compact_states': {}  # 使用緊湊狀態存儲
            }

            # 為每個參數初始化追蹤信息
            for param in group['params']:
                param_id = id(param)

                # 使用量化的重要性分數
                self.quantized_importance_scores[param_id] = quantize_importance_score(1.0)

                # 使用緊湊狀態字典
                compact_state = CompactStateDict()
                compact_state.set_scalar('initial_norm', 0.0)
                compact_state.set_scalar('change_rate', 0.0)
                compact_state.set_scalar('stability', 1.0)

                # lr_mask 狀態初始化（使用半精度）
                if self.use_lr_mask:
                    device = param.device if hasattr(param, 'device') else 'cuda'
                    shape = param.shape

                    if self.reduce_precision:
                        lr_mask = torch.ones(shape, device=device, dtype=torch.bfloat16) * self.defaults['lr']
                    else:
                        lr_mask = torch.ones(shape, device=device, dtype=torch.float32) * self.defaults['lr']

                    compact_state.set_tensor('lr_mask', lr_mask, use_half_precision=self.reduce_precision)
                    compact_state.set_tensor('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
                    compact_state.set_scalar('lr_max', self.defaults['lr'])
                    compact_state.set_scalar('avg_lr', self.defaults['lr'])
                    compact_state.set_scalar('warmup_complete', 0.0)  # 0.0 = False, 1.0 = True

                # 邊緣過擬合控制狀態初始化
                if self.edge_suppression:
                    device = param.device if hasattr(param, 'device') else 'cuda'
                    shape = param.shape

                    # 邊緣歷史追蹤
                    compact_state.set_tensor('edge_history', torch.zeros(shape, device=device, dtype=torch.float32))
                    compact_state.set_tensor('edge_momentum', torch.zeros(shape, device=device, dtype=torch.float32))
                    compact_state.set_scalar('edge_strength', 0.0)

                if self.spatial_awareness:
                    device = param.device if hasattr(param, 'device') else 'cuda'
                    shape = param.shape

                    # 空間感知狀態
                    compact_state.set_tensor('spatial_variance', torch.ones(shape, device=device, dtype=torch.float32))
                    compact_state.set_tensor('detail_tracker', torch.zeros(shape, device=device, dtype=torch.float32))
                    compact_state.set_scalar('spatial_activity', 0.0)

                if self.lora_rank_penalty and len(param.shape) == 2:
                    device = param.device if hasattr(param, 'device') else 'cuda'
                    min_dim = min(param.shape)

                    # LoRA 低秩追蹤
                    compact_state.set_tensor('rank_tracker', torch.zeros(min_dim, device=device, dtype=torch.float32))
                    compact_state.set_scalar('rank_penalty_history', 0.0)

                self.param_groups_metadata[group_idx]['compact_states'][param_id] = compact_state

    def _store_initial_parameters(self):
        """存儲初始參數以供 SPD 使用（記憶體優化版本）"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.requires_grad:
                    if self.cpu_offload_states:
                        # 將初始參數存儲在 CPU 上以節省 VRAM
                        self.initial_params[param] = param.data.clone().detach().cpu()
                    else:
                        self.initial_params[param] = param.data.clone().detach()

    def _get_optimized_buffer(self, shape: Tuple, dtype: torch.dtype,
                            device: torch.device, priority: str = 'normal') -> torch.Tensor:
        """獲取優化的緩衝區"""
        return self.buffer_pool.get_buffer_with_priority(shape, dtype, device, priority)

    def _return_optimized_buffer(self, tensor: torch.Tensor, priority: str = 'normal'):
        """歸還優化的緩衝區"""
        self.buffer_pool.return_buffer(tensor, priority)

    def _check_memory_and_adapt(self):
        """檢查記憶體並自適應調整"""
        if not self.memory_efficient:
            return

        memory_pressure = self.memory_monitor.check_memory_pressure()
        optimizations = self.memory_monitor.suggest_optimizations(memory_pressure)

        if optimizations['reduce_buffer_pool']:
            self.buffer_pool.smart_cleanup(memory_pressure)

        if optimizations['increase_gc_frequency']:
            torch.cuda.empty_cache()

        if optimizations['reduce_precision'] and not self.reduce_precision:
            print(f"[WARNING] 記憶體壓力過大，建議啟用 reduce_precision=True")

        if optimizations['use_checkpoint_offload'] and hasattr(self, '_enable_emergency_offload'):
            self._enable_emergency_offload()

    def _compute_parameter_contribution_score_optimized(self, param, compact_state, param_id):
        """
        優化版本的參數貢獻度分數計算
        使用批量操作和減少記憶體分配
        """
        # 1. 梯度相關的貢獻度分析
        grad_contribution = 0.0
        if param.grad is not None:
            current_grad_norm = torch.norm(param.grad).item()
            grad_contribution = current_grad_norm

        # 2. 參數變化相關的貢獻度分析
        change_contribution = 0.0
        initial_norm = compact_state.get_scalar('initial_norm')

        if initial_norm == 0.0:
            # 首次記錄
            initial_norm = torch.norm(param.data).item()
            compact_state.set_scalar('initial_norm', initial_norm)
        else:
            # 計算相對變化率
            current_norm = torch.norm(param.data).item()
            if initial_norm > 0:
                change_rate = abs(current_norm - initial_norm) / initial_norm
                compact_state.set_scalar('change_rate', change_rate)
                change_contribution = change_rate

        # 3. 參數內在特性分析（採樣計算以節省時間）
        if random.random() < 0.1:  # 10% 機率進行完整計算
            param_variance = torch.var(param.data).item()
            param_sparsity = (param.data.abs() < 1e-6).float().mean().item()
            intrinsic_contribution = param_variance * (1.0 - param_sparsity)
        else:
            # 使用快速估算
            intrinsic_contribution = 0.5

        # 綜合貢獻度分數
        total_contribution = (
            grad_contribution * 0.4 +
            change_contribution * 0.3 +
            intrinsic_contribution * 0.3
        )

        return max(0.01, total_contribution)

    def _discover_parameter_relationships_async(self, group_metadata):
        """異步發現參數關係"""
        def discover_relationships():
            param_list = group_metadata['param_list']
            new_relationships = []

            # 限制參數對數量以控制計算量
            max_pairs = min(100, len(param_list) * (len(param_list) - 1) // 2)
            pair_count = 0

            for i, param1 in enumerate(param_list):
                if pair_count >= max_pairs:
                    break
                if param1.dim() != 2:
                    continue

                for j, param2 in enumerate(param_list[i+1:], i+1):
                    if pair_count >= max_pairs:
                        break
                    if param2.dim() != 2:
                        continue

                    compatibility = self._compute_parameter_compatibility_fast(param1, param2)
                    if compatibility > self.compatibility_threshold:
                        param1_id = id(param1)
                        param2_id = id(param2)
                        interaction_type = self._determine_interaction_type_fast(param1, param2)

                        new_relationships.append({
                            'param1_id': param1_id,
                            'param2_id': param2_id,
                            'compatibility': compatibility,
                            'interaction_type': interaction_type
                        })
                        pair_count += 1

            return new_relationships

        # 提交異步任務
        return self.async_manager.submit_async_task(discover_relationships)

    def _compute_parameter_compatibility_fast(self, param1, param2):
        """快速版本的參數相容性計算"""
        if param1.dim() != 2 or param2.dim() != 2:
            return 0.0

        shape1, shape2 = param1.shape, param2.shape

        # 檢查矩陣乘法可能性
        multiplication_checks = [
            shape1[1] == shape2[0],
            shape1[0] == shape2[1],
            shape1[1] == shape2[1],
            shape1[0] == shape2[0]
        ]

        if not any(multiplication_checks):
            return 0.0

        # 簡化的相關性計算（採樣版本）
        try:
            # 只使用前 1000 個元素進行相關性計算
            flat1 = param1.data.flatten()[:1000]
            flat2 = param2.data.flatten()[:1000]

            min_size = min(len(flat1), len(flat2))
            if min_size > 1:
                flat1 = flat1[:min_size]
                flat2 = flat2[:min_size]

                correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
                if torch.isnan(correlation):
                    correlation = 0.0
                else:
                    correlation = abs(correlation.item())
            else:
                correlation = 0.0

            shape_compatibility = sum(multiplication_checks) / len(multiplication_checks)
            total_compatibility = (shape_compatibility * 0.7 + correlation * 0.3)

            return total_compatibility

        except Exception as e:
            #print(f"[DEBUG] 快速相容性計算失敗: {e}")
            return 0.0

    @staticmethod
    def _determine_interaction_type_fast(param1: torch.Tensor, param2: torch.Tensor) -> str:
        """快速確定交互類型"""
        shape1, shape2 = param1.shape, param2.shape

        if shape1[1] == shape2[0]:
            return 'matmul_12'
        elif shape1[0] == shape2[1]:
            return 'matmul_21'
        elif shape1[1] == shape2[1]:
            return 'matmul_12t'
        elif shape1[0] == shape2[0]:
            return 'matmul_1t2'
        else:
            return 'norm_based'

    def _update_importance_scores_batch(self, group_metadata):
        """批量更新重要性分數"""
        param_ids = []
        contribution_scores = []

        # 批量收集貢獻度分數
        for param in group_metadata['param_list']:
            param_id = id(param)
            compact_state = group_metadata['compact_states'].get(param_id)

            if compact_state is not None:
                contribution = self._compute_parameter_contribution_score_optimized(
                    param, compact_state, param_id
                )
                param_ids.append(param_id)
                contribution_scores.append(contribution)

        # 批量更新量化的重要性分數
        for param_id, contribution in zip(param_ids, contribution_scores):
            old_quantized = self.quantized_importance_scores.get(param_id, quantize_importance_score(1.0))
            old_importance = dequantize_importance_score(old_quantized)

            new_importance = (
                self.importance_decay * old_importance +
                (1 - self.importance_decay) * contribution
            )

            self.quantized_importance_scores[param_id] = quantize_importance_score(new_importance)

    def _apply_orthogonal_gradient_optimized(self, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        記憶體優化版本的正交梯度投影
        """
        param_norm = torch.norm(param.data, p=2)
        if param_norm <= 1e-30:
            return grad

        param_flat = param.data.view(-1)
        grad_flat = grad.view(-1)

        if param_flat.shape != grad_flat.shape:
            return grad

        # 使用 JIT 編譯的核心函數
        orthogonal_grad_flat = orthogonal_gradient_core_optimized(grad_flat, param_flat, 1e-30)

        return orthogonal_grad_flat.view_as(grad)

    def _compute_adaptive_lr_scale_optimized(self, param, group_metadata, state, grad=None, global_step=None):
        """
        優化版本的自適應學習率縮放計算
        """
        param_id = id(param)
        compact_state = group_metadata['compact_states'].get(param_id)

        if compact_state is None:
            return 1.0

        # === lr_mask 基礎調整 ===
        lr_mask_scale = 1.0
        if self.use_lr_mask and grad is not None and global_step is not None:
            lr_mask_scale = self._update_lr_mask_optimized(compact_state, grad, global_step)

        # === 自適應高級調整 ===
        adaptive_scale = 1.0
        if self.use_dynamic_adaptation:
            # 1. 基於重要性的調整
            quantized_importance = self.quantized_importance_scores.get(param_id, quantize_importance_score(1.0))
            importance = dequantize_importance_score(quantized_importance)
            importance_factor = min(3.0, max(0.1, importance * self.adaptation_strength))

            # 2. 基於參數關係的調整（簡化版本）
            relationship_scale = 1.0
            rel_info = self.compressed_relationships.get_relationship(param_id)
            if rel_info is not None:
                compatibility_bonus = rel_info['compatibility']
                relationship_scale = 1.0 + compatibility_bonus * 0.2

            adaptive_scale = importance_factor * relationship_scale
            adaptive_scale = max(0.01, min(5.0, adaptive_scale))

        # === 組合最終縮放因子 ===
        # 處理 lr_mask_scale 可能是張量的情況
        if isinstance(lr_mask_scale, torch.Tensor):
            # 如果 lr_mask_scale 是張量，直接與 adaptive_scale 相乘
            final_scale = lr_mask_scale * adaptive_scale
            # 對張量使用 torch.clamp 而不是 max/min
            final_scale = torch.clamp(final_scale, min=0.001, max=10.0)
        else:
            # 如果是標量，使用原來的邏輯
            final_scale = lr_mask_scale * adaptive_scale
            final_scale = max(0.001, min(10.0, final_scale))

        return final_scale

    def _update_lr_mask_optimized(self, compact_state, grad, global_step):
        """優化版本的 lr_mask 更新"""
        if not self.use_lr_mask:
            return 1.0

        # 獲取或初始化 lr_mask
        lr_mask = compact_state.get_tensor('lr_mask', torch.float32)
        if lr_mask is None:
            device = grad.device
            shape = grad.shape
            dtype = torch.bfloat16 if self.reduce_precision else torch.float32
            lr_mask = torch.ones(shape, device=device, dtype=dtype) * self.defaults['lr']
            compact_state.set_tensor('lr_mask', lr_mask, use_half_precision=self.reduce_precision)

        if global_step < self.warmup_steps:
            return self._update_warmup_lr_mask_optimized(compact_state, grad, global_step)
        else:
            return self._update_post_warmup_lr_mask_optimized(compact_state, grad, global_step)

    def _update_warmup_lr_mask_optimized(self, compact_state, grad, global_step):
        """優化版本的 warmup lr_mask 更新"""
        # 使用新的 get_tensor 方法直接獲取正確設備上的張量
        last_polarity = compact_state.get_tensor('last_polarity', target_device=grad.device)
        current_polarity = (grad > 0)

        if last_polarity is not None:
            sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
        else:
            sign_agree = torch.ones_like(current_polarity, dtype=torch.float32, device=grad.device)

        compact_state.set_tensor('last_polarity', current_polarity)

        # 使用新的 get_tensor 方法獲取正確設備和精度的 lr_mask
        lr_mask = compact_state.get_tensor('lr_mask', target_dtype=torch.float32, target_device=grad.device)

        # 使用 JIT 編譯的核心更新函數
        new_lr_mask = compute_lr_mask_update_core(lr_mask, sign_agree, self.lr_bump, self.min_lr, self.max_lr)

        # 更新狀態
        if self.reduce_precision:
            compact_state.set_tensor('lr_mask', new_lr_mask.to(torch.bfloat16), use_half_precision=True)
        else:
            compact_state.set_tensor('lr_mask', new_lr_mask)

        compact_state.set_scalar('avg_lr', torch.mean(new_lr_mask).item())

        # 返回相對縮放因子
        base_lr = self.defaults['lr']
        lr_scale = new_lr_mask / base_lr if base_lr > 0 else new_lr_mask

        return lr_scale

    def _update_post_warmup_lr_mask_optimized(self, compact_state, grad, global_step):
        """優化版本的 post-warmup lr_mask 更新"""
        warmup_complete = compact_state.get_scalar('warmup_complete')
        if warmup_complete < 0.5:  # False
            compact_state.set_scalar('warmup_complete', 1.0)  # True

        # 使用新的 get_tensor 方法獲取正確設備和精度的 lr_mask
        lr_mask = compact_state.get_tensor('lr_mask', target_dtype=torch.float32, target_device=grad.device)

        # 如果 lr_mask 為 None，返回標量縮放因子
        if lr_mask is None:
            return 1.0

        # Post-warmup 階段保持穩定
        base_lr = self.defaults['lr']
        lr_scale = lr_mask / base_lr if base_lr > 0 else lr_mask

        return lr_scale

    def _apply_spd_regularization_optimized(self, param, group, state):
        """應用 SPD 正則化（記憶體優化版本）"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]

        # 如果初始參數在 CPU 上，需要移到相同設備
        if initial_param.device != param.data.device:
            if self.cpu_offload_states:
                # 臨時移動到 GPU 進行計算
                initial_param_gpu = initial_param.to(param.data.device)
                param_diff = param.data - initial_param_gpu
                # 立即清理臨時張量
                del initial_param_gpu
            else:
                param_diff = param.data - initial_param
        else:
            param_diff = param.data - initial_param

        # 計算偏差比率
        param_norm = torch.norm(param.data)
        diff_norm = torch.norm(param_diff)

        if param_norm > 0:
            bias_ratio = diff_norm / param_norm
        else:
            bias_ratio = 0

        # SPD 懲罰項
        spd_penalty = self.spd_lambda * bias_ratio * param_diff

        return spd_penalty

    @staticmethod
    @torch.jit.script
    def _apply_agr_regularization_optimized(grad: torch.Tensor) -> torch.Tensor:
        """應用 AGR 正則化（JIT 優化版本）"""
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            clip_factor = 1.0 / grad_norm
            return grad * clip_factor
        return grad

    @staticmethod
    @torch.jit.script
    def _apply_cautious_update_optimized(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """應用謹慎更新策略（JIT 優化版本）"""
        update_flat = update.view(-1)
        grad_flat = grad.view(-1)

        update_norm = torch.norm(update_flat)
        grad_norm = torch.norm(grad_flat)

        if update_norm > 0 and grad_norm > 0:
            alignment = torch.dot(update_flat, grad_flat) / (update_norm * grad_norm)
            if alignment < 0.1:
                return update * 0.5

        return update

    def _apply_tam_damping_optimized(self, momentum, grad, state):
        """應用 TAM 阻尼（記憶體優化版本）"""
        if 'momentum_alignment' not in state:
            state['momentum_alignment'] = 0.0

        try:
            momentum_norm = torch.norm(momentum)
            grad_norm = torch.norm(grad)

            if momentum_norm > 0 and grad_norm > 0:
                momentum_flat = momentum.view(-1)
                grad_flat = grad.view(-1)

                if momentum_flat.size() == grad_flat.size():
                    alignment = torch.dot(momentum_flat, grad_flat) / (momentum_norm * grad_norm)
                    alignment = alignment.item()
                else:
                    alignment = 0.0
            else:
                alignment = 0.0
        except Exception as e:
            # print(f"[DEBUG] TAM 計算對齊度失敗: {e}")
            alignment = 0.0

        # 平滑對齊估計
        state['momentum_alignment'] = (
            self.tam_beta * state['momentum_alignment'] +
            (1 - self.tam_beta) * alignment
        )

        # 計算阻尼因子
        damping_factor = (1 + state['momentum_alignment']) / 2
        return damping_factor

    def _compute_edge_penalty_optimized(self, grad: torch.Tensor, threshold: float = 0.6,
                                      cache_key: Optional[str] = None) -> torch.Tensor:
        """
        優化版邊緣懲罰計算，用於控制邊緣過擬合

        Args:
            grad: 梯度張量
            threshold: 邊緣檢測閾值
            cache_key: 緩存鍵值（可選）

        Returns:
            邊緣懲罰張量
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        # 檢查緩存（如果啟用）
        if cache_key and hasattr(self, 'edge_cache'):
            cached = self.edge_cache.get(cache_key)
            if cached is not None and cached.shape == grad.shape:
                return cached

        with torch.no_grad():
            # 使用緩衝區池獲取張量
            laplacian = self.buffer_pool.get_buffer_with_priority(
                grad.shape, grad.dtype, grad.device, priority='normal'
            )

            # 簡化的邊緣檢測：計算拉普拉斯算子
            if len(grad.shape) == 2 and grad.shape[0] > 2 and grad.shape[1] > 2:
                # 水平方向二階導數
                laplacian[1:-1, :] = grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
                # 垂直方向二階導數
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]

            # 計算邊緣強度
            edge_strength = torch.abs(laplacian)
            edge_mask = (edge_strength > threshold).float()
            result = edge_mask * edge_strength

            # 緩存結果
            if cache_key and hasattr(self, 'edge_cache'):
                self.edge_cache[cache_key] = result.clone()

            # 歸還緩衝區
            self.buffer_pool.return_buffer(laplacian)

            return result

    def _compute_frequency_penalty_simplified(self, grad: torch.Tensor) -> torch.Tensor:
        """
        簡化版頻率懲罰計算，用於控制高頻噪聲

        Args:
            grad: 梯度張量

        Returns:
            頻率懲罰張量
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        with torch.no_grad():
            # 使用簡化的高頻檢測：計算相鄰元素差異
            if len(grad.shape) == 2:
                h, w = grad.shape
                if h > 1 and w > 1:
                    # 獲取緩衝區
                    result = self.buffer_pool.get_buffer_with_priority(
                        grad.shape, grad.dtype, grad.device, priority='normal'
                    )

                    # 計算水平和垂直差異
                    h_diff = torch.abs(grad[:, 1:] - grad[:, :-1])
                    v_diff = torch.abs(grad[1:, :] - grad[:-1, :])

                    # 組合差異信息
                    result[:, 1:] = h_diff
                    result[1:, :] += v_diff

                    return result

            return torch.zeros_like(grad)

    def _lora_rank_regularization_fast(self, param: torch.Tensor, rank_strength: float = 0.01,
                                     use_approx: bool = True) -> torch.Tensor:
        """
        快速 LoRA 低秩正則化，用於控制 LoRA 層過擬合

        Args:
            param: 參數張量
            rank_strength: 秩正則化強度
            use_approx: 是否使用近似方法

        Returns:
            低秩正則化懲罰張量
        """
        if len(param.shape) != 2:
            return torch.zeros_like(param)

        with torch.no_grad():
            if use_approx:
                # 使用近似方法：只考慮最大的幾個奇異值
                # 計算協方差矩陣
                if param.shape[0] <= param.shape[1]:
                    cov = torch.mm(param, param.t())
                else:
                    cov = torch.mm(param.t(), param)

                # 計算特徵值（只取前幾個）
                try:
                    eigenvals, _ = torch.linalg.eigh(cov)
                    # 懲罰較大的特徵值（促進低秩）
                    large_eigenvals = eigenvals[eigenvals.argsort(descending=True)[:10]]
                    rank_penalty_scalar = torch.sum(large_eigenvals) * rank_strength

                    # 創建梯度近似
                    return param * rank_penalty_scalar
                except Exception as e:
                    # print(f"[DEBUG] LoRA 秩正則化計算失敗: {e}")
                    return torch.zeros_like(param)
            else:
                # 完整 SVD（如果需要）
                try:
                    U, S, Vh = torch.linalg.svd(param, full_matrices=False)
                    # 懲罰較大的奇異值
                    large_s = S[S.argsort(descending=True)[:10]]
                    rank_penalty = torch.sum(large_s) * rank_strength
                    penalty_grad = U @ torch.diag(S * rank_penalty / torch.sum(S)) @ Vh
                    return penalty_grad
                except Exception as e:
                    # print(f"[DEBUG] 完整 SVD 秩正則化計算失敗: {e}")
                    return torch.zeros_like(param)

    def _apply_spatial_awareness_regularization(self, grad: torch.Tensor, state: dict) -> torch.Tensor:
        """
        應用空間感知正則化，用於控制空間過擬合

        Args:
            grad: 梯度張量
            state: 參數狀態

        Returns:
            正則化後的梯度
        """
        if len(grad.shape) < 2:
            return grad

        with torch.no_grad():
            # 初始化空間變異數追蹤
            if 'spatial_variance' not in state:
                state['spatial_variance'] = torch.ones_like(grad)

            if 'detail_tracker' not in state:
                state['detail_tracker'] = torch.zeros_like(grad)

            # 計算局部變異數
            local_variance = torch.zeros_like(grad)
            if len(grad.shape) == 2 and grad.shape[0] > 2 and grad.shape[1] > 2:
                # 計算3x3鄰域的變異數
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue

                        # 計算偏移後的差異
                        if i == 0:
                            if j == 1:
                                local_variance[:, :-1] += torch.pow(grad[:, 1:] - grad[:, :-1], 2)
                            elif j == -1:
                                local_variance[:, 1:] += torch.pow(grad[:, :-1] - grad[:, 1:], 2)
                        elif j == 0:
                            if i == 1:
                                local_variance[:-1, :] += torch.pow(grad[1:, :] - grad[:-1, :], 2)
                            elif i == -1:
                                local_variance[1:, :] += torch.pow(grad[:-1, :] - grad[1:, :], 2)

            # 更新空間變異數追蹤
            state['spatial_variance'] = (
                0.9 * state['spatial_variance'] +
                0.1 * local_variance
            )

            # 基於空間變異數調整梯度
            regularization_factor = 1.0 / (1.0 + state['spatial_variance'] * self.detail_preservation)

            return grad * regularization_factor

    @torch.no_grad()
    def step(self, closure=None):
        """執行優化步驟 - 記憶體優化版本"""
        loss = None
        if closure is not None:
            loss = closure()

        # 檢查記憶體並自適應調整
        self._check_memory_and_adapt()

        # 全局步數計數
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.global_step += 1

        # 收集已完成的異步任務
        completed_relationships = self.async_manager.collect_completed_tasks()
        for relationship_batch in completed_relationships:
            for rel in relationship_batch:
                self.compressed_relationships.add_relationship(
                    rel['param1_id'], rel['param2_id'],
                    rel['compatibility'], rel['interaction_type']
                )

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = self.param_groups_metadata[group_idx]

            # 定期更新參數關係和重要性分數
            should_update_relationships = (
                self.global_step - self.last_relationship_update >=
                self.relationship_discovery_interval
            )

            if should_update_relationships:
                # print(f"[DEBUG] 第 {self.global_step} 步：更新參數關係和重要性分數")

                # 批量更新重要性分數
                self._update_importance_scores_batch(group_metadata)

                # 異步重新發現參數關係
                if self.use_dynamic_adaptation and self.adaptive_features:
                    self._discover_parameter_relationships_async(group_metadata)

                self.last_relationship_update = self.global_step

            # 處理每個參數
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('HinaAdaptive 不支援稀疏梯度')

                state = self.state[param]
                param_id = id(param)
                compact_state = group_metadata['compact_states'].get(param_id)

                # 狀態初始化
                if len(state) == 0:
                    state['step'] = 0

                    # 根據記憶體設置選擇精度
                    if self.reduce_precision:
                        state['exp_avg'] = torch.zeros_like(param.data, dtype=torch.bfloat16)
                        state['exp_avg_sq'] = torch.zeros_like(param.data, dtype=torch.bfloat16)
                        if self.use_adopt_stability:
                            state['exp_avg_sq_prev'] = torch.zeros_like(param.data, dtype=torch.bfloat16)
                    else:
                        state['exp_avg'] = torch.zeros_like(param.data, dtype=torch.float32)
                        state['exp_avg_sq'] = torch.zeros_like(param.data, dtype=torch.float32)
                        if self.use_adopt_stability:
                            state['exp_avg_sq_prev'] = torch.zeros_like(param.data, dtype=torch.float32)

                state['step'] += 1

                beta1, beta2 = group['betas']
                step_size = group['lr']

                # AGR 正則化
                if self.use_agr:
                    grad = HinaAdaptive._apply_agr_regularization_optimized(grad)

                # === 邊緣過擬合控制 ===
                if len(grad.shape) >= 2:
                    # 邊緣感知的梯度正則化
                    if self.edge_suppression:
                        cache_key = f"edge_p_{param_id}_{state['step']}"
                        edge_penalty = self._compute_edge_penalty_optimized(
                            grad, self.edge_threshold, cache_key
                        )

                        # 應用邊緣懲罰
                        if edge_penalty.numel() > 0:
                            edge_factor = 1.0 + self.edge_penalty * edge_penalty
                            grad = grad * (1.0 / edge_factor)

                            # 更新邊緣歷史
                            if compact_state is not None:
                                edge_history = compact_state.get_tensor('edge_history', target_device=grad.device)
                                if edge_history is not None:
                                    edge_history = 0.9 * edge_history + 0.1 * edge_penalty
                                    compact_state.set_tensor('edge_history', edge_history)
                                    compact_state.set_scalar('edge_strength', torch.mean(edge_penalty).item())

                    # 頻率感知的梯度調整
                    if self.spatial_awareness:
                        freq_penalty = self._compute_frequency_penalty_simplified(grad)
                        if freq_penalty.numel() > 0:
                            grad = grad - self.frequency_penalty * freq_penalty

                            # 更新空間活動度
                            if compact_state is not None:
                                spatial_activity = torch.mean(torch.abs(freq_penalty)).item()
                                compact_state.set_scalar('spatial_activity', spatial_activity)

                    # 應用空間感知正則化
                    if self.spatial_awareness:
                        grad = self._apply_spatial_awareness_regularization(grad, state)



                # LoRA 低秩正則化
                if self.lora_rank_penalty and len(param.shape) == 2:
                    rank_penalty = self._lora_rank_regularization_fast(
                        param, self.rank_penalty_strength, use_approx=True
                    )
                    if rank_penalty.numel() > 0:
                        grad = grad + rank_penalty

                        # 更新秩追蹤
                        if compact_state is not None:
                            rank_penalty_magnitude = torch.mean(torch.abs(rank_penalty)).item()
                            compact_state.set_scalar('rank_penalty_history', rank_penalty_magnitude)

                # 正交梯度投影
                if self.use_orthogonal_grad:
                    grad = self._apply_orthogonal_gradient_optimized(grad, param)

                # 偏差校正的學習率
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 更新動量估計（考慮精度）
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    state['exp_avg_sq_prev'] = state['exp_avg_sq'].clone()

                # 確保計算在正確精度下進行
                if self.reduce_precision:
                    # 在 bfloat16 精度下計算
                    grad_bf16 = grad.to(torch.bfloat16)
                    state['exp_avg'].mul_(beta1).add_(grad_bf16, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad_bf16, grad_bf16, value=1 - beta2)
                else:
                    # 在 float32 精度下計算
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 計算更新
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    exp_avg_sq_hat = torch.maximum(state['exp_avg_sq'], state['exp_avg_sq_prev'])
                    denom = (exp_avg_sq_hat.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # 將更新轉換為 float32 以保持精度
                if self.reduce_precision:
                    exp_avg_corrected = (state['exp_avg'].to(torch.float32) / bias_correction1)
                    denom = denom.to(torch.float32)
                else:
                    exp_avg_corrected = state['exp_avg'] / bias_correction1

                update = exp_avg_corrected / denom

                # TAM 阻尼
                if self.use_tam:
                    damping_factor = self._apply_tam_damping_optimized(state['exp_avg'], grad, state)
                    update = update * damping_factor

                # 謹慎更新
                if self.use_cautious:
                    update = HinaAdaptive._apply_cautious_update_optimized(update, grad)

                # 動態自適應學習率調整
                current_step_size = step_size
                if (self.use_dynamic_adaptation or self.use_lr_mask) and compact_state is not None:
                    lr_scale = self._compute_adaptive_lr_scale_optimized(
                        param, group_metadata, state, grad=grad, global_step=self.global_step
                    )

                    # 處理張量或標量的 lr_scale
                    if isinstance(lr_scale, torch.Tensor):
                        if lr_scale.numel() == 1:
                            current_step_size *= lr_scale.item()
                        else:
                            # 確保 lr_scale 與 update 在同一設備上
                            if lr_scale.device != update.device:
                                lr_scale = lr_scale.to(update.device)
                            # 元素級學習率調整
                            param.data.add_(update * lr_scale, alpha=-step_size)
                            current_step_size = 0  # 跳過後續的應用更新
                    else:
                        current_step_size *= lr_scale

                # 應用更新（如果還沒有應用）
                if current_step_size != 0:
                    param.data.add_(update, alpha=-current_step_size)

                # 權重衰減
                current_weight_decay = group['weight_decay']

                # 動態權重衰減
                if self.dynamic_weight_decay:
                    if state['step'] > self.wd_transition_steps:
                        progress = (state['step'] - self.wd_transition_steps) / self.wd_transition_steps
                        decay_multiplier = max(
                            self.wd_min_ratio,
                            self.wd_decay_factor ** min(progress, 2.0)
                        )
                        current_weight_decay *= decay_multiplier

                if current_weight_decay != 0:
                    param.data.add_(param.data, alpha=-group['lr'] * current_weight_decay)

                # SPD 正則化
                if self.use_spd:
                    spd_penalty = self._apply_spd_regularization_optimized(param, group, state)
                    if isinstance(spd_penalty, torch.Tensor):
                        param.data.add_(spd_penalty, alpha=-group['lr'])

                p_fp32.data = param.data.clone().to(torch.float32)

                copy_stochastic(param.data, p_fp32.data)

        return loss

    def cleanup_resources(self):
        """清理資源並釋放記憶體"""
        # 清理緩衝區池
        self.buffer_pool._buffer_pool.clear()
        self.buffer_pool._current_memory = 0

        # 關閉異步管理器
        self.async_manager.shutdown()

        # 清理壓縮關係
        self.compressed_relationships.param_pairs.clear()
        self.compressed_relationships.compatibility_scores = torch.tensor([], dtype=torch.float16)
        self.compressed_relationships.interaction_types.clear()

        # 清理量化重要性分數
        self.quantized_importance_scores.clear()

        # 清理邊緣緩存
        if hasattr(self, 'edge_cache'):
            self.edge_cache.clear()

        # 強制垃圾回收
        torch.cuda.empty_cache()

        print("[INFO] 已清理所有優化器資源")

    def __del__(self):
        """析構函數，確保資源被正確清理"""
        try:
            self.cleanup_resources()
        except Exception as e:
            print(f"[ERROR] 清理資源時發生錯誤: {e}")