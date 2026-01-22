# ReCogDrive 代码风格和约定

## Python 风格
- **Python版本**: 3.9
- **代码格式化**: 使用4空格缩进
- **行长限制**: 建议不超过120字符
- **导入顺序**: 标准库 → 第三方库 → 本地模块

## 命名约定
- **类名**: 大驼峰式 (CamelCase), 如 `ReCogDriveAgent`
- **函数/方法名**: 小写加下划线 (snake_case), 如 `make_recogdrive_config`
- **变量名**: 小写加下划线, 如 `cache_hidden_state`
- **常量名**: 大写加下划线, 如 `IGNORE_INDEX`, `MAX_SEQ_LENGTH`
- **私有成员**: 单下划线前缀, 如 `_hidden_state`

## 类型注解
- 使用Python类型注解
- 复杂类型从`typing`模块导入
- 示例:
```python
from typing import Dict, List, Optional

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim'
) -> ReCogDriveDiffusionPlannerConfig:
```

## 文档字符串
- 使用Google风格文档字符串
- 包含Args、Returns、Raises部分
- 示例:
```python
def build_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments
) -> Dict[str, torch.utils.data.Dataset]:
    """
    构建训练和评估数据集。
    
    Args:
        data_args: 数据训练参数
        model_args: 模型参数
        
    Returns:
        包含'train'和'eval'数据集的字典
    """
```

## 配置管理
- 使用Hydra进行配置管理
- 配置文件为YAML格式
- 配置路径: `navsim/planning/script/config/`
- 示例配置:
```yaml
_target_: navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent
_convert_: 'all'
dit_type: 'small'
grpo: False
vlm_type: 'internvl'
sampling_method: 'ddim'
```

## 训练脚本约定
- Shell脚本设置环境变量和启动训练
- 使用`torchrun`进行分布式训练
- 训练参数通过命令行传递
- 日志重定向到文件并tee到控制台

## 错误处理
- 使用try-except处理可能失败的IO操作
- 验证输入参数的有效性
- 使用logging模块记录错误信息
- 示例:
```python
import logging
logger = logging.getLogger(__name__)

try:
    dataset = load_dataset(data_path)
except FileNotFoundError:
    logger.error(f"数据集文件未找到: {data_path}")
    raise
```

## 日志记录
- 使用Python标准logging模块
- 不同级别: DEBUG, INFO, WARNING, ERROR
- 训练过程中记录关键指标
- 示例配置:
```python
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
```

## 文件组织
- **internvl_chat/**: VLM相关代码，遵循InternVL项目结构
- **navsim/**: 规划器代码，集成NAVSIM框架
- **scripts/**: 可执行脚本，按功能分类
- **config/**: 配置文件，按组件分类

## 测试约定
- 单元测试放在`tests/`目录
- 使用pytest测试框架
- 集成测试验证端到端流程
- 模型推理测试验证正确性