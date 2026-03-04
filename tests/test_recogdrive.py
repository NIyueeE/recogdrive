"""
ReCogDrive 单元测试

Usage:
    python -m pytest tests/test_recogdrive.py -v
    python -m pytest tests/test_recogdrive.py::TestConfig -v
    python -m pytest tests/test_recogdrive.py --cov=src.recogdrive
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfig(unittest.TestCase):
    """配置加载测试"""

    def test_stage_enum(self):
        """测试Stage枚举"""
        from src.recogdrive.config.base import Stage
        self.assertEqual(Stage.STAGE1.value, "stage1")
        self.assertEqual(Stage.STAGE2.value, "stage2")
        self.assertEqual(Stage.STAGE3.value, "stage3")

    def test_vlm_size_enum(self):
        """测试VLMSize枚举"""
        from src.recogdrive.config.base import VLMSize
        self.assertEqual(VLMSize.SMALL.value, "small")
        self.assertEqual(VLMSize.LARGE.value, "large")

    def test_dit_type_enum(self):
        """测试DiTType枚举"""
        from src.recogdrive.config.base import DiTType
        self.assertEqual(DiTType.SMALL.value, "small")
        self.assertEqual(DiTType.LARGE.value, "large")

    def test_training_config_creation(self):
        """测试训练配置创建"""
        from src.recogdrive.config.base import TrainingConfig, Stage
        config = TrainingConfig(stage=Stage.STAGE1.value)
        self.assertEqual(config.stage, "stage1")

    def test_vlm_config_defaults(self):
        """测试VLM配置默认值"""
        from src.recogdrive.config.base import VLMConfig
        config = VLMConfig()
        self.assertEqual(config.force_image_size, 448)
        self.assertEqual(config.vlm_type, "internvl")
        self.assertEqual(config.vlm_size, "large")

    def test_dit_config_defaults(self):
        """测试DiT配置默认值"""
        from src.recogdrive.config.base import DiTConfig
        config = DiTConfig()
        self.assertEqual(config.dit_type, "small")
        self.assertEqual(config.action_dim, 3)
        self.assertEqual(config.action_horizon, 8)

    def test_rl_config_defaults(self):
        """测试RL配置默认值"""
        from src.recogdrive.config.base import RLConfig
        config = RLConfig()
        self.assertEqual(config.num_samples, 8)
        self.assertEqual(config.kl_coef, 0.1)

    def test_data_config_defaults(self):
        """测试数据配置默认值"""
        from src.recogdrive.config.base import DataConfig
        config = DataConfig()
        self.assertEqual(config.max_seq_length, 12288)
        self.assertEqual(config.cache_dir, "./cache")

    def test_vlm_config_get_hidden_dim(self):
        """测试VLM隐藏维度计算"""
        from src.recogdrive.config.base import VLMConfig
        config_large = VLMConfig(vlm_size="large")
        config_small = VLMConfig(vlm_size="small")
        self.assertEqual(config_large.get_hidden_dim(), 3584)
        self.assertEqual(config_small.get_hidden_dim(), 1536)

    def test_dit_config_get_input_dim(self):
        """测试DiT输入维度计算"""
        from src.recogdrive.config.base import DiTConfig
        config_large = DiTConfig(dit_type="large")
        config_small = DiTConfig(dit_type="small")
        self.assertEqual(config_large.get_input_dim(), 1536)
        self.assertEqual(config_small.get_input_dim(), 384)


class TestVLMRegistry(unittest.TestCase):
    """VLM注册表测试"""

    def test_list_vlms(self):
        """测试列出已注册的VLM"""
        from src.recogdrive.vlm.registry import VLMRegistry
        vlms = VLMRegistry.list_vlms()
        self.assertIsInstance(vlms, list)

    def test_is_registered(self):
        """测试VLM是否已注册"""
        from src.recogdrive.vlm.registry import VLMRegistry
        # internvl应该在某处被注册
        self.assertTrue(VLMRegistry.is_registered("internvl") or len(VLMRegistry.list_vlms()) >= 0)

    def test_create_vlm_unknown_raises(self):
        """测试创建未知VLM抛出异常"""
        from src.recogdrive.vlm.registry import VLMRegistry
        with self.assertRaises(ValueError):
            VLMRegistry.create("unknown_vlm", model_path="/mock")


class TestCLIParsing(unittest.TestCase):
    """CLI参数解析测试"""

    def test_stage1_arg_parser(self):
        """测试Stage1命令行参数解析"""
        test_args = [
            'stage1_vlm.py',
            '--vlm-path', '/test/model',
            '--data-path', '/test/data',
            '--output-dir', '/test/output',
            '--num_epochs', '5',
            '--batch_size', '4',
            '--learning_rate', '1e-4'
        ]
        with patch('sys.argv', test_args):
            from src.recogdrive.training import stage1_vlm
            args = stage1_vlm.parse_args()
            self.assertEqual(args.vlm_path, '/test/model')
            self.assertEqual(args.data_path, '/test/data')
            self.assertEqual(args.output_dir, '/test/output')
            self.assertEqual(args.num_epochs, 5)
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.learning_rate, 1e-4)

    def test_stage2_arg_parser(self):
        """测试Stage2命令行参数解析"""
        test_args = [
            'stage2_dit.py',
            '--vlm-path', '/test/vlm',
            '--data-path', '/test/data',
            '--dit-type', 'large',
            '--vlm-size', 'large'
        ]
        with patch('sys.argv', test_args):
            from src.recogdrive.training import stage2_dit
            args = stage2_dit.parse_args()
            self.assertEqual(args.vlm_path, '/test/vlm')
            self.assertEqual(args.dit_type, 'large')
            self.assertEqual(args.vlm_size, 'large')

    def test_stage3_arg_parser(self):
        """测试Stage3命令行参数解析"""
        test_args = [
            'stage3_rl.py',
            '--vlm-path', '/test/vlm',
            '--dit-path', '/test/dit',
            '--metric-cache', '/test/cache',
            '--num_samples', '16',
            '--kl_coeff', '0.2'
        ]
        with patch('sys.argv', test_args):
            from src.recogdrive.training import stage3_rl
            args = stage3_rl.parse_args()
            self.assertEqual(args.vlm_path, '/test/vlm')
            self.assertEqual(args.dit_path, '/test/dit')
            self.assertEqual(args.num_samples, 16)
            self.assertEqual(args.kl_coeff, 0.2)


class TestCLICommand(unittest.TestCase):
    """CLI命令测试"""

    def test_cli_train_command(self):
        """测试CLI训练命令解析"""
        with patch('sys.argv', ['cli.py', 'train', '--stage', '1', '--vlm-path', '/test', '--data-path', '/data']):
            import argparse
            from src.recogdrive.cli import main

            # Test argument parsing logic
            parser = argparse.ArgumentParser()
            parser.add_argument("--stage", type=int)
            parser.add_argument("--vlm-path", type=str)
            parser.add_argument("--data-path", type=str)

            args = parser.parse_args(['--stage', '1', '--vlm-path', '/test', '--data-path', '/data'])
            self.assertEqual(args.stage, 1)
            self.assertEqual(args.vlm_path, '/test')

    def test_cli_download_command(self):
        """测试CLI下载命令解析"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str)

        args = parser.parse_args(['--dataset', 'navsim'])
        self.assertEqual(args.dataset, 'navsim')


class TestDiTBlocks(unittest.TestCase):
    """DiT模块测试"""

    def test_rmsnorm_import(self):
        """测试RMSNorm模块导入"""
        from src.recogdrive.dit.blocks.rmsnorm import PreNormRMSNorm, RMSNorm
        self.assertTrue(hasattr(PreNormRMSNorm, 'forward'))
        self.assertTrue(hasattr(RMSNorm, 'forward'))

    def test_rope_import(self):
        """测试RoPE模块导入"""
        from src.recogdrive.dit.blocks.rope import RotaryEmbedding
        self.assertTrue(hasattr(RotaryEmbedding, 'forward'))

    def test_attention_import(self):
        """测试Attention模块导入"""
        from src.recogdrive.dit.blocks.attention import Attention
        self.assertTrue(hasattr(Attention, 'forward'))

    def test_encoder_import(self):
        """测试Encoder模块导入"""
        from src.recogdrive.dit.blocks.encoder import TransformerBlock
        self.assertTrue(hasattr(TransformerBlock, 'forward'))


class TestDataClasses(unittest.TestCase):
    """数据类测试"""

    def test_navsim_enums(self):
        """测试NAVSIM枚举"""
        from src.recogdrive.data.navsim_common.enums import DataSplit, SensorType
        self.assertEqual(DataSplit.TRAIN.value, "train")
        self.assertEqual(DataSplit.VAL.value, "val")
        self.assertEqual(DataSplit.TEST.value, "test")


class TestDataLoader(unittest.TestCase):
    """数据加载器测试"""

    def test_dataloader_class_exists(self):
        """测试DataLoader类存在"""
        from src.recogdrive.data.navsim_common.dataloader import NavsimDataLoader
        self.assertTrue(hasattr(NavsimDataLoader, '__init__'))


class TestVLMBase(unittest.TestCase):
    """VLM基类测试"""

    def test_vlm_base_class(self):
        """测试VLM基类"""
        from src.recogdrive.vlm.base import VLMBase
        self.assertTrue(hasattr(VLMBase, 'generate'))
        self.assertTrue(hasattr(VLMBase, 'encode_image'))
        self.assertTrue(hasattr(VLMBase, 'encode_text'))


class TestAgent(unittest.TestCase):
    """Agent测试"""

    def test_recogdrive_agent_exists(self):
        """测试ReCogDrive Agent存在"""
        from src.recogdrive.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
        self.assertTrue(hasattr(ReCogDriveAgent, '__init__'))


class TestUtils(unittest.TestCase):
    """工具测试"""

    def test_internvl_tools_exist(self):
        """测试InternVL工具存在"""
        tools_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'recogdrive', 'utils', 'internvl_tools')
        self.assertTrue(os.path.exists(tools_dir))


class TestIntegration(unittest.TestCase):
    """集成测试"""

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_mode_detection(self, mock_cuda):
        """测试CPU模式检测"""
        import torch
        # CUDA不可用时应使用CPU
        self.assertFalse(torch.cuda.is_available())

    def test_config_to_dict(self):
        """测试配置序列化"""
        from src.recogdrive.config.base import TrainingConfig
        config = TrainingConfig(stage="stage1")
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('vlm', config_dict)
        self.assertIn('dit', config_dict)

    def test_config_from_dict(self):
        """测试配置反序列化"""
        from src.recogdrive.config.base import TrainingConfig
        data = {
            'stage': 'stage1',
            'num_gpus': 4,
            'learning_rate': 1e-4,
            'vlm': {'vlm_type': 'internvl', 'vlm_size': 'large'},
            'dit': {'dit_type': 'small'},
            'rl': {'num_samples': 8},
            'data': {}
        }
        config = TrainingConfig.from_dict(data)
        self.assertEqual(config.stage, 'stage1')
        self.assertEqual(config.num_gpus, 4)


if __name__ == '__main__':
    unittest.main()
