import pytest
import torch
import numpy as np
from solver import Instance, InstanceType, load_instance


def sanity_heatmap(heatmap):
    assert heatmap.min() >= 0, "Heatmap values should be non-negative"
    assert heatmap.max() <= 1, "Heatmap values should be at most 1"
    assert not np.isnan(heatmap).any(), "Heatmap should not contain NaN values"
    assert not np.isinf(heatmap).any(), "Heatmap should not contain inf values"


class TestInstance:
    """Test suite for Instance class and heatmap generation"""

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA available")
    def test_heatmap_cpu(self):
        """Test heatmap generation on CPU with a valid 100-node instance"""
        # Create a 100-node instance (supported size)
        np.random.seed(42)
        coordinates = list(np.random.rand(100, 2))

        instance = Instance(
            instance_type=InstanceType.EUC_2D,
            instance_id=0,
            coordinates=coordinates
        )

        # Generate heatmap on CPU
        heatmap = instance._get_heatmap(device='cpu')

        # Verify heatmap properties
        assert heatmap.shape == (100, 100), "Heatmap should be 100x100"
        sanity_heatmap(heatmap)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_heatmap_gpu(self):
        """Test heatmap generation on GPU with a valid 100-node instance"""
        # Create a 100-node instance (supported size)
        np.random.seed(42)
        coordinates = list(np.random.rand(100, 2))

        instance = Instance(
            instance_type=InstanceType.EUC_2D,
            instance_id=1,
            coordinates=coordinates
        )

        # Generate heatmap on GPU
        heatmap = instance._get_heatmap(device='cuda')

        # Verify heatmap properties
        assert heatmap.shape == (100, 100), "Heatmap should be 100x100"
        sanity_heatmap(heatmap)

    def test_custom_instance_fails(self):
        """Test that a custom instance with size 10 fails as expected"""
        instance = load_instance(0, InstanceType.EUC_2D)
        heatmap = instance._get_heatmap(device='cuda')
        assert heatmap.shape == (10, 10), "Heatmap should be 10x10"
        sanity_heatmap(heatmap)


