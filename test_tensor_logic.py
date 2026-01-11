"""
Unit tests for Tensor Logic framework.
"""

import unittest
import numpy as np
from tensor_logic import (
    create_backend, create_strategy,
    Backend, CompilationStrategy,
    logical_and, logical_or, logical_not, logical_implies,
    exists, forall,
    apply_temperature, reason, quantify,
    SoftDifferentiableStrategy, HardBooleanStrategy, GodelStrategy,
    ProductStrategy, LukasiewiczStrategy
)


class TestBackend(unittest.TestCase):
    """Test backend functionality."""
    
    def setUp(self):
        self.backend = create_backend("numpy")
    
    def test_create_backend(self):
        """Test backend creation."""
        backend = create_backend("numpy")
        self.assertIsInstance(backend, Backend)
        self.assertEqual(backend.name, "numpy")
    
    def test_tensor_creation(self):
        """Test tensor creation."""
        data = [[1, 0], [0, 1]]
        tensor = self.backend.tensor(data)
        np.testing.assert_array_equal(tensor, np.array(data, dtype=np.float32))
    
    def test_zeros_ones(self):
        """Test zeros and ones creation."""
        zeros = self.backend.zeros((2, 3))
        ones = self.backend.ones((2, 3))
        self.assertEqual(zeros.shape, (2, 3))
        self.assertEqual(ones.shape, (2, 3))
        np.testing.assert_array_equal(zeros, np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(ones, np.ones((2, 3), dtype=np.float32))
    
    def test_step_function(self):
        """Test step (Heaviside) function."""
        x = np.array([0.2, 0.5, 0.7, 1.0])
        result = self.backend.step(x, threshold=0.5)
        expected = np.array([0., 1., 1., 1.])
        np.testing.assert_array_equal(result, expected)
    
    def test_sigmoid(self):
        """Test sigmoid function."""
        x = np.array([0.0, 1.0, -1.0])
        result = self.backend.sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_allclose(result, expected)
    
    def test_einsum(self):
        """Test Einstein summation."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = self.backend.einsum('ij,jk->ik', a, b)
        expected = np.matmul(a, b)
        np.testing.assert_array_equal(result, expected)


class TestHardBooleanStrategy(unittest.TestCase):
    """Test hard Boolean compilation strategy."""
    
    def setUp(self):
        self.backend = create_backend()
        self.strategy = create_strategy(CompilationStrategy.HARD_BOOLEAN, self.backend)
    
    def test_logical_and(self):
        """Test hard Boolean AND."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = self.strategy.logical_and(a, b)
        expected = np.array([1., 0., 0., 0.])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_or(self):
        """Test hard Boolean OR."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = self.strategy.logical_or(a, b)
        expected = np.array([1., 1., 1., 0.])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_not(self):
        """Test hard Boolean NOT."""
        a = np.array([1.0, 0.0, 0.5])
        result = self.strategy.logical_not(a)
        expected = np.array([0., 1., 1.])  # step(1 - a) with threshold 0.5
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_implies(self):
        """Test hard Boolean IMPLIES."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = self.strategy.logical_implies(a, b)
        # a -> b = not(a) or b
        expected = np.array([1., 0., 1., 1.])
        np.testing.assert_array_equal(result, expected)


class TestGodelStrategy(unittest.TestCase):
    """Test Gödel (min/max) compilation strategy."""
    
    def setUp(self):
        self.backend = create_backend()
        self.strategy = create_strategy(CompilationStrategy.GODEL, self.backend)
    
    def test_logical_and(self):
        """Test Gödel AND (minimum)."""
        a = np.array([0.8, 0.6, 0.3])
        b = np.array([0.9, 0.4, 0.5])
        result = self.strategy.logical_and(a, b)
        expected = np.array([0.8, 0.4, 0.3])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_or(self):
        """Test Gödel OR (maximum)."""
        a = np.array([0.8, 0.6, 0.3])
        b = np.array([0.9, 0.4, 0.5])
        result = self.strategy.logical_or(a, b)
        expected = np.array([0.9, 0.6, 0.5])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_not(self):
        """Test Gödel NOT (1 - x)."""
        a = np.array([0.2, 0.5, 0.8])
        result = self.strategy.logical_not(a)
        expected = np.array([0.8, 0.5, 0.2])
        np.testing.assert_allclose(result, expected)


class TestProductStrategy(unittest.TestCase):
    """Test product (probabilistic) compilation strategy."""
    
    def setUp(self):
        self.backend = create_backend()
        self.strategy = create_strategy(CompilationStrategy.PRODUCT, self.backend)
    
    def test_logical_and(self):
        """Test product AND (multiplication)."""
        a = np.array([0.8, 0.6, 0.5])
        b = np.array([0.9, 0.5, 0.4])
        result = self.strategy.logical_and(a, b)
        expected = np.array([0.72, 0.3, 0.2])
        np.testing.assert_allclose(result, expected)
    
    def test_logical_or(self):
        """Test product OR (a + b - a*b)."""
        a = np.array([0.6, 0.5])
        b = np.array([0.5, 0.4])
        result = self.strategy.logical_or(a, b)
        expected = np.array([0.8, 0.7])  # 0.6+0.5-0.3=0.8, 0.5+0.4-0.2=0.7
        np.testing.assert_allclose(result, expected)


class TestLukasiewiczStrategy(unittest.TestCase):
    """Test Łukasiewicz (bounded) compilation strategy."""
    
    def setUp(self):
        self.backend = create_backend()
        self.strategy = create_strategy(CompilationStrategy.LUKASIEWICZ, self.backend)
    
    def test_logical_and(self):
        """Test Łukasiewicz AND (max(0, a+b-1))."""
        a = np.array([0.8, 0.6, 0.3])
        b = np.array([0.9, 0.5, 0.4])
        result = self.strategy.logical_and(a, b)
        expected = np.array([0.7, 0.1, 0.0])  # max(0, 1.7-1), max(0, 1.1-1), max(0, 0.7-1)
        np.testing.assert_allclose(result, expected)
    
    def test_logical_or(self):
        """Test Łukasiewicz OR (min(1, a+b))."""
        a = np.array([0.8, 0.6, 0.3])
        b = np.array([0.9, 0.5, 0.4])
        result = self.strategy.logical_or(a, b)
        expected = np.array([1.0, 1.0, 0.7])  # min(1, 1.7), min(1, 1.1), min(1, 0.7)
        np.testing.assert_allclose(result, expected)


class TestSoftDifferentiableStrategy(unittest.TestCase):
    """Test soft differentiable compilation strategy."""
    
    def setUp(self):
        self.backend = create_backend()
        self.strategy = create_strategy(CompilationStrategy.SOFT_DIFFERENTIABLE, self.backend)
    
    def test_logical_and(self):
        """Test soft AND (sigmoid(a+b-1))."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        result = self.strategy.logical_and(a, b)
        # sigmoid(1+1-1)=sigmoid(1)≈0.73, sigmoid(0+1-1)=sigmoid(0)=0.5
        self.assertGreater(result[0], 0.7)
        self.assertAlmostEqual(result[1], 0.5, places=5)
    
    def test_logical_or(self):
        """Test soft OR (sigmoid(a+b))."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 1.0])
        result = self.strategy.logical_or(a, b)
        # sigmoid(0+0)=0.5, sigmoid(0+1)≈0.73
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertGreater(result[1], 0.7)
    
    def test_logical_not(self):
        """Test soft NOT (1 - a)."""
        a = np.array([0.2, 0.8])
        result = self.strategy.logical_not(a)
        expected = np.array([0.8, 0.2])
        np.testing.assert_allclose(result, expected)


class TestCoreLogicalOperations(unittest.TestCase):
    """Test core logical operations."""
    
    def setUp(self):
        self.backend = create_backend()
    
    def test_logical_and_default(self):
        """Test logical_and with default parameters."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = logical_and(a, b, backend=self.backend)
        expected = np.array([1., 0., 0., 0.])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_or_default(self):
        """Test logical_or with default parameters."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = logical_or(a, b, backend=self.backend)
        expected = np.array([1., 1., 1., 0.])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_not_default(self):
        """Test logical_not with default parameters."""
        a = np.array([1.0, 0.0])
        result = logical_not(a, backend=self.backend)
        expected = np.array([0., 1.])
        np.testing.assert_array_equal(result, expected)
    
    def test_logical_implies_default(self):
        """Test logical_implies with default parameters."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        result = logical_implies(a, b, backend=self.backend)
        expected = np.array([1., 0., 1., 1.])
        np.testing.assert_array_equal(result, expected)


class TestQuantifiers(unittest.TestCase):
    """Test existential and universal quantifiers."""
    
    def setUp(self):
        self.backend = create_backend()
    
    def test_exists_all_false(self):
        """Test EXISTS with all false."""
        predicate = np.array([0.0, 0.0, 0.0])
        result = exists(predicate, backend=self.backend)
        self.assertEqual(result, 0.0)
    
    def test_exists_one_true(self):
        """Test EXISTS with one true."""
        predicate = np.array([0.0, 1.0, 0.0])
        result = exists(predicate, backend=self.backend)
        self.assertEqual(result, 1.0)
    
    def test_exists_axis(self):
        """Test EXISTS along specific axis."""
        predicate = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]])
        result = exists(predicate, axis=1, backend=self.backend)
        expected = np.array([1., 0., 1.])
        np.testing.assert_array_equal(result, expected)
    
    def test_forall_all_true(self):
        """Test FORALL with all true."""
        predicate = np.array([1.0, 1.0, 1.0])
        result = forall(predicate, backend=self.backend)
        self.assertEqual(result, 1.0)
    
    def test_forall_one_false(self):
        """Test FORALL with one false."""
        predicate = np.array([1.0, 0.0, 1.0])
        result = forall(predicate, backend=self.backend)
        self.assertEqual(result, 0.0)
    
    def test_forall_axis(self):
        """Test FORALL along specific axis."""
        predicate = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = forall(predicate, axis=1, backend=self.backend)
        expected = np.array([1., 0., 1.])
        np.testing.assert_array_equal(result, expected)


class TestTemperatureControlled(unittest.TestCase):
    """Test temperature-controlled reasoning."""
    
    def setUp(self):
        self.backend = create_backend()
    
    def test_temperature_zero(self):
        """Test temperature=0 (pure deductive)."""
        logits = np.array([0.2, 0.5, 0.7, 1.0])
        result = apply_temperature(logits, temperature=0.0, backend=self.backend)
        expected = np.array([0., 1., 1., 1.])  # Hard threshold at 0.5
        np.testing.assert_array_equal(result, expected)
    
    def test_temperature_positive(self):
        """Test temperature>0 (analogical)."""
        logits = np.array([0.0, 1.0, -1.0])
        result = apply_temperature(logits, temperature=1.0, backend=self.backend)
        # Should be sigmoid of logits
        expected = 1 / (1 + np.exp(-logits))
        np.testing.assert_allclose(result, expected)
    
    def test_temperature_scaling(self):
        """Test temperature scaling effect."""
        logits = np.array([2.0])
        result_low = apply_temperature(logits, temperature=0.5, backend=self.backend)
        result_high = apply_temperature(logits, temperature=2.0, backend=self.backend)
        # Lower temperature should give more extreme values
        self.assertGreater(result_low[0], result_high[0])


class TestKnowledgeGraphReasoning(unittest.TestCase):
    """Test knowledge graph reasoning scenarios."""
    
    def setUp(self):
        self.backend = create_backend()
        # Family tree: Alice -> Bob -> Carol
        self.parent = np.array([
            [0., 1., 0.],  # Alice is parent of Bob
            [0., 0., 1.],  # Bob is parent of Carol
            [0., 0., 0.],  # Carol has no children
        ], dtype=np.float32)
    
    def test_grandparent_inference(self):
        """Test grandparent inference via composition."""
        # Grandparent = Parent @ Parent
        grandparent = self.backend.matmul(self.parent, self.parent)
        # Alice should be grandparent of Carol
        self.assertGreater(grandparent[0, 2], 0)
        # Bob should not be grandparent of anyone
        self.assertEqual(np.sum(grandparent[1, :]), 0)
    
    def test_has_children_query(self):
        """Test 'has children' query using EXISTS."""
        # exists y: Parent(x, y)
        has_children = exists(self.parent, axis=1, backend=self.backend)
        expected = np.array([1., 1., 0.])  # Alice and Bob have children, Carol doesn't
        np.testing.assert_array_equal(has_children, expected)
    
    def test_reason_grandparent_deductive(self):
        """Test reason() for grandparent with T=0."""
        result = reason('Grandparent(x, z)', 
                       predicates={'Parent': self.parent},
                       temperature=0.0,
                       backend=self.backend)
        # Alice (0) should be grandparent of Carol (2)
        self.assertEqual(result[0, 2], 1.0)
    
    def test_reason_grandparent_analogical(self):
        """Test reason() for grandparent with T>0."""
        result = reason('Grandparent(x, z)', 
                       predicates={'Parent': self.parent},
                       temperature=1.0,
                       backend=self.backend)
        # Should give continuous values via sigmoid
        self.assertGreater(result[0, 2], 0.5)
        self.assertLess(result[0, 2], 1.0)
    
    def test_quantify_has_children(self):
        """Test quantify() for 'has children' pattern."""
        result = quantify('exists y: Parent(x, y)',
                         predicates={'Parent': self.parent},
                         backend=self.backend)
        expected = np.array([1., 1., 0.])
        np.testing.assert_array_equal(result, expected)
    
    def test_quantify_grandparent(self):
        """Test quantify() for grandparent pattern."""
        result = quantify('exists y: Parent(x, y) and Parent(y, z)',
                         predicates={'Parent': self.parent},
                         backend=self.backend)
        # Should infer grandparent relation
        self.assertEqual(result[0, 2], 1.0)


class TestMultiHopReasoning(unittest.TestCase):
    """Test multi-hop reasoning scenarios."""
    
    def setUp(self):
        self.backend = create_backend()
        # Extended family tree
        self.parent = np.array([
            [0., 1., 1., 0., 0.],  # Alice -> Bob, Carol
            [0., 0., 0., 1., 0.],  # Bob -> David
            [0., 0., 0., 0., 1.],  # Carol -> Eve
            [0., 0., 0., 0., 0.],  # David -> (none)
            [0., 0., 0., 0., 0.],  # Eve -> (none)
        ], dtype=np.float32)
    
    def test_two_hop_reasoning(self):
        """Test 2-hop reasoning (grandparent)."""
        grandparent = self.backend.matmul(self.parent, self.parent)
        # Alice should be grandparent of David and Eve
        self.assertEqual(grandparent[0, 3], 1.0)
        self.assertEqual(grandparent[0, 4], 1.0)
    
    def test_three_hop_reasoning(self):
        """Test 3-hop reasoning (great-grandparent)."""
        grandparent = self.backend.matmul(self.parent, self.parent)
        great_grandparent = self.backend.matmul(grandparent, self.parent)
        # Should propagate relationships
        self.assertIsInstance(great_grandparent, np.ndarray)


class TestCompilationStrategyComparison(unittest.TestCase):
    """Test comparison of different compilation strategies."""
    
    def setUp(self):
        self.backend = create_backend()
        self.a = np.array([0.6, 0.7, 0.8])
        self.b = np.array([0.5, 0.6, 0.7])
    
    def test_and_across_strategies(self):
        """Test AND operation across all strategies."""
        strategies = [
            create_strategy(s, self.backend) 
            for s in CompilationStrategy
        ]
        results = [s.logical_and(self.a, self.b) for s in strategies]
        # All should return arrays of same shape
        for result in results:
            self.assertEqual(result.shape, self.a.shape)
    
    def test_or_across_strategies(self):
        """Test OR operation across all strategies."""
        strategies = [
            create_strategy(s, self.backend) 
            for s in CompilationStrategy
        ]
        results = [s.logical_or(self.a, self.b) for s in strategies]
        # All should return arrays of same shape
        for result in results:
            self.assertEqual(result.shape, self.a.shape)


if __name__ == '__main__':
    unittest.main()
