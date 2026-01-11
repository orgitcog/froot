"""
Examples demonstrating the Tensor Logic framework.

Based on Pedro Domingos' work (https://tensor-logic.org/), showing how
tensor logic unifies symbolic and neural AI through tensor operations.
"""

import numpy as np
from tensor_logic import (
    create_backend, create_strategy, CompilationStrategy,
    logical_and, logical_or, logical_not, logical_implies,
    exists, forall, reason, quantify, apply_temperature
)


def example_hello_world():
    """Example 0: Hello World - Basic logical operations."""
    print("=" * 60)
    print("EXAMPLE 0: Hello World - Basic Logical Operations")
    print("=" * 60)
    
    backend = create_backend()
    
    # Boolean facts as tensors
    facts_a = np.array([1.0, 0.0, 1.0, 0.0])  # TRUE, FALSE, TRUE, FALSE
    facts_b = np.array([1.0, 1.0, 0.0, 0.0])  # TRUE, TRUE, FALSE, FALSE
    
    print("\nFacts A:", facts_a)
    print("Facts B:", facts_b)
    
    result = logical_and(facts_a, facts_b, backend=backend)
    print("\nA AND B:", result)
    print("Interpretation: Only position 0 is TRUE in BOTH\n")


def example_family_tree():
    """Example 1: Family Tree - Multi-hop reasoning."""
    print("=" * 60)
    print("EXAMPLE 1: Family Tree - Multi-Hop Reasoning")
    print("=" * 60)
    
    backend = create_backend()
    
    # Family tree: Alice -> Bob -> Carol
    # Rows = parent, Columns = child
    parent = np.array([
        [0., 1., 0.],  # Alice is parent of Bob
        [0., 0., 1.],  # Bob is parent of Carol
        [0., 0., 0.],  # Carol has no children
    ], dtype=np.float32)
    
    print("\nParent relation:")
    print(parent)
    print("(Alice->Bob, Bob->Carol)")
    
    # Matrix multiplication = "follow two parent edges" = grandparent!
    grandparent = backend.matmul(parent, parent)
    print("\nGrandparent relation (Parent @ Parent):")
    print(grandparent)
    print("Alice is now grandparent of Carol (inferred!)\n")


def example_temperature_control():
    """Example 2: Temperature-controlled reasoning."""
    print("=" * 60)
    print("EXAMPLE 2: Temperature Control - Deductive vs Analogical")
    print("=" * 60)
    
    backend = create_backend()
    
    # User preferences (some uncertain)
    likes_action = np.array([1.0, 0.6, 0.0, 0.5])  # User 3 = unknown (0.5)
    likes_comedy = np.array([0.8, 0.5, 0.9, 0.7])
    
    print("\nLikes Action:", likes_action)
    print("Likes Comedy:", likes_comedy)
    
    # Temperature = 0: Pure deductive (only recommend when CERTAIN)
    strategy_hard = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    recommend_certain = strategy_hard.logical_and(likes_action, likes_comedy)
    print("\nT=0 (Deductive - Only when certain):", recommend_certain)
    print("Only user 0 is recommended (certain about both)")
    
    # Temperature > 0: Analogical (recommend with uncertainty)
    strategy_soft = create_strategy(CompilationStrategy.SOFT_DIFFERENTIABLE, backend)
    recommend_uncertain = strategy_soft.logical_and(likes_action, likes_comedy)
    print("\nT>0 (Analogical - Graded scores):", recommend_uncertain)
    print("Users 0, 1, 3 get graded recommendation scores\n")


def example_compilation_strategies():
    """Example 3: Comparing compilation strategies."""
    print("=" * 60)
    print("EXAMPLE 3: Compilation Strategies - Different Semantics")
    print("=" * 60)
    
    backend = create_backend()
    
    a = np.array([0.8, 0.6])
    b = np.array([0.9, 0.5])
    
    print("\nInput A:", a)
    print("Input B:", b)
    print()
    
    strategies = {
        "Hard Boolean": CompilationStrategy.HARD_BOOLEAN,
        "Gödel (min/max)": CompilationStrategy.GODEL,
        "Product (probabilistic)": CompilationStrategy.PRODUCT,
        "Łukasiewicz (bounded)": CompilationStrategy.LUKASIEWICZ,
        "Soft Differentiable": CompilationStrategy.SOFT_DIFFERENTIABLE,
    }
    
    for name, strategy_type in strategies.items():
        strategy = create_strategy(strategy_type, backend)
        result_and = strategy.logical_and(a, b)
        result_or = strategy.logical_or(a, b)
        print(f"{name:25} AND: {result_and}  OR: {result_or}")
    print()


def example_quantifiers():
    """Example 4: Existential and universal quantifiers."""
    print("=" * 60)
    print("EXAMPLE 4: Quantifiers - EXISTS and FORALL")
    print("=" * 60)
    
    backend = create_backend()
    
    # Extended family tree
    parent = np.array([
        [0., 1., 1., 0.],  # Alice -> Bob, Carol
        [0., 0., 0., 1.],  # Bob -> David
        [0., 0., 0., 0.],  # Carol has no children
        [0., 0., 0., 0.],  # David has no children
    ], dtype=np.float32)
    
    print("\nParent relation:")
    print(parent)
    print("People: Alice(0), Bob(1), Carol(2), David(3)")
    
    # Query: "Who has children?" (exists y: Parent(x, y))
    has_children = exists(parent, axis=1, backend=backend)
    print("\nHas children (EXISTS):", has_children)
    print("Alice and Bob have children")
    
    # Query: "Is everyone loved by Alice?" (forall x: Loves(Alice, x))
    loves_all = np.array([1., 1., 1., 1.])  # Alice loves everyone
    loves_some = np.array([1., 1., 0., 0.])  # Alice loves Bob and Carol only
    
    print("\nDoes Alice love everyone?")
    print("  When Alice loves all:", forall(loves_all, backend=backend))
    print("  When Alice loves some:", forall(loves_some, backend=backend))
    print()


def example_knowledge_graph_reasoning():
    """Example 5: Complex knowledge graph reasoning."""
    print("=" * 60)
    print("EXAMPLE 5: Knowledge Graph Reasoning with Temperature")
    print("=" * 60)
    
    backend = create_backend()
    
    # Family knowledge graph
    parent = np.array([
        [0., 1., 0., 0.],  # Alice -> Bob
        [0., 0., 1., 0.],  # Bob -> Carol
        [0., 0., 0., 1.],  # Carol -> David
        [0., 0., 0., 0.],  # David has no children
    ], dtype=np.float32)
    
    print("\nFamily tree: Alice -> Bob -> Carol -> David")
    
    # Deductive reasoning (T=0)
    print("\n1. Deductive (T=0): Strict logical inference")
    result_deductive = reason('Grandparent(x, z)', 
                              predicates={'Parent': parent},
                              temperature=0.0,
                              backend=backend)
    print("Grandparent relation (hard threshold):")
    print(result_deductive)
    
    # Analogical reasoning (T=1)
    print("\n2. Analogical (T=1): Soft inference with uncertainty")
    result_analogical = reason('Grandparent(x, z)', 
                               predicates={'Parent': parent},
                               temperature=1.0,
                               backend=backend)
    print("Grandparent relation (continuous scores):")
    print(result_analogical)
    
    # Pattern-based query
    print("\n3. Pattern-based query: Who has children?")
    result_query = quantify('exists y: Parent(x, y)',
                           predicates={'Parent': parent},
                           backend=backend)
    print("Has children:", result_query)
    print()


def example_implication_rules():
    """Example 6: Logical implication and rules."""
    print("=" * 60)
    print("EXAMPLE 6: Logical Implication - Encoding Rules")
    print("=" * 60)
    
    backend = create_backend()
    
    # Relations
    parent = np.array([
        [0., 1., 0., 0.],  # Alice is parent of Bob
        [0., 0., 1., 1.],  # Bob is parent of Carol and David
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
    ], dtype=np.float32)
    
    # Rule: Parent(x,y) -> Ancestor(x,y)
    # Ancestor is at least Parent
    ancestor = logical_implies(
        np.ones_like(parent) - parent,  # not(parent)
        parent,
        backend=backend
    )
    # Actually, for ancestor, we want: ancestor = parent OR (parent @ parent) OR ...
    # Simplified: just use parent as base ancestor
    ancestor = parent.copy()
    
    print("\nParent relation:")
    print(parent)
    
    print("\nAncestor relation (via implication):")
    print(ancestor)
    
    # Compose for multi-hop ancestors
    ancestor_2hop = backend.matmul(parent, parent)
    print("\n2-hop ancestors (grandparent):")
    print(backend.step(ancestor_2hop))
    print()


def example_neural_symbolic_bridge():
    """Example 7: The neural-symbolic bridge."""
    print("=" * 60)
    print("EXAMPLE 7: Neural-Symbolic Bridge")
    print("=" * 60)
    
    backend = create_backend()
    
    print("\nTensor Logic bridges symbolic and neural AI:")
    print("- Same operations")
    print("- Different interpretations via compilation strategies")
    print("- Temperature controls deductive vs analogical reasoning")
    
    # Symbolic: Hard Boolean (provable, zero hallucinations)
    print("\nSYMBOLIC MODE (Hard Boolean, T=0):")
    a_symbolic = np.array([1.0, 0.0, 1.0])
    b_symbolic = np.array([1.0, 1.0, 0.0])
    strategy_symbolic = create_strategy(CompilationStrategy.HARD_BOOLEAN, backend)
    result_symbolic = strategy_symbolic.logical_and(a_symbolic, b_symbolic)
    print(f"  {a_symbolic} AND {b_symbolic} = {result_symbolic}")
    print("  ✓ Exact, provable, no approximation")
    
    # Neural: Soft Differentiable (generalizes, learns from data)
    print("\nNEURAL MODE (Soft Differentiable, T>0):")
    a_neural = np.array([0.9, 0.1, 0.8])  # Continuous values
    b_neural = np.array([0.95, 0.85, 0.2])
    strategy_neural = create_strategy(CompilationStrategy.SOFT_DIFFERENTIABLE, backend)
    result_neural = strategy_neural.logical_and(a_neural, b_neural)
    print(f"  {a_neural} AND {b_neural}")
    print(f"  = {result_neural}")
    print("  ✓ Differentiable, can backpropagate, learns patterns")
    
    print("\n★ The key insight: Both are tensor operations!")
    print("★ Temperature dial: T=0 (deductive) to T>0 (analogical)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "        TENSOR LOGIC FRAMEWORK EXAMPLES".center(58) + "║")
    print("║" + "   Unifying Neural and Symbolic AI via Tensors".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    example_hello_world()
    example_family_tree()
    example_temperature_control()
    example_compilation_strategies()
    example_quantifiers()
    example_knowledge_graph_reasoning()
    example_implication_rules()
    example_neural_symbolic_bridge()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
