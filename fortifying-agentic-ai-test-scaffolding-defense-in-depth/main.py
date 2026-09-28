"""main.py
Entry point for Fortifying Agentic AI test scaffolding demonstration.
"""

from test_runner import (
    demo_layer_1_foundation,
    demo_layer_2_trajectory,
    demo_layer_3_semantic,
    demo_layer_4_perimeter,
)


def main():
    print("Executing Fortifying Agentic AI Defense-in-Depth Demonstration...")
    demo_layer_1_foundation()
    demo_layer_2_trajectory()
    demo_layer_3_semantic()
    demo_layer_4_perimeter()
    print("\n[SUCCESS] All defense layers evaluated and verified.")


if __name__ == "__main__":
    main()
