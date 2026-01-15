#!/usr/bin/env python3
"""
Verify that analytical forcing derivatives match numerical finite differences.

For the steady advection equation: c * du/dx = f
So: du/dx = f / c

This script compares:
  - Numerical: du/dx ≈ (u(x+h) - u(x-h)) / (2h)  via exact_solution()
  - Analytical: du/dx = f(x) / c                  via eff()

Run from project root:
    python analysis/verification/verify_eff_derivatives.py
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from numerical.solvers.utils import exact_solution, eff


def verify_derivative_at_point(icase, x, t, h=1e-6):
    """
    Compare analytical and numerical derivatives at a single point.
    
    Returns:
        tuple: (du_dx_numerical, du_dx_analytical, relative_error)
    """
    # Numerical derivative via central difference
    u_plus, c = exact_solution(np.array([x + h]), 1, t, icase)
    u_minus, _ = exact_solution(np.array([x - h]), 1, t, icase)
    du_dx_numerical = (u_plus[0] - u_minus[0]) / (2 * h)
    
    # Analytical derivative from forcing function: f = c * du/dx
    f = eff(np.array([x]), 1, icase, c, t)
    du_dx_analytical = f[0] / c
    
    # Relative error (with safety for near-zero derivatives)
    denom = max(abs(du_dx_numerical), abs(du_dx_analytical), 1e-10)
    rel_error = abs(du_dx_numerical - du_dx_analytical) / denom
    
    return du_dx_numerical, du_dx_analytical, rel_error


def verify_icase(icase, test_points=None, test_times=None, tolerance=1e-4, verbose=True):
    """
    Verify derivatives for a single icase across multiple points and times.
    
    Returns:
        tuple: (passed, max_error, worst_point_info)
    """
    if test_points is None:
        # Test across domain [-1, 1], avoiding exact boundaries
        test_points = np.linspace(-0.95, 0.95, 21)
    
    if test_times is None:
        test_times = [0.0, 0.1, 0.25]
    
    max_error = 0.0
    worst_info = None
    all_errors = []
    
    for t in test_times:
        for x in test_points:
            try:
                du_num, du_ana, rel_err = verify_derivative_at_point(icase, x, t)
                all_errors.append(rel_err)
                
                if rel_err > max_error:
                    max_error = rel_err
                    worst_info = {
                        'x': x,
                        't': t,
                        'du_dx_numerical': du_num,
                        'du_dx_analytical': du_ana,
                        'rel_error': rel_err
                    }
            except Exception as e:
                if verbose:
                    print(f"  ERROR at x={x:.3f}, t={t:.2f}: {e}")
                return False, float('inf'), {'error': str(e)}
    
    passed = max_error < tolerance
    
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  icase {icase:2d}: {status}  max_rel_error = {max_error:.2e}  "
              f"(mean = {np.mean(all_errors):.2e})")
        if not passed and worst_info:
            print(f"           Worst at x={worst_info['x']:.3f}, t={worst_info['t']:.2f}")
            print(f"           Numerical: {worst_info['du_dx_numerical']:.6e}")
            print(f"           Analytical: {worst_info['du_dx_analytical']:.6e}")
    
    return passed, max_error, worst_info


def main():
    print("=" * 70)
    print("Verifying eff() Forcing Function Derivatives")
    print("=" * 70)
    print()
    print("Method: Compare f(x)/c from eff() to finite difference of exact_solution()")
    print("Tolerance: 1e-4 relative error")
    print()
    
    # Define test cases
    # Existing cases that should work
    existing_cases = [1, 6, 7, 8]
    
    # New cases we need to verify (10-16)
    new_cases = [10, 11, 12, 13, 14, 15, 16]
    
    # Case descriptions for reference
    case_names = {
        1: "Gaussian pulse",
        6: "Sine wave",
        7: "Tanh profile",
        8: "Sine(pi*x)",
        10: "Tanh smooth square",
        11: "Erf smooth square",
        12: "Sigmoid smooth square",
        13: "Multi-Gaussian",
        14: "Bump function",
        15: "Sech² soliton",
        16: "Mexican hat (Ricker)",
    }
    
    results = {}
    
    # Test existing cases first (sanity check)
    print("-" * 70)
    print("EXISTING CASES (sanity check):")
    print("-" * 70)
    for icase in existing_cases:
        name = case_names.get(icase, "Unknown")
        print(f"\nTesting icase {icase}: {name}")
        passed, max_err, info = verify_icase(icase)
        results[icase] = {'passed': passed, 'max_error': max_err, 'name': name}
    
    # Test new cases
    print()
    print("-" * 70)
    print("NEW CASES (icases 10-16):")
    print("-" * 70)
    for icase in new_cases:
        name = case_names.get(icase, "Unknown")
        print(f"\nTesting icase {icase}: {name}")
        passed, max_err, info = verify_icase(icase)
        results[icase] = {'passed': passed, 'max_error': max_err, 'name': name}
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for icase in existing_cases + new_cases:
        r = results[icase]
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  icase {icase:2d} ({r['name']:25s}): {status}  (max err: {r['max_error']:.2e})")
        if not r['passed']:
            all_passed = False
    
    print()
    if all_passed:
        print("All derivative verifications PASSED!")
        print("The eff() forcing functions are correctly implemented.")
        return 0
    else:
        print("Some verifications FAILED!")
        print("Review the failing cases and fix the eff() implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())