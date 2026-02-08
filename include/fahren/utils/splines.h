/*
 * FAHREN Spline Basis Functions
 * 
 * B-spline and other basis function implementations for KAN layers.
 */

#ifndef FAHREN_SPLINES_H
#define FAHREN_SPLINES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate B-spline basis function
 * 
 * Parameters:
 *  - degree: B-spline degree (1=linear, 2=quadratic, 3=cubic)
 *  - index: basis function index
 *  - x: input value in [0, 1]
 *  - knots: knot vector
 *  - num_knots: number of knots
 * Returns: basis function value
 */
float fahren_bspline_basis(int degree, int index, float x,
                          const float* knots, int num_knots);

/* Evaluate derivative of B-spline basis function */
float fahren_bspline_basis_derivative(int degree, int index, float x,
                                     const float* knots, int num_knots);

/* Generate uniform knot vector for B-splines
 * 
 * Parameters:
 *  - degree: B-spline degree
 *  - num_basis: number of basis functions
 * Returns: allocated knot vector (must be freed by caller)
 */
float* fahren_bspline_knots_uniform(int degree, int num_basis);

/* Evaluate spline function (linear combination of basis functions)
 * 
 * Parameters:
 *  - degree: B-spline degree
 *  - x: input value in [0, 1]
 *  - coeffs: spline coefficients
 *  - num_coeffs: number of coefficients (basis functions)
 *  - knots: knot vector
 * Returns: spline value
 */
float fahren_spline_evaluate(int degree, float x, const float* coeffs,
                            int num_coeffs, const float* knots);

/* Evaluate spline derivative */
float fahren_spline_evaluate_derivative(int degree, float x, const float* coeffs,
                                       int num_coeffs, const float* knots);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_SPLINES_H */
