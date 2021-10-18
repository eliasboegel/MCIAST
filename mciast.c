#include <stddef.h>
#include <math.h>

// 10/2021, Elias Boegel
// This function calculates the spreading pressure based on isotherms modelled as any number of langmuir terms
// In: partial_p: Partial pressure of the component
// In: coeffs[]: Langmuir term coefficients, each langmuir term has 2 coefficients that are directly one after another in the array. Example: Ap/(1+Bp)+Cp/(Dp+1) -> coeffs[]: {A, B, C, D}
// In: n_langmuir_terms: Example: Ap/(1+Bp)+Cp/(Dp+1) -> 2 langmuir terms -> n_langmuir_terms=2
// Out: spreading_p: Spreading pressure 
double spreading_pressure_langmuir(double partial_p, double coeffs[], size_t n_langmuir_terms)
{
	// Spreading pressure starting point for integral sum
	double spreading_p = 0.0;

	// Form sum of analytically solved integrals with lower bound 0 and upper bound partial_p, one iteration/addition per langmuir term being integrated
	for (size_t i = 0; i<n_langmuir_terms; i++)
	{
		spreading_p += coeffs[i] / coeffs[i+1] * log(coeffs[i+1] * partial_p + 1);
	}

	return spreading_p;
}
