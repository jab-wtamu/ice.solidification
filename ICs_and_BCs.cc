// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                                            [[maybe_unused]] const unsigned int index,
                                            [[maybe_unused]] double            &scalar_IC,
                                            [[maybe_unused]] Vector<double>    &vector_IC)
{
  // ---------------------------------------------------------------------
  // ENTER THE INITIAL CONDITIONS HERE
  // ---------------------------------------------------------------------
  // Initial condition:
  //   u   = u0 everywhere
  //   phi = one seed defined by a hyperbolic tangent profile
  //   xi1 = 0 everywhere initially

  double center[1][3] = {
    {0.5, 0.5, 0.5}
  };
  double rad[1] = {5.0};
  double dist;

  scalar_IC = 0.0;

  // Initial condition for the supersaturation field u
  if (index == 0)
    {
      scalar_IC = u0;
    }

  // Initial condition for the order parameter field phi
  else if (index == 1)
    {
      for (unsigned int i = 0; i < 1; i++)
        {
          dist = 0.0;
          for (unsigned int dir = 0; dir < dim; dir++)
            {
              dist += (p[dir] - center[i][dir] * userInputs.domain_size[dir]) *
                      (p[dir] - center[i][dir] * userInputs.domain_size[dir]);
            }
          dist = std::sqrt(dist);

          scalar_IC += (-std::tanh((dist - rad[i]) / std::sqrt(2.0)));
        }
    }

  // Initial condition for the auxiliary field xi1
  else if (index == 2)
    {
      scalar_IC = 0.0;
    }

  // --------------------------------------------------------------------------
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setNonUniformDirichletBCs(
  [[maybe_unused]] const Point<dim>  &p,
  [[maybe_unused]] const unsigned int index,
  [[maybe_unused]] const unsigned int direction,
  [[maybe_unused]] const double       time,
  [[maybe_unused]] double            &scalar_BC,
  [[maybe_unused]] Vector<double>    &vector_BC)
{
  // --------------------------------------------------------------------------
  // ENTER THE NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS HERE
  // --------------------------------------------------------------------------
  // This function is intentionally left blank because the parameter file uses
  // NATURAL boundary conditions for u, phi, and xi1.

  // --------------------------------------------------------------------------
}
