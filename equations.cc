// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
//u residuals → explicitEquationRHS
//phi residual → explicitEquationRHS
//xi1 residuals → nonExplicitEquationRHS

void
customAttributeLoader::loadVariableAttributes()
{
  // ---------------------------------------------------------------------------
  // Variable 0: u
  // ---------------------------------------------------------------------------
  // This is the supersaturation / diffusion field.
  //
  // In the derived weak form:
  //
  //   r_u   = u^n - dt * (Lsat/2) * (B_n / A_n^2) * xi1^n
  //   r_u_x = -dt * F2^n
  //
  // So for the value residual, u depends on:
  //   - u itself
  //   - xi1
  //   - phi
  //   - grad(phi)
  //   - grad(u)
  //
  // And for the gradient residual, u depends on:
  //   - phi
  //   - grad(u)
  //
  // This keeps the field wiring aligned with the intended weak-form structure.
  set_variable_name(0, "u");
  set_variable_type(0, SCALAR);
  set_variable_equation_type(0, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(0, "u,xi1,phi,grad(phi),grad(u)");
  set_dependencies_gradient_term_RHS(0, "phi,grad(u)");

  // ---------------------------------------------------------------------------
  // Variable 1: phi
  // ---------------------------------------------------------------------------
  // This is the phase-field / order parameter.
  //
  // The explicit residual used for phi is:
  //
  //   r_phi = phi^n + dt * xi1^n / A_n^2
  //
  // So phi depends on:
  //   - phi
  //   - xi1
  //   - grad(phi)
  //
  // There is no separate gradient residual term for phi in this explicit update.
  set_variable_name(1, "phi");
  set_variable_type(1, SCALAR);
  set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(1, "phi,xi1,grad(phi)");
  set_dependencies_gradient_term_RHS(1, "");

  // ---------------------------------------------------------------------------
  // Variable 2: xi1
  // ---------------------------------------------------------------------------
  // This is the auxiliary field used to hold the phase-equation RHS structure.
  //
  // The intended residuals are:
  //
  //   r_xi   = f1^n
  //   r_xi_x = -F1^n
  //
  // So xi1 is treated as an AUXILIARY variable rather than an explicitly
  // time-advanced primary evolution field.
  set_variable_name(2, "xi1");
  set_variable_type(2, SCALAR);
  set_variable_equation_type(2, AUXILIARY);

  set_dependencies_value_term_RHS(2, "phi,u,grad(phi)");
  set_dependencies_gradient_term_RHS(2, "grad(phi)");
}

// =============================================================================================
// explicitEquationRHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::explicitEquationRHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // ---------------------------------------------------------------------------
  // This implementation is intentionally restricted to 3D.
  //
  // The current snow-crystal model uses the derived 3D formulas for:
  //   - the interface normal n
  //   - theta
  //   - psi
  //   - A(n)
  //   - B(n)
  //
  //
  // IMPORTANT:
  // The main application still instantiates both dim=2 and dim=3 template
  // versions during compilation, so this function must remain compile-safe for
  // dim=2 as well. Therefore, the actual 3D snow physics is placed inside
  // if constexpr (dim == 3), while a compile-only fallback is provided for
  // dim=2.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Get the values and gradients of the model variables at the quadrature point
  // ---------------------------------------------------------------------------

  // u and grad(u)
  scalarvalueType u  = variable_list.get_scalar_value(0);
  scalargradType  ux = variable_list.get_scalar_gradient(0);

  // phi and grad(phi)
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // xi1
  scalarvalueType xi1 = variable_list.get_scalar_value(2);

  // ---------------------------------------------------------------------------
  // Interface geometry: build the unit normal vector
  //
  // n = -grad(phi) / |grad(phi)|
  //
  // regval is used to prevent division by zero near vanishing gradients.
  // ---------------------------------------------------------------------------

  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = (-phix) / (normgradn + constV(regval));

  scalarvalueType A2_n;
  scalarvalueType B_n;
  scalargradType  F2;

  if constexpr (dim == 3)
    {
      scalarvalueType nx = normal[0];
      scalarvalueType ny = normal[1];
      scalarvalueType nz = normal[2];

      // ---------------------------------------------------------------------------
      // Convert the interface normal into the angular representation used in the
      // derived anisotropy model:
      //
      //   theta = atan2(ny, nx)
      //   psi   = atan2(sqrt(nx^2 + ny^2), -nz)
      //
      // theta controls the in-plane sixfold anisotropy,
      // psi controls the vertical anisotropy contribution.
      //
      // NOTE:
      // theta and psi are computed lane-by-lane because scalarvalueType is a
      // VectorizedArray and std::atan2 cannot act on it directly.
      // ---------------------------------------------------------------------------

      scalarvalueType rho_n = std::sqrt(nx * nx + ny * ny);

      scalarvalueType theta;
      scalarvalueType psi;
      for (unsigned int i = 0; i < theta.size(); ++i)
        {
          theta[i] = std::atan2(ny[i], nx[i]);
          psi[i]   = std::atan2(rho_n[i], -nz[i]);
        }

      // ---------------------------------------------------------------------------
      // Model functions
      // ---------------------------------------------------------------------------

      // A(n) = 1 + eps_xy*cos(6*theta) + eps_z*cos(2*psi)
      //
      // This is the anisotropy factor that multiplies the phase evolution.
      scalarvalueType A_n =
        constV(1.0) +
        constV(eps_xy) * std::cos(constV(6.0) * theta) +
        constV(eps_z)  * std::cos(constV(2.0) * psi);

      // A(n)^2 appears in the explicit time-discretized phase update
      // A small floor is added for numerical stabilization.
      A2_n = A_n * A_n + constV(1.0e-8);

      // B(n) = sqrt(nx^2 + ny^2 + Gamma^2 * nz^2)
      //
      // This is the anisotropy factor used in the kinetic/coupling term.
      B_n =
        std::sqrt(nx * nx + ny * ny + constV(Gamma) * constV(Gamma) * nz * nz);

      // q(phi) used in the supersaturation diffusion flux F2
      scalarvalueType q_phi = constV(1.0) - phi;

      // ---------------------------------------------------------------------------
      // F2 = D_tilde * Gamma^T * q(phi) * Gamma * grad(u)
      //
      // In this 3D implementation, Gamma = diag(1, 1, Gamma).
      // Therefore:
      //   x-component: unchanged
      //   y-component: unchanged
      //   z-component: scaled by Gamma^2
      // ---------------------------------------------------------------------------

      F2[0] = constV(D_tilde) * q_phi * ux[0];
      F2[1] = constV(D_tilde) * q_phi * ux[1];
      F2[2] = constV(D_tilde) * q_phi * constV(Gamma) * constV(Gamma) * ux[2];
    }
  else
    {
      // ---------------------------------------------------------------------------
      // Compile-only fallback for dim = 2
      //
      // This branch is not the intended snow-crystal physics. It exists only so
      // that the dim=2 templates can still compile, because the main application
      // instantiates both 2D and 3D versions.
      // ---------------------------------------------------------------------------

      scalarvalueType q_phi = constV(1.0) - phi;

      A2_n = constV(1.0);
      B_n  = constV(1.0);
      F2   = constV(D_tilde) * q_phi * ux;
    }

  // ---------------------------------------------------------------------------
  // Explicit residual terms
  // ---------------------------------------------------------------------------

  // r_phi = phi^n + dt * xi1^n / A_n^2
  //
  // This advances phi explicitly using xi1 as the already-computed auxiliary RHS.
  scalarvalueType eq_phi =
    phi + constV(userInputs.dtValue) * xi1 / A2_n;

  // r_u = u^n - dt * (Lsat/2) * (B_n / A_n^2) * xi1^n
  //
  // This is the scalar/value residual part of the u equation.
  scalarvalueType eq_u =
    u - constV(userInputs.dtValue) * constV(Lsat / 2.0) * (B_n / A2_n) * xi1;

  // r_u_x = -dt * F2^n
  //
  // This is the gradient/flux residual part of the u equation.
  scalargradType eqx_u =
    constV(-1.0) * constV(userInputs.dtValue) * F2;

  // ---------------------------------------------------------------------------
  // Submit the explicit RHS terms
  // ---------------------------------------------------------------------------

  // Terms for the equation to evolve u
  variable_list.set_scalar_value_term_RHS(0, eq_u);
  variable_list.set_scalar_gradient_term_RHS(0, eqx_u);

  // Terms for the equation to evolve phi
  variable_list.set_scalar_value_term_RHS(1, eq_phi);
}

// =============================================================================================
// nonExplicitEquationRHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::nonExplicitEquationRHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // ---------------------------------------------------------------------------
  // This auxiliary-field implementation is also restricted to 3D for the same
  // reason as the explicitEquationRHS: the anisotropy formulas are written in
  // terms of 3D interface angles theta and psi.
  //
  // IMPORTANT:
  // The main application still instantiates both dim=2 and dim=3 template
  // versions during compilation, so this function must remain compile-safe for
  // dim=2 as well. Therefore, the actual 3D snow physics is placed inside
  // if constexpr (dim == 3), while a compile-only fallback is provided for
  // dim=2.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Get the values and derivatives of the model variables
  // ---------------------------------------------------------------------------

  // supersaturation
  scalarvalueType u = variable_list.get_scalar_value(0);

  // phase field and its gradient
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // ---------------------------------------------------------------------------
  // Build the interface normal and spherical-angle representation again
  // ---------------------------------------------------------------------------

  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = (-phix) / (normgradn + constV(regval));

  scalarvalueType A2_n;
  scalarvalueType B_n;
  scalargradType  F1;

  if constexpr (dim == 3)
    {
      scalarvalueType nx = normal[0];
      scalarvalueType ny = normal[1];
      scalarvalueType nz = normal[2];

      scalarvalueType rho_n = std::sqrt(nx * nx + ny * ny);

      scalarvalueType theta;
      scalarvalueType psi;

      for (unsigned int i = 0; i < theta.size(); ++i)
        {
          theta[i] = std::atan2(ny[i], nx[i]);
          psi[i]   = std::atan2(rho_n[i], -nz[i]);
        }

      // ---------------------------------------------------------------------------
      // Model functions
      // ---------------------------------------------------------------------------

      // A(n) and A(n)^2
      scalarvalueType A_n =
        constV(1.0) +
        constV(eps_xy) * std::cos(constV(6.0) * theta) +
        constV(eps_z)  * std::cos(constV(2.0) * psi);

      A2_n = A_n * A_n + constV(1.0e-8);

      // B(n)
      B_n =
        std::sqrt(nx * nx + ny * ny + constV(Gamma) * constV(Gamma) * nz * nz);

      // ---------------------------------------------------------------------------
      // Local scalar term f1(phi, grad(phi), u)
      
      // The derived scalar residual is:
      //
      //   r_xi = f1^n = -f'(phi) + lambda * B(n) * g'(phi) * u
      //
      // Here:
      //   -f'(phi) = phi - phi^3
      //   g'(phi)  = (1 - phi^2)^2
      // ---------------------------------------------------------------------------

      // -----------------------------------------------------------------------------
      // d(A^2)/d(grad(phi)) via chain rule
      //
      // From derived Eqs:
      //
      //   d(A^2)/d(grad(phi))
      //     = d(A^2)/d(theta) * d(theta)/d(grad(phi))
      //     + d(A^2)/d(psi)   * d(psi)/d(grad(phi))
      //
      // where
      //   A = 1 + eps_xy*cos(6 theta) + eps_z*cos(2 psi)
      // -----------------------------------------------------------------------------

      scalarvalueType dA2_dtheta =
        constV(-12.0 * eps_xy) * A_n * std::sin(constV(6.0) * theta);

      scalarvalueType dA2_dpsi =
        constV(-4.0 * eps_z) * A_n * std::sin(constV(2.0) * psi);

      // gradxy = sqrt(phi_x^2 + phi_y^2)
      scalarvalueType gradxy2 = phix[0] * phix[0] + phix[1] * phix[1];
      scalarvalueType gradxy  = std::sqrt(gradxy2);

      // Regularized denominators to avoid division by zero
      scalarvalueType denom_theta = gradxy2 + constV(regval);
      scalarvalueType denom_psi   = gradxy2 + phix[2] * phix[2] + constV(regval);
      scalarvalueType safe_gradxy = gradxy + constV(regval);

      // dtheta / d(grad(phi)) for theta = atan2(phi_y, phi_x)
      scalargradType dtheta_dgradphi;
      dtheta_dgradphi[0] = -phix[1] / denom_theta;
      dtheta_dgradphi[1] =  phix[0] / denom_theta;
      dtheta_dgradphi[2] =  constV(0.0);

      // dpsi / d(grad(phi)) for psi = atan2(sqrt(phi_x^2 + phi_y^2), -phi_z)
      scalargradType dpsi_dgradphi;
      dpsi_dgradphi[0] = (-phix[2] * phix[0]) / (safe_gradxy * denom_psi);
      dpsi_dgradphi[1] = (-phix[2] * phix[1]) / (safe_gradxy * denom_psi);
      dpsi_dgradphi[2] = gradxy / denom_psi;

      // Full chain-rule derivative
      scalargradType dA2_dgradphi =
        dA2_dtheta * dtheta_dgradphi + dA2_dpsi * dpsi_dgradphi;

      // -----------------------------------------------------------------------------
      // F1 = (Gamma^T / 2) * ( |grad(phi)|^2 d(A^2)/d(grad(phi)) + A^2 Gamma grad(phi) )
      //
      // Gamma = diag(1,1,Gamma)
      // -----------------------------------------------------------------------------

      // |grad(phi)|^2
      scalarvalueType gradphi2 = phix.norm_square();

      // Gamma * grad(phi)
      scalargradType Gamma_gradphi;
      Gamma_gradphi[0] = phix[0];
      Gamma_gradphi[1] = phix[1];
      Gamma_gradphi[2] = constV(Gamma) * phix[2];

      // Inner quantity:
      // |grad(phi)|^2 d(A^2)/d(grad(phi)) + A^2 Gamma grad(phi)
      scalargradType inside_F1;
      inside_F1[0] = gradphi2 * dA2_dgradphi[0] + A2_n * Gamma_gradphi[0];
      inside_F1[1] = gradphi2 * dA2_dgradphi[1] + A2_n * Gamma_gradphi[1];
      inside_F1[2] = gradphi2 * dA2_dgradphi[2] + A2_n * Gamma_gradphi[2];

      // Apply Gamma^T / 2
      F1[0] = constV(0.5) * inside_F1[0];
      F1[1] = constV(0.5) * inside_F1[1];
      F1[2] = constV(0.5 * Gamma) * inside_F1[2];
    }
  else
    {
      // ---------------------------------------------------------------------------
      // Compile-only fallback for dim = 2
      //
      // This branch is not the intended snow-crystal physics. It exists only so
      // that the dim=2 templates can still compile, because the main application
      // instantiates both 2D and 3D versions.
      // ---------------------------------------------------------------------------

      A2_n = constV(1.0);
      B_n  = constV(1.0);
      F1   = constV(0.0) * phix;
    }

  // ---------------------------------------------------------------------------
  // Local scalar term f1(phi, grad(phi), u)
  //
  //
  //   r_xi = f1^n = -f'(phi) + lambda * B(n) * g'(phi) * u
  //
  // Here:
  //   -f'(phi) = phi - phi^3
  //   g'(phi)  = (1 - phi^2)^2
  // ---------------------------------------------------------------------------

  scalarvalueType minus_fprime = phi - phi * phi * phi;

  scalarvalueType gprime =
    (constV(1.0) - phi * phi) * (constV(1.0) - phi * phi);

  scalarvalueType eq_xi1 =
    minus_fprime + constV(lambda) * B_n * gprime * u;

  // r_xi_x = -F1
  scalargradType eqx_xi1 = -F1;

  // ---------------------------------------------------------------------------
  // Submit the auxiliary residual terms
  // ---------------------------------------------------------------------------

  variable_list.set_scalar_value_term_RHS(2, eq_xi1);
  variable_list.set_scalar_gradient_term_RHS(2, eqx_xi1);
}

// =============================================================================================
// equationLHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::equationLHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // No separate time-independent left-hand-side terms are added here.
  // This function remains empty for the current explicit/auxiliary formulation.
}
