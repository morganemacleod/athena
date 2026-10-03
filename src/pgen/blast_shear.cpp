//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//!        cylindrical, and spherical coordinates.  Contains post-processing code
//!        to check whether blast is spherical for regression tests
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


void Mesh::InitUserMeshData(ParameterInput *pin) {
  //Real omega0 = pin->GetOrAddReal("orbital_advection", "Omega0",1.0);
  //std::cout << "DEBUG: Omega0 is succesfully loaded as: " << omega0 << std::flush;
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rout = pin->GetReal("problem", "radius");
  Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real gamma = peos->GetGamma();

  // shearing box parameters
  Real omega0 = pin->GetReal("orbital_advection", "Omega0");
  Real qshear = pin->GetReal("orbital_advection", "qshear");

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
	Real x = pcoord->x1v(i);
	Real y = pcoord->x2v(j);
	Real z = pcoord->x3v(k);
	rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        
        Real den = da;
        if (rad < rout) {
          if (rad < rin) {
            den = drat*da;
          } else {   // add smooth ramp in density
            Real f = (rad-rin) / (rout-rin);
            Real log_den = (1.0-f) * std::log(drat*da) + f * std::log(da);
            den = std::exp(log_den);
          }
        }

	Real v2 = -qshear * omega0 * x;
	
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = den * v2;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          Real pres = pa;
          if (rad < rout) {
            if (rad < rin) {
              pres = prat*pa;
            } else {  // add smooth ramp in pressure
              Real f = (rad-rin) / (rout-rin);
              Real log_pres = (1.0-f) * std::log(prat*pa) + f * std::log(pa);
              pres = std::exp(log_pres);
            }
          }
          phydro->u(IEN,k,j,i) = pres/(gamma-1);
	}
      }
    }
  }

}
