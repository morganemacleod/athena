//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file planet_wind_lambda.cpp: tidal perturbation of planet wind defined by hydrodynamic escape parameter lambda
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
//#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../utils/utils.hpp"



Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length);



//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres;
  Real gamma_gas;
  Real GM;

  const int NARRAY = 10000;
  Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc_init[NARRAY];  // initial profile

  // read in profile arrays from file
  std::ifstream infile("hse_profile.dat");
  if (infile.is_open()){
      for(int i=0;i<NARRAY;i++){
	infile >> rad[i] >> rho[i] >> p[i] >> menc_init[i];
	if (i%1000==0){ 
	  std:: cout <<i <<"   "<< rad[i] << "    " << rho[i] << std::endl;
	}
      }
      infile.close();
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR, hse_profile.dat not found in ProblemGenerator" << std::endl;        
    ATHENA_ERROR(msg);
  }
    

  gamma_gas = pin->GetReal("hydro","gamma");
  GM = pin->GetReal("problem","GM");
  Real pfactor = pin->GetOrAddReal("problem","pfactor",1.0);

  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

	Real r  = pcoord->x1v(i);
	den  = Interpolate1DArrayEven(rad, rho, r , NARRAY);
	pres = Interpolate1DArrayEven(rad,   p, r , NARRAY);

	// multiply the pressure by factor inside radius
	if(r<1.1){
	  pres *= pfactor;
	}
	
	phydro->u(IDN,k,j,i) = den;
	phydro->u(IM1,k,j,i) = 0.0;
	phydro->u(IM2,k,j,i) = 0.0;
	phydro->u(IM3,k,j,i) = 0.0;
	phydro->u(IEN,k,j,i) = pres/(gamma_gas-1.0);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator





// 1D Interpolation that assumes EVEN spacing in x array
Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length){ 
  // check the lower bound
  if(x[0] >= x0){
    //std::cout << "hit lower bound!\n";
    return y[0];
  }
  // check the upper bound
  if(x[length-1] <= x0){
    //std::cout << "hit upper bound!\n";
    return y[length-1];
  }

  int i = floor( (x0-x[0])/(x[1]-x[0]) );
  
  // if in the interior, do a linear interpolation
  if (x[i+1] >= x0){ 
    Real dx =  (x[i+1]-x[i]);
    Real d = (x0 - x[i]);
    Real s = (y[i+1]-y[i]) /dx;
    return s*d + y[i];
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}
