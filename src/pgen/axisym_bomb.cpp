//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file axisym_bomb.cpp
//! \brief 2D axisymmetric (r, theta) spherical-polar thermal bomb inside a remapped
//!        stellar-merger remnant.
//!
//! The envelope comes from a 3D spherical-polar merger run (pm_envelope_cool_merger.cpp),
//! azimuthally averaged (conserved variables) and equatorially symmetrized by
//! make_axisym_2d.py (IaCSM/Analysis). That file holds flat 2D arrays [theta, r] of
//! rho, press, vel1 (v_r), vel2 (v_theta), vel3 (v_phi) on the source's finest lattice,
//! theta in [0, pi/2], plus x1f, x1v, x2f, x2v.
//!
//! Initial state, by radius r (cell center) and outer radial face r+:
//!  - r+ <= x1min_src (source inner radius): thermal bomb. Uniform rho = M_bomb / V,
//!    P = (gamma-1) E_bomb / V, v = 0, where V is the FULL-SPHERE volume of the bomb cells,
//!    so M_bomb and E_bomb are totals for the whole star (the domain may be a wedge).
//!  - r > x1max_src: ambient (rho_amb, p_amb, v = 0).
//!  - otherwise: bilinear interpolation of the primitives in (ln r, theta) from the file;
//!    r and theta are clamped to the source cell-center range.
//!
//! Hydro only, no gravity, ideal gas. Requires --coord=spherical_polar, -hdf5, and a mesh
//! without refinement (the bomb volume is computed from the root-level radial faces).
//! Units: those of the remap file (cgs for the merger runs).
//!
//! Parameters (<problem>): remap_file, M_bomb, E_bomb, rho_amb, p_amb

// C headers
#include <hdf5.h>

// C++ headers
#include <algorithm>  // upper_bound, min, max
#include <cmath>
#include <cstring>    // strcmp
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#ifndef HDF5OUTPUT
#error "axisym_bomb.cpp requires HDF5 (configure with -hdf5)"
#endif

namespace {
// remap data: 1D coordinates and 2D fields stored [j*nr + i] (theta-major, as in the file)
std::vector<double> src_x1f, src_x2f, src_lnr, src_x2v;
std::vector<double> src_rho, src_press, src_v1, src_v2, src_v3;
int src_nr, src_nth;
Real rho_bomb, p_bomb, rho_amb, p_amb;

//! read a whole 1D or 2D double dataset; returns the number of elements along each dim
std::vector<double> ReadDataset(hid_t file, const char *name, hsize_t dims[2]) {
  hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
  if (dset < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: dataset " << name << " not found" << std::endl;
    ATHENA_ERROR(msg);
  }
  hid_t space = H5Dget_space(dset);
  int ndim = H5Sget_simple_extent_ndims(space);
  dims[0] = 1; dims[1] = 1;
  H5Sget_simple_extent_dims(space, dims, nullptr);
  std::vector<double> v(dims[0] * (ndim == 2 ? dims[1] : 1));
  H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Sclose(space);
  H5Dclose(dset);
  return v;
}

//! index i such that x[i] <= xq <= x[i+1], for monotonically increasing x (xq clamped)
int LowerIndex(const std::vector<double> &x, double xq) {
  int i = static_cast<int>(std::upper_bound(x.begin(), x.end(), xq) - x.begin()) - 1;
  return std::max(0, std::min(i, static_cast<int>(x.size()) - 2));
}

//! is this radial cell part of the bomb? (cell entirely inside the source inner radius)
//! The source faces come from float32 .athdf data (relative precision ~6e-8), so a face
//! placed at x1min_src in the input file can sit just above the stored value; the 1e-6
//! tolerance is far below any cell width.
bool InBomb(Real r_outer_face) {
  return r_outer_face <= src_x1f.front() * (1.0 + 1.0e-6);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Read the remap file and compute the bomb density and pressure.

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: requires --coord=spherical_polar" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!NON_BAROTROPIC_EOS || multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: requires an adiabatic EOS and no mesh "
        << "refinement" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string fn = pin->GetString("problem", "remap_file");
  Real M_bomb = pin->GetReal("problem", "M_bomb");
  Real E_bomb = pin->GetReal("problem", "E_bomb");
  rho_amb = pin->GetReal("problem", "rho_amb");
  p_amb = pin->GetReal("problem", "p_amb");
  Real gamma = pin->GetReal("hydro", "gamma");

  // every rank reads the (small) file
  hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: could not open " << fn << std::endl;
    ATHENA_ERROR(msg);
  }
  hsize_t d1[2], d2[2], df[2];
  src_x1f = ReadDataset(file, "x1f", df);
  std::vector<double> x1v = ReadDataset(file, "x1v", d1);
  src_x2f = ReadDataset(file, "x2f", df);
  src_x2v = ReadDataset(file, "x2v", d2);
  src_nr = static_cast<int>(d1[0]);
  src_nth = static_cast<int>(d2[0]);
  src_rho = ReadDataset(file, "rho", df);
  src_press = ReadDataset(file, "press", df);
  src_v1 = ReadDataset(file, "vel1", df);
  src_v2 = ReadDataset(file, "vel2", df);
  src_v3 = ReadDataset(file, "vel3", df);
  H5Fclose(file);
  if (static_cast<int>(df[0]) != src_nth || static_cast<int>(df[1]) != src_nr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: field shape (" << df[0] << "," << df[1]
        << ") does not match coordinates (" << src_nth << "," << src_nr << ")" << std::endl;
    ATHENA_ERROR(msg);
  }
  // interpolate linearly in ln r (source grid is uniform in ln r)
  src_lnr.resize(src_nr);
  for (int i = 0; i < src_nr; ++i) src_lnr[i] = std::log(x1v[i]);

  if (mesh_size.x1min >= src_x1f.front() || mesh_size.x2max > 0.5*PI*(1.0 + 1.0e-10)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: need mesh x1min < source x1min ("
        << src_x1f.front() << ") and x2max <= pi/2 (data cover theta in [0, pi/2])"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // bomb volume: full-sphere volume of the root-level radial cells with r+ <= x1min_src,
  // using the same mesh generator that sets the MeshBlock coordinates
  Real V_bomb = 0.0, r_last = mesh_size.x1min;
  int n_bomb = 0;
  for (int i = 0; i < mesh_size.nx1; ++i) {
    Real rm = MeshGenerator_[X1DIR](
        ComputeMeshGeneratorX(i, mesh_size.nx1, use_uniform_meshgen_fn_[X1DIR]), mesh_size);
    Real rp = MeshGenerator_[X1DIR](
        ComputeMeshGeneratorX(i + 1, mesh_size.nx1, use_uniform_meshgen_fn_[X1DIR]),
        mesh_size);
    if (i == 0) rm = mesh_size.x1min;
    if (InBomb(rp)) {
      V_bomb += (4.0*PI/3.0) * (rp*rp*rp - rm*rm*rm);
      r_last = rp;
      ++n_bomb;
    }
  }
  if (n_bomb == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in axisym_bomb: no radial cells inside source x1min"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  rho_bomb = M_bomb / V_bomb;
  p_bomb = (gamma - 1.0) * E_bomb / V_bomb;

  if (Globals::my_rank == 0) {
    std::cout << "axisym_bomb: remap file " << fn << " (" << src_nth << " x " << src_nr
              << ", r in [" << src_x1f.front() << ", " << src_x1f.back() << "])\n"
              << "  bomb: " << n_bomb << " radial cells, r in [" << mesh_size.x1min
              << ", " << r_last << "], gap to source x1min = "
              << src_x1f.front() - r_last << "\n"
              << "  full-sphere V = " << V_bomb << ", rho = " << rho_bomb
              << ", P = " << p_bomb
              << ", c_s = " << std::sqrt(gamma * p_bomb / rho_bomb) << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Set the bomb, the interpolated envelope, and the ambient medium.

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real x1max_src = src_x1f.back();

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      Real th = pcoord->x2v(j);
      // theta weights (clamped to the source cell-center range)
      double thc = std::max(src_x2v.front(), std::min(static_cast<double>(th),
                                                      src_x2v.back()));
      int j0 = LowerIndex(src_x2v, thc);
      double wt = (thc - src_x2v[j0]) / (src_x2v[j0+1] - src_x2v[j0]);
      for (int i = is; i <= ie; ++i) {
        Real r = pcoord->x1v(i);
        Real rho, press, v1, v2, v3;
        if (InBomb(pcoord->x1f(i+1))) {
          rho = rho_bomb; press = p_bomb; v1 = 0.0; v2 = 0.0; v3 = 0.0;
        } else if (r > x1max_src) {
          rho = rho_amb; press = p_amb; v1 = 0.0; v2 = 0.0; v3 = 0.0;
        } else {
          double lnr = std::max(src_lnr.front(), std::min(std::log(r), src_lnr.back()));
          int i0 = LowerIndex(src_lnr, lnr);
          double wr = (lnr - src_lnr[i0]) / (src_lnr[i0+1] - src_lnr[i0]);
          // bilinear weights of the 4 corners [j0, i0], [j0, i0+1], [j0+1, i0], [j0+1, i0+1]
          int n00 = j0*src_nr + i0, n01 = n00 + 1, n10 = n00 + src_nr, n11 = n10 + 1;
          double w00 = (1.0-wt)*(1.0-wr), w01 = (1.0-wt)*wr;
          double w10 = wt*(1.0-wr), w11 = wt*wr;
          auto interp = [&](const std::vector<double> &q) {
            return static_cast<Real>(w00*q[n00] + w01*q[n01] + w10*q[n10] + w11*q[n11]);
          };
          rho = interp(src_rho); press = interp(src_press);
          v1 = interp(src_v1); v2 = interp(src_v2); v3 = interp(src_v3);
        }
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho*v1;
        phydro->u(IM2,k,j,i) = rho*v2;
        phydro->u(IM3,k,j,i) = rho*v3;
        phydro->u(IEN,k,j,i) = press/gm1 + 0.5*rho*(v1*v1 + v2*v2 + v3*v3);
      }
    }
  }
  return;
}
