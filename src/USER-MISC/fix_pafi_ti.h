/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(pafi/ti,FixPAFITI)

#else

#ifndef LMP_FIX_PAFI_TI_H
#define LMP_FIX_PAFI_TI_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPAFITI : public Fix {
 public:
  FixPAFITI(class LAMMPS *, int, char **);
  virtual ~FixPAFITI();
  int setmask();
  virtual void init();

  void setup(int);
  void min_setup(int);
  virtual void post_force(int);
  virtual void pre_force(int);

  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_vector(int);
  // nve
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual void initial_integrate_respa(int, int, int);
  virtual void final_integrate_respa(int, int);
  virtual void reset_dt();

  double memory_usage();

 protected:
  int varflag,icompute;
  char *computename;
  class Compute *PathCompute;
  double ***scale;
  double proj[6], proj_all[6]; // f,v,h, psi
  double results[5], results_all[5]; // f.n, (f.n)**2, psi, dx.n
  double lambda,c_v[10],c_v_all[10];
  double temperature,gamma,sqrtD,t_period,local_norm,mass_f;
  int force_flag,od_flag,com_flag,ref_pair;
  int nlevels_respa,ilevel_respa;
  int maxatom;
  class RanMars *random;
  int seed;
  double **h;
  // nve
  double dtv,dtf;
  double *step_respa;
  int mass_require;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute ID for fix pafiTI does not exist

Self-explanatory.

E: Compute for fix pafiTI does not calculate a local array

Self-explanatory.

E: Compute for fix pafiTI has < 3*DIM fields per atom

Self-explanatory.

E: Fix pafiTI requires a damped dynamics minimizer

Use a different minimization style.

*/
