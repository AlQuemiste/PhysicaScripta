module MolecularDynamics
  use, intrinsic :: iso_c_binding, only: c_double, c_int
  implicit none

  ! double precision kind parameter
  integer, parameter :: f64 = selected_real_kind(c_double)
  integer, parameter :: i32 = selected_int_kind(c_int)

  ! assumed to be defined elsewhere in the module
  real(kind=f64), parameter :: EPS = 1.0E-10
  integer(kind=i32) :: n_particles  ! total number of particles
  real(kind=f64) :: T  ! temperature

  type, bind(C) :: Parameters
     integer(i32) :: thermalization  ! nr of warm-up steps
     integer(i32) :: steps           ! nr of MC steps
     integer(i32) :: j               ! J / |J| = +-1, the sign of the spin coupling, J
     ! +: ferromagnetic coupling, -: antiferromagnetic coupling
     real(f64) :: t               ! T / |J|, temperature in units of J
     real(f64) :: beta            ! 1 / (T/|J|), inverse temperature in units of J
     real(f64) :: h               ! h / |J|, external magnetic field in units of J
     integer(i32) :: L               ! linear dimension of the lattice
  end type Parameters

contains

  subroutine step(dt, prm, fixedParticles, boundParticles, state, stat)
    real(f64), intent(in) :: dt
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:), intent(in) :: fixedParticles
    real(f64), dimension(:, 2), intent(in) :: boundParticles
    real(f64), dimension(prm%n_conf, :), intent(inout), target :: state
    type(Statistics), intent(out) :: stat

    integer, parameter :: n_particles = size(state, 2)
    real(f64), dimension(n_particles), pointer :: X, Y, VX, VY, AX, AY
    real(f64) :: half_dt, half_dt2
    integer :: iP

    !! velocity Verlet algorithm

    ! configurations
    X  => state(prm%i_x, :)
    Y  => state(prm%i_y, :)
    ! TH => state(prm%i_TH, :)
    VX => state(prm%i_vx, :)
    VY => state(prm%i_vy, :)
    ! VTH => state(prm%i_vth, :)
    AX => state(prm%i_ax, :)
    AX => state(prm%i_ay, :)
    ! ATH => state(prm%i_ath, :)

    ! time step parameters
    half_dt = 0.5D0 * dt
    half_dt2 = half_dt * dt  ! 0.5 * dt^2

    ! r_i += v_i * dt + a_i * dt^2 / 2
    ! v_i += a_i * dt / 2
    do iP = 1, n_particles
       X(iP)  = X(iP) + VX(iP) * dt + AX(iP) * half_dt2
       VX(iP) = VX(iP) + AX(iP) * half_dt

       Y(iP)  = Y(iP) + VY(iP) * dt + AY(iP) * half_dt2
       VY(iP) = VY(iP) + AY(iP) * half_dt
    end do

    ! compute accelerations
    call compute_accelerations(prm, boundParticles, X, Y, AX, AY, stat)

    ! second half of velocity update
    do iP = 1, n_particles
       VX(iP) = VX(iP) + AX(iP) * half_dt
       VY(iP) = VY(iP) + AY(iP) * half_dt
    end do

    ! force zero velocity for fixed particles
    do iP = 1, size(fixedParticles)
       VX(iP) = 0.0D0
       VY(iP) = 0.0D0
    end do
  end subroutine step


  subroutine singleParticleForces(prm, X, Y, AX, AY, stat)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:), intent(in) :: X, Y
    integer, parameter :: n_particles = size(X)
    real(f64), dimension(:), intent(inout) :: AX, AY
    type(Statistics), intent(out) :: stat

    real(f64) :: wallForce_tot

    ! box area
    boxArea = 4.0D0 * boxWidth

    ! UWall = 0.0D0
    wallForce_tot = 0.0D0

    ! TODO: use do concurrent
    ! check for bounces off walls and apply wall forces
    do iP = 1, n_particles
       AX(iP) = 0.0D0
       AY(iP) = 0.0D0

       ! apply gravity (constant force)
       AY(iP) = AY(iP) - prm%gGravity

       wallForce_tot = wallForce_tot + wallForce(X(iP), Y(iP))
    end do

    ! calculate instantaneous pressure on walls
    stat%pressure = wallForce_tot / prm%boxArea

  contains

    function wallForce(x_i, y_i, ai_x, ai_y)

      real(f64), intent(in) :: x_i, y_i
      real(f64), intent(out) :: ai_x, ai_y
      real(f64) :: wallForce

      ! wall interactions on x-axis
      call wallForce1d(x_i, ai_x)
      wallForce = wallForce + ai_x
      ! UWall = UWall + U_w

      ! wall interactions on y-axis
      call wallForce1d(y_i, ai_y)
      wallForce = wallForce + ai_y
      ! UWall = UWall + U_w

    contains

      subroutine wallForce1d(rP, A_w)
        real(f64), intent(in) :: rP
        real(f64), intent(out) :: A_w

        real(f64) :: dist, wallLim

        wallLim = prm%boxWidth - prm%wallThickness
        A_w = 0.0D0
        ! U_w = 0.0D0

        ! apply wall force

        ! CHECK
        if (rP < prm%wallThickness) then
           dist = prm%wallThickness - rP   ! > 0
           A_w = prm%wallStiffness * dist  ! positive
           ! U_w = 0.5D0 * A_w * dist - prm%U_wc
        else if (rP > wallLim) then
           dist = wallLim - rP         ! < 0
           A_w = wallStiffness * dist  ! negative
           ! U_w = 0.5D0 * A_w * dist - prm%U_wc
        end if
      end subroutine wallForce1d

    end function wallForce

  end subroutine singleParticleForces


  subroutine twoParticleForce(prm, X, Y, AX, AY)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:), intent(in) :: X, Y
    integer, parameter :: n_particles = size(X)
    real(f64), dimension(:), intent(inout) :: AX, AY

    real(f64) :: U_LJ, dx, dy, dx2, dy2, r2, r2Inv, &
         attract, repel, Fi_from_j, fx, fy
    integer :: iP, jP

    ! r_c for Lennard-Jones potential (in units of sigma)
    rC_LJ = 3.0D0
    rC_LJ2 = rC_LJ * rC_LJ
    ! U_c for Lennard-Jones potential (in units of epsilon)
    UC_LJ = 4.0D0 * (1.0D0 / rC_LJ**12 - 1.0D0 / rC_LJ**6)

    ! U_LJ = 0.0D0

    ! apply inter-particle Lennard-Jones forces
    do iP = 1, n_particles
       do jP = 1, iP - 1
          dx = X(iP) - X(jP)
          dx2 = dx * dx
          if (dx2 >= prm%rC_LJ2) cycle

          dy = Y(iP) - Y(jP)
          dy2 = dy * dy
          if (dy2 >= prm%rC_LJ2) cycle

          r2 = dx2 + dy2
          if (r2 >= prm%rC_LJ2) cycle

          call LennardJones2P(X(iP), Y(iP), X(jP), Y(jP), ai_x, ai_y)

          ! force on i from j
          AX(iP) = AX(iP) + ai_x
          AY(iP) = AY(iP) + ai_y

          ! force on j from i
          AX(jP) = AX(jP) - ai_x
          AY(jP) = AY(jP) - ai_y
       end do
    end do

  contains

    subroutine LennardJones2P(prm, x_i, y_i, x_j, y_j, ai_x, ai_y)
      type(Parameters), intent(in) :: prm
      real(f64), intent(in) :: x_i, y_i, x_j, y_j
      real(f64), intent(out) :: ai_x, ai_y

      ! r_c for Lennard-Jones potential (in units of sigma)
      rC_LJ = 3.0D0
      rC_LJ2 = rC_LJ * rC_LJ
      ! U_c for Lennard-Jones potential (in units of epsilon)
      UC_LJ = 4.0D0 * (1.0D0 / (rC_LJ**12 + EPS) - 1.0D0 / (rC_LJ**6 + EPS))

      ! U_LJ = 0.0D0

      ! r2 < rC_LJ2
      r2Inv = 1.0D0 / (r2 + EPS)
      attract = r2Inv * r2Inv * r2Inv  ! = 1 / r^6
      repel = attract * attract        ! = 1 / r^12

      ! U_LJ = U_LJ + (4.0D0 * (repel - attract)) - prm%UC_LJ

      Fi_from_j = 24.0D0 * (2.0D0 * repel - attract) * r2Inv

      ai_x = Fi_from_j * dx
      ai_y = Fi_from_j * dy
    end subroutine LennardJones2P

  end subroutine twoParticleForce


  subroutine bondForce(prm, boundParticles, X, Y, AX, AY)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:, 2), intent(in) :: boundParticles
    real(f64), dimension(:), intent(in) :: X, Y
    real(f64), dimension(:), intent(inout) :: AX, AY

    real(f64) :: UBond, dx, dy, r_, dR, F0, fx, fy
    integer :: iB, iP, jP

    ! bonds
    bondStrength = 100.0D0  ! in units of sigma^2
    R0 = 1.122462D0

    ! UBond = 0.0D0

    ! add elastic forces between bonded atoms
    do iB = 1, size(boundParticles, 1)
       iP = boundParticles(iB, 1)
       jP = boundParticles(iB, 2)

       dx = X(iP) - X(jP)
       dy = Y(iP) - Y(jP)
       rr = sqrt(dx * dx + dy * dy)
       dR = rr - prm%R0
       F0 = prm%bondStrength * dR / (rr + EPS)

       ! UBond = UBond + 0.5D0 * F0 * dR

       fx = F0 * dx
       fy = F0 * dy

       AX(iP) = AX(iP) - fx
       AY(iP) = AY(iP) - fy

       AX(jP) = AX(jP) + fx
       AY(jP) = AY(jP) + fy
    end do

  end subroutine bondForce


  subroutine compute_accelerations(prm, boundParticles, X, Y, AX, AY, stat)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:, 2), intent(in) :: boundParticles
    real(f64), dimension(:), intent(in) :: X, Y
    integer, parameter :: n_particles = size(X)
    real(f64), dimension(:), intent(inout) :: AX, AY
    type(Statistics), intent(out) :: stat

    ! apply single-particle forces
    call singleParticleForces(prm, X, Y, AX, AY, stat)

    ! apply two-particle forces
    call twoParticleForces(prm, X, Y, AX, AY, stat)

    ! apply elastic forces between bonded particles
    call bondForce(prm, boundParticles, X, Y, AX, AY)
  end subroutine compute_accelerations


  subroutine statistics(prm, X, Y, VX, VY, stat)
    real(f64), dimension(:), intent(in) :: X, Y, VX, VY
    integer, parameter :: n_particles = size(X)
    type(Statistics), intent(inout) :: stat

    integer :: iP

    ! reset statistical accumulators
    stat%kineticE = 0.0D0

    ! kinetic energy
    do iP = 1, n_particles
       ! K = 0.5 * v^2
       stat%kineticE = stat%kineticE + 0.5D0 * (VX(iP) * VX(iP) + VY(iP) * VY(iP))
    end do

    ! current temperature
    stat%currentT = stat%kineticE / real(n_particles - prm%n_fixed, f64)

    ! TODO: calculate avg. polarization
  end subroutine statistics


  subroutine BoxMuller()
    ! TODO: use do concurrent
    threshold = 5.0D0 * dt
    wt = 2.0D0
    T_sqrt = sqrt(T)

    ! Box-Muller algorithm: <https://rh8liuqy.github.io/Box_Muller_Algorithm.html>
    ! assign random velocities to fixed-temperature atoms
    do i_F = 1, size(fixedTList)
       ! probabilistic update (equivalent to random < 5*dt)
       call random_number(r1)
       ! perform this only a small percentage of the time
       if (r1 < threshold) then
          ! Box-Muller transformation
          do while wt >= 1.0D0
             call random_number(r1)
             call random_number(r2)
             ! r = 2 * random[0, 1) - 1
             r1 = 2.0D0 * r1 - 1.0D0
             r2 = 2.0D0 * r2 - 1.0D0
             wt = r1 * r1 + r2 * r2
          end do

          ! 0 < wt < 1
          uBM = sqrt(-2.0D0 * log(wt) / wt)

          ! update velocities with Gaussian distribution
          uT = uBM * T_sqrt
          X(fixedTList(i_F)) = uT * r1
          Y(fixedTList(i_F)) = uT * r2
       end if
    end do

  end subroutine BoxMuller

end module MolecularDynamics
