module MolecularDynamics
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none

    ! double precision kind parameter
    integer, parameter :: f64 = selected_real_kind(c_double)
    integer, parameter :: i32 = selected_int_kind(c_int)

    ! assumed to be defined elsewhere in the module
    integer(kind=i32) :: n_particles  ! total number of particles
    real(kind=f64) :: T  ! temperature

    real(kind=f64), allocatable :: state(:, :)
    integer, allocatable :: fixedParticles(:)
    integer, allocatable :: boundParticles(:, 2)

contains

  subroutine step(dt, prm, fixedParticles, boundParticles, state)
    real(f64), intent(in) :: dt
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:), intent(in) :: fixedParticles
    real(f64), dimension(:, 2), intent(in) :: boundParticles
    real(f64), dimension(prm%n_conf, :), intent(inout), target :: state
    integer, parameter :: n_particles = size(state, 2)
    real(f64), dimension(n_particles), pointer :: X, Y, VX, VY, AX, AY
    real(f64) :: half_dt, half_dt2
    integer :: iP, iFx

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
    call compute_accelerations(prm, boundParticles, X, Y, AX, AY)

    ! second half of velocity update
    do iP = 1, n_particles
       VX(iP) = VX(iP) + AX(iP) * half_dt
       VY(iP) = VY(iP) + AY(iP) * half_dt
    end do

    ! force zero velocity for fixed particles
    do iFx = 1, size(fixedParticles)
       VX(iFx) = 0.0D0
       VY(iFx) = 0.0D0
    end do
  end subroutine step


  subroutine wallForce(prm, X, Y, AX, AY, pressure)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:), intent(in) :: X, Y
    integer, parameter :: n_particles = size(X)
    real(f64), dimension(:), intent(inout) :: AX, AY
    real(f64), intent(out) :: pressure

    real(f64) :: half_wallThickness, UWall, wallForce, U_w, A_w

    ! walls
    wallThickness = 0.5D0   ! in units of sigma
    half_wallThickness = 0.5D0 * wallStiffness
    wallStiffness = 50.D0   ! in units of sigma^2
    wallLim = boxWidth - wallThickness
    wallUc = 0.5 * wallStiffness * wallThickness * wallThickness

    ! box area
    boxArea = 4.0D0 * boxWidth

    ! UWall = 0.0D0
    wallForce = 0.0D0
    pressure = 0.0D0

    ! TODO: use do concurrent
    ! check for bounces off walls and apply wall forces
    do iP = 1, n_particles
       AX(iP) = 0.0D0
       AY(iP) = 0.0D0

       ! wall interactions on x-axis
       call wallForce1d(X(iP), A_w)
       AX(iP) = A_w
       wallForce = wallForce + A_w
       ! UWall = UWall + U_w

       ! wall interactions on y-axis
       call wallForce1d(Y(iP), A_w)
       AY(iP) = A_w
       wallForce = wallForce + A_w
       ! UWall = UWall + U_w
    end do

    ! calculate instantaneous pressure on walls
    pressure = wallForce / prm%boxArea

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

  end subroutine wallForce


  subroutine LennardJonesForce(prm, X, Y, AX, AY)
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

          dy = Y(iP) - Y(jP)
          dy2 = dy * dy

          r2 = dx2 + dy2

          if (r2 >= prm%rC_LJ2) continue

          ! r2 < rC_LJ2
          r2Inv = 1.0D0 / r2
          attract = r2Inv * r2Inv * r2Inv  ! = 1 / r^6
          repel = attract * attract        ! = 1 / r^12

          ! U_LJ = U_LJ + (4.0D0 * (repel - attract)) - prm%UC_LJ

          Fi_from_j = 24.0D0 * (2.0D0 * repel - attract) * r2Inv

          fx = Fi_from_j * dx
          fy = Fi_from_j * dy

          ! Force on i from j
          AX(iP) = AX(iP) + fx
          AY(iP) = AY(iP) + fy

          ! Force on j from i
          AX(jP) = AX(jP) - fx
          AY(jP) = AY(jP) - fy
       end do
    end do

  end subroutine LennardJonesForce


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
       r_ = sqrt(dx * dx + dy * dy)
       dR = r_ - prm%R0
       F0 = prm%bondStrength * dR / r_

       ! UBond = UBond + 0.5D0 * F0 * dR

       fx = F0 * dx
       fy = F0 * dy

       AX(iP) = AX(iP) - fx
       AY(iP) = AY(iP) - fy

       AX(jP) = AX(jP) + fx
       AY(jP) = AY(jP) + fy
    end do

  end subroutine bondForce


  subroutine compute_accelerations(prm, boundParticles, X, Y, AX, AY, pressure)
    type(Parameters), intent(in) :: prm
    real(f64), dimension(:, 2), intent(in) :: boundParticles
    real(f64), dimension(:), intent(in) :: X, Y
    integer, parameter :: n_particles = size(X)
    real(f64), dimension(:), intent(inout) :: AX, AY
    real(f64), intent(out) :: pressure

    integer :: iP

    ! apply gravity (constant force)
    do iP = 1, n_particles
       AY(iP) = AY(iP) - prm%gGravity
    end do

    ! apply wall forces
    call wallForce(prm, X, Y, AX, AY, pressure)

    ! apply elastic forces between bonded particles
    call bondForce(prm, boundParticles, X, Y, AX, AY)

    ! Lennard-Jones force
    call LennardJonesForce(prm, X, Y, AX, AY)
  end subroutine compute_accelerations


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
  end subroutine stats

end module MolecularDynamics
