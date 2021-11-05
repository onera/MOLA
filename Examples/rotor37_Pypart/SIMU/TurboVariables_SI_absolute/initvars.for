
! ----------------------------------------------------------------------
      subroutine omegadim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        omegadim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: omegadim

!
!     Compute OmegaDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              omegadim(i) = omega
        enddo

      end


! ----------------------------------------------------------------------
      subroutine omegadim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        omegadim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: omegadim

!
!     Compute OmegaDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              omegadim(i,j) = omega
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine omegadim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        omegadim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: omegadim

!
!     Compute OmegaDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              omegadim(i,j,k) = omega
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine radius_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radius)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: radius

!
!     Compute Radius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              radius(i) = sqrt(coordinatey(i)**2+coordinatez(i)**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine radius_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radius)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: radius

!
!     Compute Radius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              radius(i,j) = sqrt(coordinatey(i,j)**2+coordinatez(i,j)**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine radius_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radius)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: radius

!
!     Compute Radius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              radius(i,j,k) = sqrt(coordinatey(i,j,k)**2+coordinatez(i,j,k)**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine radiusdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radiusdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: radiusdim

!
!     Compute RadiusDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              radiusdim(i) = sqrt(coordinatey(i)**2+coordinatez(i)**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine radiusdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radiusdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: radiusdim

!
!     Compute RadiusDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              radiusdim(i,j) = sqrt(coordinatey(i,j)**2+coordinatez(i,j)**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine radiusdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        radiusdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: radiusdim

!
!     Compute RadiusDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              radiusdim(i,j,k) = sqrt(coordinatey(i,j,k)**2+coordinatez(i,j,k)**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine thetaangle_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangle)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: thetaangle

!
!     Compute ThetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              thetaangle(i) = atan2(coordinatey(i), coordinatez(i))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine thetaangle_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: thetaangle

!
!     Compute ThetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              thetaangle(i,j) = atan2(coordinatey(i,j), coordinatez(i,j))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine thetaangle_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: thetaangle

!
!     Compute ThetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              thetaangle(i,j,k) = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine thetaangledegree_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: thetaangledegree

!
!     Compute ThetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              thetaangledegree(i) = (180.d0/3.1415926535897931)*thetaangle
        enddo

      end


! ----------------------------------------------------------------------
      subroutine thetaangledegree_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: thetaangledegree

!
!     Compute ThetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              thetaangledegree(i,j) = (180.d0/3.1415926535897931)*thetaangle
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine thetaangledegree_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        thetaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: thetaangledegree

!
!     Compute ThetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              thetaangledegree(i,j,k) = (180.d0/3.1415926535897931)*thetaangle
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine densitydim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        densitydim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: densitydim

!
!     Compute DensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              densitydim(i) = density(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine densitydim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        densitydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: densitydim

!
!     Compute DensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              densitydim(i,j) = density(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine densitydim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        densitydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: densitydim

!
!     Compute DensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              densitydim(i,j,k) = density(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumxabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumxabsdim

!
!     Compute MomentumXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim(i) = momentumx(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumxabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumxabsdim

!
!     Compute MomentumXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim(i,j) = momentumx(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumxabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumxabsdim

!
!     Compute MomentumXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim(i,j,k) = momentumx(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumyabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumyabsdim

!
!     Compute MomentumYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim(i) = momentumy(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumyabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumyabsdim

!
!     Compute MomentumYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim(i,j) = momentumy(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumyabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumyabsdim

!
!     Compute MomentumYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim(i,j,k) = momentumy(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumzabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumzabsdim

!
!     Compute MomentumZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumzabsdim(i) = momentumz(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumzabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumzabsdim

!
!     Compute MomentumZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumzabsdim(i,j) = momentumz(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumzabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumzabsdim

!
!     Compute MomentumZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumzabsdim(i,j,k) = momentumz(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine energystagnationdensitydim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        energystagnationdensitydim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: energystagnationdensitydim

!
!     Compute EnergyStagnationDensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim(i) = energystagnationdensity(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine energystagnationdensitydim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        energystagnationdensitydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: energystagnationdensitydim

!
!     Compute EnergyStagnationDensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim(i,j) = energystagnationdensity(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine energystagnationdensitydim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        energystagnationdensitydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: energystagnationdensitydim

!
!     Compute EnergyStagnationDensityDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim(i,j,k) = energystagnationdensity(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine staticpressure_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressure)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: staticpressure

!
!     Compute StaticPressure with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              staticpressure(i) = (gamma-1.)*(energystagnationdensity(i) - (0.5*(momentumx(i)**2+momentumy(i)**2+momentumz(i)**2)/density(i)))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticpressure_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressure)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: staticpressure

!
!     Compute StaticPressure with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              staticpressure(i,j) = (gamma-1.)*(energystagnationdensity(i,j) - (0.5*(momentumx(i,j)**2+momentumy(i,j)**2+momentumz(i,j)**2)/density(i,j)))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticpressure_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressure)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: staticpressure

!
!     Compute StaticPressure with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              staticpressure(i,j,k) = (gamma-1.)*(energystagnationdensity(i,j,k) - (0.5*(momentumx(i,j,k)**2+momentumy(i,j,k)**2+momentumz(i,j,k)**2)/density(i,j,k)))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine staticpressuredim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressuredim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: staticpressuredim

!
!     Compute StaticPressureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim(i) = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticpressuredim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressuredim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: staticpressuredim

!
!     Compute StaticPressureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim(i,j) = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticpressuredim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticpressuredim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: staticpressuredim

!
!     Compute StaticPressureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim(i,j,k) = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine statictemperature_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperature)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: staticpressure
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: statictemperature

!
!     Compute StaticTemperature with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i) - (0.5*(momentumx(i)**2+momentumy(i)**2+momentumz(i)**2)/density(i)))
              statictemperature(i) = staticpressure/(rgaz*density(i))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine statictemperature_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperature)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: staticpressure
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: statictemperature

!
!     Compute StaticTemperature with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i,j) - (0.5*(momentumx(i,j)**2+momentumy(i,j)**2+momentumz(i,j)**2)/density(i,j)))
              statictemperature(i,j) = staticpressure/(rgaz*density(i,j))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine statictemperature_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperature)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: staticpressure
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: statictemperature

!
!     Compute StaticTemperature with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i,j,k) - (0.5*(momentumx(i,j,k)**2+momentumy(i,j,k)**2+momentumz(i,j,k)**2)/density(i,j,k)))
              statictemperature(i,j,k) = staticpressure/(rgaz*density(i,j,k))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine statictemperaturedim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperaturedim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: statictemperaturedim

!
!     Compute StaticTemperatureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim(i) = staticpressuredim/(rgaz*densitydim)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine statictemperaturedim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperaturedim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: statictemperaturedim

!
!     Compute StaticTemperatureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim(i,j) = staticpressuredim/(rgaz*densitydim)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine statictemperaturedim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        statictemperaturedim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: statictemperaturedim

!
!     Compute StaticTemperatureDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim(i,j,k) = staticpressuredim/(rgaz*densitydim)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine staticenthalpy_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpy)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: staticpressure
        real(kind=8) :: statictemperature
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: staticenthalpy

!
!     Compute StaticEnthalpy with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i) - (0.5*(momentumx(i)**2+momentumy(i)**2+momentumz(i)**2)/density(i)))
              statictemperature = staticpressure/(rgaz*density(i))
              staticenthalpy(i) = (rgaz)/(gamma-1.)*statictemperature
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticenthalpy_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpy)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: staticpressure
        real(kind=8) :: statictemperature
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: staticenthalpy

!
!     Compute StaticEnthalpy with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i,j) - (0.5*(momentumx(i,j)**2+momentumy(i,j)**2+momentumz(i,j)**2)/density(i,j)))
              statictemperature = staticpressure/(rgaz*density(i,j))
              staticenthalpy(i,j) = (rgaz)/(gamma-1.)*statictemperature
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticenthalpy_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpy)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: staticpressure
        real(kind=8) :: statictemperature
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: staticenthalpy

!
!     Compute StaticEnthalpy with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              staticpressure = (gamma-1.)*(energystagnationdensity(i,j,k) - (0.5*(momentumx(i,j,k)**2+momentumy(i,j,k)**2+momentumz(i,j,k)**2)/density(i,j,k)))
              statictemperature = staticpressure/(rgaz*density(i,j,k))
              staticenthalpy(i,j,k) = (rgaz)/(gamma-1.)*statictemperature
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine staticenthalpydim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpydim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: staticenthalpydim

!
!     Compute StaticEnthalpyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              staticenthalpydim(i) = (rgaz)/(gamma-1.)*statictemperaturedim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticenthalpydim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: staticenthalpydim

!
!     Compute StaticEnthalpyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              staticenthalpydim(i,j) = (rgaz)/(gamma-1.)*statictemperaturedim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine staticenthalpydim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        staticenthalpydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: staticenthalpydim

!
!     Compute StaticEnthalpyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              staticenthalpydim(i,j,k) = (rgaz)/(gamma-1.)*statictemperaturedim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityx_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityx)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityx

!
!     Compute VelocityX with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityx(i) = momentumx(i)/density(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityx_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityx)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityx

!
!     Compute VelocityX with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityx(i,j) = momentumx(i,j)/density(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityx_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityx)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityx

!
!     Compute VelocityX with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityx(i,j,k) = momentumx(i,j,k)/density(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityy_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityy)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityy

!
!     Compute VelocityY with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityy(i) = momentumy(i)/density(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityy_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityy)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityy

!
!     Compute VelocityY with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityy(i,j) = momentumy(i,j)/density(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityy_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityy)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityy

!
!     Compute VelocityY with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityy(i,j,k) = momentumy(i,j,k)/density(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityz_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityz)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityz

!
!     Compute VelocityZ with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityz(i) = momentumz(i)/density(i)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityz_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityz)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityz

!
!     Compute VelocityZ with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityz(i,j) = momentumz(i,j)/density(i,j)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityz_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityz)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k


        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityz

!
!     Compute VelocityZ with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityz(i,j,k) = momentumz(i,j,k)/density(i,j,k)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitymagnitude_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitude)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitymagnitude

!
!     Compute VelocityMagnitude with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityx = momentumx(i)/density(i)
              velocityy = momentumy(i)/density(i)
              velocityz = momentumz(i)/density(i)
              velocitymagnitude(i) = sqrt(velocityx**2+velocityy**2+velocityz**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitude_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitude)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitymagnitude

!
!     Compute VelocityMagnitude with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityx = momentumx(i,j)/density(i,j)
              velocityy = momentumy(i,j)/density(i,j)
              velocityz = momentumz(i,j)/density(i,j)
              velocitymagnitude(i,j) = sqrt(velocityx**2+velocityy**2+velocityz**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitude_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitude)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitymagnitude

!
!     Compute VelocityMagnitude with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityx = momentumx(i,j,k)/density(i,j,k)
              velocityy = momentumy(i,j,k)/density(i,j,k)
              velocityz = momentumz(i,j,k)/density(i,j,k)
              velocitymagnitude(i,j,k) = sqrt(velocityx**2+velocityy**2+velocityz**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitymeridian_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridian)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8) :: velocityradius
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitymeridian

!
!     Compute VelocityMeridian with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityx = momentumx(i)/density(i)
              velocityy = momentumy(i)/density(i)
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              velocityz = momentumz(i)/density(i)
              velocityradius = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
              velocitymeridian(i) = sqrt(velocityx**2+velocityradius**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymeridian_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridian)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8) :: velocityradius
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitymeridian

!
!     Compute VelocityMeridian with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityx = momentumx(i,j)/density(i,j)
              velocityy = momentumy(i,j)/density(i,j)
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              velocityz = momentumz(i,j)/density(i,j)
              velocityradius = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
              velocitymeridian(i,j) = sqrt(velocityx**2+velocityradius**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymeridian_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridian)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityz
        real(kind=8) :: velocityx
        real(kind=8) :: velocityy
        real(kind=8) :: velocityradius
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitymeridian

!
!     Compute VelocityMeridian with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityx = momentumx(i,j,k)/density(i,j,k)
              velocityy = momentumy(i,j,k)/density(i,j,k)
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              velocityz = momentumz(i,j,k)/density(i,j,k)
              velocityradius = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
              velocitymeridian(i,j,k) = sqrt(velocityx**2+velocityradius**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityxabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityxabsdim

!
!     Compute VelocityXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim(i) = momentumxabsdim/densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityxabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityxabsdim

!
!     Compute VelocityXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim(i,j) = momentumxabsdim/densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityxabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityxabsdim

!
!     Compute VelocityXAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim(i,j,k) = momentumxabsdim/densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityyabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityyabsdim

!
!     Compute VelocityYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim(i) = momentumyabsdim/densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityyabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityyabsdim

!
!     Compute VelocityYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim(i,j) = momentumyabsdim/densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityyabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityyabsdim

!
!     Compute VelocityYAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim(i,j,k) = momentumyabsdim/densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityzabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityzabsdim

!
!     Compute VelocityZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              velocityzabsdim(i) = momentumzabsdim/densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityzabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityzabsdim

!
!     Compute VelocityZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              velocityzabsdim(i,j) = momentumzabsdim/densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityzabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityzabsdim

!
!     Compute VelocityZAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              velocityzabsdim(i,j,k) = momentumzabsdim/densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityxreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityxreldim

!
!     Compute VelocityXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim(i) = velocityxabsdim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityxreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityxreldim

!
!     Compute VelocityXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim(i,j) = velocityxabsdim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityxreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityxreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityxreldim

!
!     Compute VelocityXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim(i,j,k) = velocityxabsdim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityyreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityyreldim

!
!     Compute VelocityYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim(i) = velocityyabsdim + omegadim*coordinatez(i)*lref
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityyreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityyreldim

!
!     Compute VelocityYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim(i,j) = velocityyabsdim + omegadim*coordinatez(i,j)*lref
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityyreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityyreldim

!
!     Compute VelocityYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim(i,j,k) = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityzreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityzreldim

!
!     Compute VelocityZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim(i) = velocityzabsdim - omegadim*coordinatey(i)*lref
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityzreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityzreldim

!
!     Compute VelocityZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim(i,j) = velocityzabsdim - omegadim*coordinatey(i,j)*lref
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityzreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityzreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityzreldim

!
!     Compute VelocityZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim(i,j,k) = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitymagnitudeabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudeabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitymagnitudeabsdim

!
!     Compute VelocityMagnitudeAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitymagnitudeabsdim(i) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitudeabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudeabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitymagnitudeabsdim

!
!     Compute VelocityMagnitudeAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitymagnitudeabsdim(i,j) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitudeabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudeabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitymagnitudeabsdim

!
!     Compute VelocityMagnitudeAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitymagnitudeabsdim(i,j,k) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitymagnitudereldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudereldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitymagnitudereldim

!
!     Compute VelocityMagnitudeRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocitymagnitudereldim(i) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitudereldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitymagnitudereldim

!
!     Compute VelocityMagnitudeRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocitymagnitudereldim(i,j) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymagnitudereldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymagnitudereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitymagnitudereldim

!
!     Compute VelocityMagnitudeRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocitymagnitudereldim(i,j,k) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityradius_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradius)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityradius

!
!     Compute VelocityRadius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityy = momentumy(i)/density(i)
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              velocityz = momentumz(i)/density(i)
              velocityradius(i) = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradius_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradius)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityradius

!
!     Compute VelocityRadius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityy = momentumy(i,j)/density(i,j)
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              velocityz = momentumz(i,j)/density(i,j)
              velocityradius(i,j) = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradius_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradius)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityradius

!
!     Compute VelocityRadius with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityy = momentumy(i,j,k)/density(i,j,k)
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              velocityz = momentumz(i,j,k)/density(i,j,k)
              velocityradius(i,j,k) = velocityy*sin(thetaangle) + velocityz*cos(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitytheta_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitytheta)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitytheta

!
!     Compute VelocityTheta with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              velocityy = momentumy(i)/density(i)
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              velocityz = momentumz(i)/density(i)
              velocitytheta(i) = velocityy*cos(thetaangle) - velocityz*sin(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitytheta_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitytheta)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitytheta

!
!     Compute VelocityTheta with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              velocityy = momentumy(i,j)/density(i,j)
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              velocityz = momentumz(i,j)/density(i,j)
              velocitytheta(i,j) = velocityy*cos(thetaangle) - velocityz*sin(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitytheta_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitytheta)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityz
        real(kind=8) :: velocityy
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitytheta

!
!     Compute VelocityTheta with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              velocityy = momentumy(i,j,k)/density(i,j,k)
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              velocityz = momentumz(i,j,k)/density(i,j,k)
              velocitytheta(i,j,k) = velocityy*cos(thetaangle) - velocityz*sin(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityradiusabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityradiusabsdim

!
!     Compute VelocityRadiusAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim(i) = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradiusabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityradiusabsdim

!
!     Compute VelocityRadiusAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim(i,j) = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradiusabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityradiusabsdim

!
!     Compute VelocityRadiusAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim(i,j,k) = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitythetaabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetaabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitythetaabsdim

!
!     Compute VelocityThetaAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim(i) = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitythetaabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetaabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitythetaabsdim

!
!     Compute VelocityThetaAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim(i,j) = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitythetaabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetaabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitythetaabsdim

!
!     Compute VelocityThetaAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim(i,j,k) = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocityradiusreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocityradiusreldim

!
!     Compute VelocityRadiusRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocityradiusreldim(i) = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradiusreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocityradiusreldim

!
!     Compute VelocityRadiusRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocityradiusreldim(i,j) = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocityradiusreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocityradiusreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocityradiusreldim

!
!     Compute VelocityRadiusRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocityradiusreldim(i,j,k) = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitythetareldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetareldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitythetareldim

!
!     Compute VelocityThetaRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocitythetareldim(i) = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitythetareldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetareldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitythetareldim

!
!     Compute VelocityThetaRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocitythetareldim(i,j) = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitythetareldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitythetareldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitythetareldim

!
!     Compute VelocityThetaRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocitythetareldim(i,j,k) = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine velocitymeridiandim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridiandim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: velocitymeridiandim

!
!     Compute VelocityMeridianDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim(i) = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymeridiandim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridiandim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: velocitymeridiandim

!
!     Compute VelocityMeridianDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim(i,j) = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine velocitymeridiandim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        velocitymeridiandim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: velocitymeridiandim

!
!     Compute VelocityMeridianDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim(i,j,k) = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumxreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumxreldim

!
!     Compute MomentumXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumxreldim(i) = velocityxreldim*densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumxreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumxreldim

!
!     Compute MomentumXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumxreldim(i,j) = velocityxreldim*densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumxreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumxreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumxreldim

!
!     Compute MomentumXRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumxreldim(i,j,k) = velocityxreldim*densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumyreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumyreldim

!
!     Compute MomentumYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              momentumyreldim(i) = velocityyreldim*densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumyreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumyreldim

!
!     Compute MomentumYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              momentumyreldim(i,j) = velocityyreldim*densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumyreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumyreldim

!
!     Compute MomentumYRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              momentumyreldim(i,j,k) = velocityyreldim*densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine momentumzreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: momentumzreldim

!
!     Compute MomentumZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              momentumzreldim(i) = velocityzreldim*densitydim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumzreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: momentumzreldim

!
!     Compute MomentumZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              momentumzreldim(i,j) = velocityzreldim*densitydim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine momentumzreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        momentumzreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityzabsdim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: momentumzreldim

!
!     Compute MomentumZRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              omegadim = omega
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              momentumzreldim(i,j,k) = velocityzreldim*densitydim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine alphaangle_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangle)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: alphaangle

!
!     Compute AlphaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle(i) = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine alphaangle_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: alphaangle

!
!     Compute AlphaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle(i,j) = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine alphaangle_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: alphaangle

!
!     Compute AlphaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle(i,j,k) = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine betaangle_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangle)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: betaangle

!
!     Compute BetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle(i) = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine betaangle_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: betaangle

!
!     Compute BetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle(i,j) = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine betaangle_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: betaangle

!
!     Compute BetaAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle(i,j,k) = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine phiangle_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangle)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: phiangle

!
!     Compute PhiAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle(i) = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine phiangle_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: phiangle

!
!     Compute PhiAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle(i,j) = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine phiangle_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangle)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: phiangle

!
!     Compute PhiAngle with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle(i,j,k) = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine alphaangledegree_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: alphaangle
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: alphaangledegree

!
!     Compute AlphaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
              alphaangledegree(i) = (180.d0/3.1415926535897931)*alphaangle
        enddo

      end


! ----------------------------------------------------------------------
      subroutine alphaangledegree_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: alphaangle
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: alphaangledegree

!
!     Compute AlphaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
              alphaangledegree(i,j) = (180.d0/3.1415926535897931)*alphaangle
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine alphaangledegree_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        alphaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: velocitythetaabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: alphaangle
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: alphaangledegree

!
!     Compute AlphaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocitythetaabsdim = velocityyabsdim*cos(thetaangle) - velocityzabsdim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              alphaangle = atan(velocitythetaabsdim/max(0.0000001d0, velocitymeridiandim))
              alphaangledegree(i,j,k) = (180.d0/3.1415926535897931)*alphaangle
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine betaangledegree_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: betaangle
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: betaangledegree

!
!     Compute BetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
              betaangledegree(i) = (180.d0/3.1415926535897931)*betaangle
        enddo

      end


! ----------------------------------------------------------------------
      subroutine betaangledegree_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: betaangle
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: betaangledegree

!
!     Compute BetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
              betaangledegree(i,j) = (180.d0/3.1415926535897931)*betaangle
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine betaangledegree_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        betaangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityradiusabsdim
        real(kind=8) :: betaangle
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: velocitythetareldim
        real(kind=8) :: velocitymeridiandim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: betaangledegree

!
!     Compute BetaAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocitythetareldim = velocityyreldim*cos(thetaangle) - velocityzreldim*sin(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityradiusabsdim = velocityyabsdim*sin(thetaangle) + velocityzabsdim*cos(thetaangle)
              velocitymeridiandim = sqrt(velocityxabsdim**2+velocityradiusabsdim**2)
              betaangle = atan(velocitythetareldim/max(0.0000001d0, velocitymeridiandim))
              betaangledegree(i,j,k) = (180.d0/3.1415926535897931)*betaangle
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine phiangledegree_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangledegree)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: phiangle
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: phiangledegree

!
!     Compute PhiAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumyabsdim = momentumy(i)
              densitydim = density(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              thetaangle = atan2(coordinatey(i), coordinatez(i))
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
              phiangledegree(i) = (180.d0/3.1415926535897931)*phiangle
        enddo

      end


! ----------------------------------------------------------------------
      subroutine phiangledegree_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: phiangle
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: phiangledegree

!
!     Compute PhiAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumyabsdim = momentumy(i,j)
              densitydim = density(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              thetaangle = atan2(coordinatey(i,j), coordinatez(i,j))
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
              phiangledegree(i,j) = (180.d0/3.1415926535897931)*phiangle
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine phiangledegree_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        phiangledegree)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityradiusreldim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: phiangle
        real(kind=8) :: thetaangle
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: phiangledegree

!
!     Compute PhiAngleDegree with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumyabsdim = momentumy(i,j,k)
              densitydim = density(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              thetaangle = atan2(coordinatey(i,j,k), coordinatez(i,j,k))
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              velocityradiusreldim = velocityyreldim*sin(thetaangle) + velocityzreldim*cos(thetaangle)
              momentumxabsdim = momentumx(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              phiangle = atan(velocityradiusreldim/max(0.0000001d0, velocityxreldim))
              phiangledegree(i,j,k) = (180.d0/3.1415926535897931)*phiangle
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine entropydim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        entropydim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: entropydim

!
!     Compute EntropyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              entropydim(i) = (gamma*rgaz)/(gamma-1.)*log(statictemperaturedim) - rgaz*log(staticpressuredim)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine entropydim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        entropydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: entropydim

!
!     Compute EntropyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              entropydim(i,j) = (gamma*rgaz)/(gamma-1.)*log(statictemperaturedim) - rgaz*log(staticpressuredim)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine entropydim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        entropydim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: entropydim

!
!     Compute EntropyDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              entropydim(i,j,k) = (gamma*rgaz)/(gamma-1.)*log(statictemperaturedim) - rgaz*log(staticpressuredim)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine machnumberabs_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberabs)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: machnumberabs

!
!     Compute MachNumberAbs with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              energystagnationdensitydim = energystagnationdensity(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs(i) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine machnumberabs_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberabs)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: machnumberabs

!
!     Compute MachNumberAbs with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              energystagnationdensitydim = energystagnationdensity(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs(i,j) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine machnumberabs_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberabs)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: machnumberabs

!
!     Compute MachNumberAbs with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              momentumyabsdim = momentumy(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs(i,j,k) = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine machnumberrel_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberrel)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: machnumberrel

!
!     Compute MachNumberRel with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              momentumxabsdim = momentumx(i)
              densitydim = density(i)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              momentumzabsdim = momentumz(i)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              energystagnationdensitydim = energystagnationdensity(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel(i) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine machnumberrel_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberrel)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: machnumberrel

!
!     Compute MachNumberRel with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              momentumxabsdim = momentumx(i,j)
              densitydim = density(i,j)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i,j)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              momentumzabsdim = momentumz(i,j)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              energystagnationdensitydim = energystagnationdensity(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel(i,j) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine machnumberrel_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        machnumberrel)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: machnumberrel

!
!     Compute MachNumberRel with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              momentumxabsdim = momentumx(i,j,k)
              densitydim = density(i,j,k)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              momentumyabsdim = momentumy(i,j,k)
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              momentumzabsdim = momentumz(i,j,k)
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel(i,j,k) = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationpressureabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationpressureabsdim

!
!     Compute StagnationPressureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressureabsdim(i) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberabs**2)**(gamma/(gamma-1.))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationpressureabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationpressureabsdim

!
!     Compute StagnationPressureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressureabsdim(i,j) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberabs**2)**(gamma/(gamma-1.))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationpressureabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationpressureabsdim

!
!     Compute StagnationPressureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressureabsdim(i,j,k) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberabs**2)**(gamma/(gamma-1.))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationtemperatureabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperatureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationtemperatureabsdim

!
!     Compute StagnationTemperatureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim(i) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationtemperatureabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperatureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationtemperatureabsdim

!
!     Compute StagnationTemperatureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim(i,j) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationtemperatureabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperatureabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationtemperatureabsdim

!
!     Compute StagnationTemperatureAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim(i,j,k) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationpressurereldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressurereldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationpressurereldim

!
!     Compute StagnationPressureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressurereldim(i) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberrel**2)**(gamma/(gamma-1.))
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationpressurereldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressurereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationpressurereldim

!
!     Compute StagnationPressureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressurereldim(i,j) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberrel**2)**(gamma/(gamma-1.))
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationpressurereldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationpressurereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationpressurereldim

!
!     Compute StagnationPressureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationpressurereldim(i,j,k) = staticpressuredim*(1.+0.5*(gamma-1.)*machnumberrel**2)**(gamma/(gamma-1.))
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationtemperaturereldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperaturereldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationtemperaturereldim

!
!     Compute StagnationTemperatureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim(i) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationtemperaturereldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperaturereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationtemperaturereldim

!
!     Compute StagnationTemperatureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim(i,j) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationtemperaturereldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationtemperaturereldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationtemperaturereldim

!
!     Compute StagnationTemperatureRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim(i,j,k) = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationenthalpyreldim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyreldim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: stagnationtemperaturereldim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationenthalpyreldim

!
!     Compute StagnationEnthalpyRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
              stagnationenthalpyreldim(i) = (gamma*rgaz)/(gamma-1.)*stagnationtemperaturereldim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationenthalpyreldim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: stagnationtemperaturereldim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationenthalpyreldim

!
!     Compute StagnationEnthalpyRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
              stagnationenthalpyreldim(i,j) = (gamma*rgaz)/(gamma-1.)*stagnationtemperaturereldim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationenthalpyreldim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyreldim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: momentumzabsdim
        real(kind=8) :: densitydim
        real(kind=8) :: omegadim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: velocityyabsdim
        real(kind=8) :: stagnationtemperaturereldim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: velocityxreldim
        real(kind=8) :: velocityzreldim
        real(kind=8) :: velocityyreldim
        real(kind=8) :: statictemperaturedim
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: machnumberrel
        real(kind=8) :: energystagnationdensitydim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationenthalpyreldim

!
!     Compute StagnationEnthalpyRelDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityxreldim = velocityxabsdim
              velocityyabsdim = momentumyabsdim/densitydim
              omegadim = omega
              velocityyreldim = velocityyabsdim + omegadim*coordinatez(i,j,k)*lref
              velocityzabsdim = momentumzabsdim/densitydim
              velocityzreldim = velocityzabsdim - omegadim*coordinatey(i,j,k)*lref
              machnumberrel = sqrt(velocityxreldim**2+velocityyreldim**2+velocityzreldim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperaturereldim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberrel**2)
              stagnationenthalpyreldim(i,j,k) = (gamma*rgaz)/(gamma-1.)*stagnationtemperaturereldim
            enddo
          enddo
        enddo

      end




! ----------------------------------------------------------------------
      subroutine stagnationenthalpyabsdim_1d(im, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im
        integer(kind=4) :: i

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: stagnationtemperatureabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im), intent(in) :: coordinatex
        real(kind=8), dimension(im), intent(in) :: coordinatey
        real(kind=8), dimension(im), intent(in) :: coordinatez
        real(kind=8), dimension(im), intent(in) :: density
        real(kind=8), dimension(im), intent(in) :: momentumx
        real(kind=8), dimension(im), intent(in) :: momentumy
        real(kind=8), dimension(im), intent(in) :: momentumz
        real(kind=8), dimension(im), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im), intent(inout) :: stagnationenthalpyabsdim

!
!     Compute StagnationEnthalpyAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=1)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=1)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=1)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=1)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=1)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=1)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=1)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=1)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=1)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=1)
!        :param: density: density
!        :type:  density: numpy array (rank=1)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=1)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=1)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=1)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=1)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im
        do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i)
              momentumxabsdim = momentumx(i)
              momentumyabsdim = momentumy(i)
              momentumzabsdim = momentumz(i)
              densitydim = density(i)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
              stagnationenthalpyabsdim(i) = (gamma*rgaz)/(gamma-1.)*stagnationtemperatureabsdim
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationenthalpyabsdim_2d(im, jm, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm
        integer(kind=4) :: i,j

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: stagnationtemperatureabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm), intent(in) :: density
        real(kind=8), dimension(im,jm), intent(in) :: momentumx
        real(kind=8), dimension(im,jm), intent(in) :: momentumy
        real(kind=8), dimension(im,jm), intent(in) :: momentumz
        real(kind=8), dimension(im,jm), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm), intent(inout) :: stagnationenthalpyabsdim

!
!     Compute StagnationEnthalpyAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=2)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=2)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=2)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=2)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=2)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=2)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=2)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=2)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=2)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=2)
!        :param: density: density
!        :type:  density: numpy array (rank=2)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=2)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=2)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=2)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=2)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm
        do j = 1,jm
          do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j)
              momentumxabsdim = momentumx(i,j)
              momentumyabsdim = momentumy(i,j)
              momentumzabsdim = momentumz(i,j)
              densitydim = density(i,j)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
              stagnationenthalpyabsdim(i,j) = (gamma*rgaz)/(gamma-1.)*stagnationtemperatureabsdim
          enddo
        enddo

      end


! ----------------------------------------------------------------------
      subroutine stagnationenthalpyabsdim_3d(im, jm, km, rgaz,  &
        gamma,  &
        roio,  &
        aio,  &
        omega,  &
        lref,  &
        coordinatex,  &
        coordinatey,  &
        coordinatez,  &
        density,  &
        momentumx,  &
        momentumy,  &
        momentumz,  &
        energystagnationdensity,  &
        stagnationenthalpyabsdim)
      implicit none
        integer(kind=4), intent(in) :: im,jm,km
        integer(kind=4) :: i,j,k

        real(kind=8) :: velocityyabsdim
        real(kind=8) :: momentumzabsdim
        real(kind=8) :: staticpressuredim
        real(kind=8) :: densitydim
        real(kind=8) :: energystagnationdensitydim
        real(kind=8) :: velocityxabsdim
        real(kind=8) :: stagnationtemperatureabsdim
        real(kind=8) :: machnumberabs
        real(kind=8) :: momentumyabsdim
        real(kind=8) :: momentumxabsdim
        real(kind=8) :: velocityzabsdim
        real(kind=8) :: statictemperaturedim
        real(kind=8), intent(in) :: rgaz
        real(kind=8), intent(in) :: gamma
        real(kind=8), intent(in) :: roio
        real(kind=8), intent(in) :: aio
        real(kind=8), intent(in) :: omega
        real(kind=8), intent(in) :: lref
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatex
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatey
        real(kind=8), dimension(im,jm,km), intent(in) :: coordinatez
        real(kind=8), dimension(im,jm,km), intent(in) :: density
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumx
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumy
        real(kind=8), dimension(im,jm,km), intent(in) :: momentumz
        real(kind=8), dimension(im,jm,km), intent(in) :: energystagnationdensity
        real(kind=8), dimension(im,jm,km), intent(inout) :: stagnationenthalpyabsdim

!
!     Compute StagnationEnthalpyAbsDim with respect to Rgaz, Gamma, roio, aio, omega, Lref, CoordinateX, CoordinateY, CoordinateZ, Density, MomentumX, MomentumY, MomentumZ, EnergyStagnationDensity (rank=3)
!
!        :param: rgaz: rgaz
!        :type:  rgaz: numpy array (rank=3)
!        :param: gamma: gamma
!        :type:  gamma: numpy array (rank=3)
!        :param: roio: roio
!        :type:  roio: numpy array (rank=3)
!        :param: aio: aio
!        :type:  aio: numpy array (rank=3)
!        :param: omega: omega
!        :type:  omega: numpy array (rank=3)
!        :param: lref: lref
!        :type:  lref: numpy array (rank=3)
!        :param: coordinatex: coordinatex
!        :type:  coordinatex: numpy array (rank=3)
!        :param: coordinatey: coordinatey
!        :type:  coordinatey: numpy array (rank=3)
!        :param: coordinatez: coordinatez
!        :type:  coordinatez: numpy array (rank=3)
!        :param: density: density
!        :type:  density: numpy array (rank=3)
!        :param: momentumx: momentumx
!        :type:  momentumx: numpy array (rank=3)
!        :param: momentumy: momentumy
!        :type:  momentumy: numpy array (rank=3)
!        :param: momentumz: momentumz
!        :type:  momentumz: numpy array (rank=3)
!        :param: energystagnationdensity: energystagnationdensity
!        :type:  energystagnationdensity: numpy array (rank=3)
!
! ----------------------------------------------------------------------



!       print *,"im = ",im,", jm = ",jm,", km = ",km
        do k = 1,km
          do j = 1,jm
            do i = 1,im
              energystagnationdensitydim = energystagnationdensity(i,j,k)
              momentumxabsdim = momentumx(i,j,k)
              momentumyabsdim = momentumy(i,j,k)
              momentumzabsdim = momentumz(i,j,k)
              densitydim = density(i,j,k)
              staticpressuredim = (gamma-1.)*(energystagnationdensitydim - (0.5*(momentumxabsdim**2+momentumyabsdim**2+momentumzabsdim**2)/densitydim))
              statictemperaturedim = staticpressuredim/(rgaz*densitydim)
              velocityxabsdim = momentumxabsdim/densitydim
              velocityyabsdim = momentumyabsdim/densitydim
              velocityzabsdim = momentumzabsdim/densitydim
              machnumberabs = sqrt(velocityxabsdim**2+velocityyabsdim**2+velocityzabsdim**2)/sqrt(gamma*rgaz*statictemperaturedim)
              stagnationtemperatureabsdim = statictemperaturedim*(1.+0.5*(gamma-1.)*machnumberabs**2)
              stagnationenthalpyabsdim(i,j,k) = (gamma*rgaz)/(gamma-1.)*stagnationtemperatureabsdim
            enddo
          enddo
        enddo

      end



