# -*- coding: utf-8 -*-
import math
import numpy
import torch
from torch import Tensor
import torch.nn
import torch.nn.functional
import torch.linalg

from mlmsdft.nn.functional import ScalarFunction, MatrixFunction
from pyscf.dft import libxc


__all__ = [
    "lda_x_dirac",
    "lda_c_chachiyo",
    "lda_xc_dirac_chachiyo",
    "lda_xc_dirac_chachiyo_unpolarized",
    "complementary_sr_lda_unpolarized",
    "complementary_sr_lda_spin_polarized",
]

# Cₓ = (3/4) (3/pi)¹ᐟ³ = 0.7386 from Dirac's exchange-energy, Eqn. (6.1.20) in [Parr&Yang]
Cx_Dirac = 3.0/4.0 * pow(3.0/math.pi, 1.0/3.0)
# Cₓ from the "Gaussian" approximation in Eqn. (6.5.25) of [Parr&Yang]
Cx_Gaussian = 0.7937

class _LDAExchangeDirac(ScalarFunction):
    # The prefactor Cₓ for the exchange energy.
    # Depending on whether the exchange energy is calculated from the spin density or the
    # total density, the prefactor is different. Cx is used for spin densities and Cx_Dirac
    # for the total density. For a closed shell, where ρᵅ=ρᵝ=ρ/2 such that ρ=ρᵅ+ρᵝ=2 ρᵅ, we
    # have
    #   Cx_Dirac (2 ρᵅ)⁴ᐟ³ = Cx [(ρᵅ)⁴ᐟ³ + (ρᵝ)⁴ᐟ³]
    # which means that
    #   Cx = 2¹ᐟ³ Cx_Dirac
    Cx = pow(2.0, 1.0/3.0) * Cx_Dirac

    @staticmethod
    def value(scalar_density: Tensor) -> Tensor:
        """
        compute the energy density for the exchange-like part of the electron-electron
        repulsion for a scalar density,

            XED[ρ](r) = -Cₓ ρ(r)⁴ᐟ³

        NOTE: At odds with the usual definition of the exchange energy density,
        (εₓ,ᵢⱼ(r) ∝ ρ(r)¹ᐟ³), XED contains an additional factor of ρ(r)
        (XED(r) ∝ ρ(r)⁴ᐟ³), since the exchange energy is calculated
        as X[D] = ∫ XED(r) dr rather than X[ρ] = ∫ ρ(r) εₓ(r) dr.

        :param scalar_density: density ρ
        :type scalar_density: arbitrary Tensor

        :return xed: exchange energy density
        :rtype xed: Tensor with same shape as input
        """
        # Since the matrix density is positive definite, the argument
        # `scalar_density` should always be positive. However, due to
        # finite numerical precision negative values close to 0 might occur.
        # To avoid NaN's in pow(rho,4/3.0) we take the absolute value.
        rho = torch.abs(scalar_density)
        # exchange energy is always negative
        xed = (-1) * _LDAExchangeDirac.Cx * torch.pow(rho, 4.0/3.0)
        return xed

    @staticmethod
    def derivative1(scalar_density: Tensor) -> Tensor:
        """
        Derivative of exchange energy density w/r/t density

            XED'[ρ] = d(XED[ρ])/dρ = -4/3 Cₓ ρ¹ᐟ³
        """
        rho = torch.abs(scalar_density)
        xed_deriv1 = -4.0/3.0 * _LDAExchangeDirac.Cx * torch.pow(rho, 1.0/3.0)
        return xed_deriv1


def lda_x_dirac(
        matrix_density: Tensor,
        # `grad_dummy` and `lapl_dummy` arguments are ignored.
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None
    ) -> Tensor:
    """
    Multi-state exchange energy according to the local density approximation
    (eqn. 6.5.29 in Ref. [Yang&Parr]),

    X[Dᵅ(r)] = Cₓ ∫ Dᵅ(r)⁴ᐟ³ dr = ∫ XED(r) dr

    Dᵅ(r)⁴ᐟ³ is a fractional matrix-power of Dᵅ(r), which is calculated by diagonalizing Dᵅ.

    The value of the prefactor Cₓ = 2¹ᐟ³ 0.7386 is taken from Dirac's approximation in
    Eqn. (6.1.20) of chapter 6 in Ref. [Yang&Parr]

    References
    ----------
    [Yang&Parr] Parr & Yang (1989), "Density Functional Theory of Atoms and Molecules".

    :param matrix_density: matrix density, Dᵅᵢⱼ(r), Dᵝᵢⱼ(r) or Dᵢⱼ/2
            D[...,i,j] = Dᵅᵢⱼ
    :type matrix_density: Tensor of shape (...,n,n)

    :return xed: exchange energy density
        xed[...,i,j] = Cₓ (Dᵅ(r)⁴ᐟ³)ᵢⱼ
    :rtype xed: Tensor of shape (...,n,n)

    where the indices i,j=1,...,n run over the number of electronic states.
    """
    return MatrixFunction.apply(_LDAExchangeDirac, matrix_density)


class _LDACorrelationChachiyo(ScalarFunction):
    @staticmethod
    def functional_parameters(spin: int):
        """ parameters in Chachiyo's functional """
        assert spin in [0,1]
        # Parameters of Chachiyo's functional from Eqn.(3) of [Chachiyo]
        a = (math.log(2.0)-1.0)/(2*math.pi**2)
        # b from Eqn.(3) for the paramagnetic part εᶜ₀
        b_paramagnetic = 20.4562557
        # b from Eqn.(12) for the ferromagnetic part εᶜ₁
        b_ferromagnetic = 27.4203609

        # The parameter b is different from paramagnetic or ferromagnetic densities.
        if spin == 1:
            a = 0.5 * a
            b = b_ferromagnetic
        else:
            b = b_paramagnetic
        b1 = pow(4.0/3.0*math.pi, 1.0/3.0) * b
        b2 = pow(4.0/3.0*math.pi, 2.0/3.0) * b
        return (a, b1, b2)

    @staticmethod
    def value(scalar_density: Tensor, spin: int) -> Tensor:
        """
        compute the energy density for the correlation-like part of the electron-electron
        repulsion for a scalar density,

            CED[ρ](r) = a log(1 + b₁ ρ(r)¹ᐟ³ + b₂ ρ(r)²ᐟ³ ) ρ(r)

        NOTE: At odds with the usual definition of the correlation energy density,
        CED contains an additional factor of ρ(r), since the correlation energy is calculated as
        C[D] = ∫ CED(r) dr rather than C[ρ] = ∫ ρ(r) εᶜ(r) dr.

        :param scalar_density: density ρ
        :type scalar_density: arbitrary Tensor

        :return ced: correlation energy density
        :rtype ced: Tensor with same shape as input
        """
        a, b1, b2 = _LDACorrelationChachiyo.functional_parameters(spin)
        # Since the matrix density is positive definite, the argument
        # `scalar_density` should always be positive. However, due to
        # finite numerical precision negative values close to 0 might occur.
        # To avoid NaN's in pow(rho,4/3.0) we take the absolute value
        # and add a tiny positive number.
        rho = torch.abs(scalar_density) + 1.0e-15
        # In terms of the density the correlation energy becomes
        #  εᶜ(ρ) = a log( 1 + b1 ρ¹ᐟ³ + b2 ρ²ᐟ³ )
        arg = 1.0 + b1 * torch.pow(rho, 1.0/3.0) + b2 * torch.pow(rho, 2.0/3.0)
        epsilon_c = a * torch.log(arg)
        # Multiply the correlation energy per particle by the particle density
        # to get the correlation energy density (CED(r))
        ced = epsilon_c * rho
        return ced

    @staticmethod
    def derivative1(scalar_density: Tensor, spin: int) -> Tensor:
        """
        Derivative of correlation energy density w/r/t density

            CED'[ρ] = d(CED[ρ])/dρ = d(εᶜ(ρ))/dρ ρ + εᶜ(ρ)
        """
        a, b1, b2 = _LDACorrelationChachiyo.functional_parameters(spin)
        rho = torch.abs(scalar_density) + 1.0e-15
        # d(ced)/dρ = d(εᶜ(ρ))/dρ ρ + εᶜ(ρ)
        arg = 1.0 + b1 * torch.pow(rho, 1.0/3.0) + b2 * torch.pow(rho, 2.0/3.0)
        ced_deriv1 = a * (
            (1.0/3.0 * b1 * torch.pow(rho, 1.0/3.0) + 2.0/3.0 * b2 * torch.pow(rho, 2.0/3.0)
            ) / arg + torch.log(arg))
        return ced_deriv1


def lda_c_chachiyo(
        matrix_density: Tensor,
        # `grad_dummy` and `lapl_dummy` arguments are ignored.
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
        spin=0
    ) -> Tensor:
    """
    Multi-state correlation energy according to the local density approximation.

    For a single electronic state it reduces to the correlation energy of the uniform
    electron gas. The functional form from [Chachiyo] is a simple and elegant parameterization
    of the correlation energy per electron of the uniform electron gas.
    It recovers the exact high density limit and fits the quantum Monte-Carlo results of
    [Ceperley&Alder] in the medium density range rather well.

    Taking the paramagnetic part of the correlation energy (spin polarization = 0) and
    replacing the electron density ρ(r) with the density matrix D(r), the multistate extension
    of the Chachiyo functional (Eqn.8 in [Chachiyo]) can be written in the following form:

        C[D(r)] = a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr

                = ∫ CED(r) dr

    with

        a = (log(2)-1)/(2 π²) = -0.01554534543482745
        b = 20.4562557 (paramagnetic)
        b₁ = (4π/3)¹ᐟ³ b = 32.975319597703546
        b₂ = (4π/3)²ᐟ³ b = 53.155949872619715

    `f[D] = a log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³) D(r)` is a matrix funtional,
    which is calculated by diagonalizing D and applying the function f to the eigenvalues.

    References
    ----------
    [Chachiyo] T. Chachiyo (2016), J. Chem. Phys. 145, 2
        "Communication: Simple and accurate uniform electron gas correlation energy for the full range of densities"
    [Ceperley&Alder] D. Ceperley, B. Alder (1980), Phys. Rev. Lett., 45, 7, 566.
        "Ground state of the electron gas by a stochastic method"

    :param matrix_density: matrix density summed over spins, Dᵢⱼ = Dᵅᵢⱼ(r) + Dᵝᵢⱼ(r)
        matrix_density[...,i,j] = Dᵢⱼ
    :type matrix_density: Tensor of shape (...,n,n)

    :param spin: The spin parameter determines whether the paramagnetic (spin=0)
        or ferromagnetic (spin=1) correlation energy is calculated.
    :type spin: int

    :return: Electron correlation energy density CED
    :rtype: Tensor of same shape as input `matrix_density`.
    """
    return MatrixFunction.apply(_LDACorrelationChachiyo, matrix_density, spin)


class _LDAXCDiracChachiyo(ScalarFunction):
    @staticmethod
    def value(scalar_density: Tensor, spin: int = 0) -> Tensor:
        return (
            _LDAExchangeDirac.value(scalar_density) +
            _LDACorrelationChachiyo.value(scalar_density, spin)
        )

    @staticmethod
    def derivative1(scalar_density: Tensor, spin: int = 0) -> Tensor:
        return (
            _LDAExchangeDirac.derivative1(scalar_density) +
            _LDACorrelationChachiyo.derivative1(scalar_density, spin)
        )


def lda_xc_dirac_chachiyo(
        matrix_density: Tensor,
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
        spin=0
    ) -> Tensor:
    """Fused Dirac exchange plus Chachiyo correlation matrix functional."""
    return MatrixFunction.apply(_LDAXCDiracChachiyo, matrix_density, spin)


class _LDAXCDiracChachiyoUnpolarized(ScalarFunction):
    @staticmethod
    def value(scalar_density: Tensor, spin: int = 0) -> Tensor:
        return (
            2.0 * _LDAExchangeDirac.value(scalar_density / 2.0) +
            _LDACorrelationChachiyo.value(scalar_density, spin)
        )

    @staticmethod
    def derivative1(scalar_density: Tensor, spin: int = 0) -> Tensor:
        return (
            _LDAExchangeDirac.derivative1(scalar_density / 2.0) +
            _LDACorrelationChachiyo.derivative1(scalar_density, spin)
        )


def lda_xc_dirac_chachiyo_unpolarized(
        matrix_density: Tensor,
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
        spin=0
    ) -> Tensor:
    """Fused spin-unpolarized LDA XC matrix functional, 2*X(D/2)+C(D)."""
    return MatrixFunction.apply(_LDAXCDiracChachiyoUnpolarized, matrix_density, spin)


class _ComplementarySRLDAUnpolarized(ScalarFunction):
    """LibXC-backed complementary scalar srLDA.

    The correlation branch follows the implementation plan's initial
    PW_MOD-minus-PMGB06 choice.  The exact ``omega=0`` and ``omega=inf``
    branches avoid LibXC's range-parameter sentinel behavior.
    """

    @staticmethod
    def _evaluate(scalar_density: Tensor, omega: float):
        values = torch.clamp(scalar_density.detach(), min=0.0)
        flat = values.reshape(-1).cpu().numpy()
        # LDA uses one spin channel and one density variable.  The compact
        # ``(1, npoints)`` layout is accepted by both the local PySCF bridge
        # and the newer LibXC bridge on the GPU environment.
        rho = flat[numpy.newaxis, :]

        if omega == math.inf:
            value = numpy.zeros_like(flat)
            derivative = numpy.zeros_like(flat)
        else:
            if omega == 0.0:
                x_code, x_omega = "LDA_X", None
            else:
                x_code, x_omega = "LDA_X_ERF", omega
            x_exc, x_vrho, *_ = libxc.eval_xc(
                x_code, rho, spin=0, deriv=1, omega=x_omega
            )
            if omega == 0.0:
                c_exc, c_vrho, *_ = libxc.eval_xc(
                    "LDA_C_PW_MOD", rho, spin=0, deriv=1
                )
                exc = x_exc
                vrho = x_vrho[0]
                cexc = c_exc
                cvrho = c_vrho[0]
            else:
                c_pw, c_pw_vrho, *_ = libxc.eval_xc(
                    "LDA_C_PW_MOD", rho, spin=0, deriv=1
                )
                c_lr, c_lr_vrho, *_ = libxc.eval_xc(
                    "LDA_C_PMGB06", rho, spin=0, deriv=1, omega=omega
                )
                exc = x_exc
                vrho = x_vrho[0]
                cexc = c_pw - c_lr
                cvrho = c_pw_vrho[0] - c_lr_vrho[0]
            value = flat * (exc + cexc)
            derivative = vrho + cvrho

        value = torch.as_tensor(value, dtype=scalar_density.dtype, device=scalar_density.device)
        derivative = torch.as_tensor(derivative, dtype=scalar_density.dtype, device=scalar_density.device)
        return value.reshape_as(scalar_density), derivative.reshape_as(scalar_density)

    @staticmethod
    def value(scalar_density: Tensor, omega: float = 0.0) -> Tensor:
        value, _derivative = _ComplementarySRLDAUnpolarized._evaluate(scalar_density, omega)
        return value

    @staticmethod
    def derivative1(scalar_density: Tensor, omega: float = 0.0) -> Tensor:
        _value, derivative = _ComplementarySRLDAUnpolarized._evaluate(scalar_density, omega)
        return derivative


class _ComplementarySRExchange(ScalarFunction):
    """Unpolarized scalar short-range exchange component."""

    @staticmethod
    def _evaluate(scalar_density: Tensor, omega: float):
        values = torch.clamp(scalar_density.detach(), min=0.0)
        flat = values.reshape(-1).cpu().numpy()
        if omega == math.inf:
            value = numpy.zeros_like(flat)
            derivative = numpy.zeros_like(flat)
        else:
            code = "LDA_X" if omega == 0.0 else "LDA_X_ERF"
            rho = flat[numpy.newaxis, :]
            exc, vrho, *_ = libxc.eval_xc(
                code, rho, spin=0, deriv=1,
                **({} if omega == 0.0 else {"omega": omega}),
            )
            value = flat * exc
            derivative = vrho[0]
        value = torch.as_tensor(value, dtype=scalar_density.dtype, device=scalar_density.device)
        derivative = torch.as_tensor(derivative, dtype=scalar_density.dtype, device=scalar_density.device)
        return value.reshape_as(scalar_density), derivative.reshape_as(scalar_density)

    @staticmethod
    def value(scalar_density: Tensor, omega: float = 0.0) -> Tensor:
        value, _derivative = _ComplementarySRExchange._evaluate(scalar_density, omega)
        return value

    @staticmethod
    def derivative1(scalar_density: Tensor, omega: float = 0.0) -> Tensor:
        _value, derivative = _ComplementarySRExchange._evaluate(scalar_density, omega)
        return derivative


def complementary_sr_lda_unpolarized(
        matrix_density: Tensor,
        omega: float,
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
    ) -> Tensor:
    """Complementary unpolarized srLDA matrix functional.

    For finite ``omega`` this uses erfc exchange plus the PW_MOD minus PMGB06
    correlation complement.  The result is an energy density per volume and
    is lifted to the matrix domain with the same divided-difference machinery
    as the existing LMDA functionals.
    """
    if omega < 0.0:
        raise ValueError(f"omega must be non-negative, got {omega!r}")
    return MatrixFunction.apply(_ComplementarySRLDAUnpolarized, matrix_density, omega)


def complementary_sr_lda_spin_polarized(
        spin_matrix_density: Tensor,
        omega: float,
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
    ) -> Tensor:
    """Collinear spin-polarized complementary srLDA matrix functional.

    Exchange is spin-scaled through the two diagonal spin blocks.  The
    correlation complement is evaluated on the spin-traced matrix density,
    matching the existing polarized LMDA convention for scalar correlation
    generators while retaining the physically distinct alpha/beta exchange.
    """
    if omega < 0.0:
        raise ValueError(f"omega must be non-negative, got {omega!r}")
    if spin_matrix_density.size()[:2] != (2, 2):
        raise ValueError("spin_matrix_density must have leading spin-block shape (2, 2)")
    alpha = spin_matrix_density[0, 0]
    beta = spin_matrix_density[1, 1]
    total = alpha + beta
    # Levy-Perdew spin scaling: E_x[nα,nβ] = 1/2 E_x^u[2nα]
    # + 1/2 E_x^u[2nβ].  This makes the closed-shell limit exactly equal
    # to the unpolarized generator evaluated on nα+nβ.
    exchange_alpha = 0.5 * MatrixFunction.apply(_ComplementarySRExchange, 2.0 * alpha, omega)
    exchange_beta = 0.5 * MatrixFunction.apply(_ComplementarySRExchange, 2.0 * beta, omega)
    exchange_total = MatrixFunction.apply(_ComplementarySRExchange, total, omega)
    correlation = complementary_sr_lda_unpolarized(total, omega) - exchange_total
    return exchange_alpha + exchange_beta + correlation
