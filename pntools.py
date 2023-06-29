from __future__ import annotations

from typing import Tuple
import pprint
import sympy as sp
import numpy as np
import scipy as sc
from prettytable import PrettyTable

C99_NUM_FLUX_RUS_HEADER = """
void num_flux_rus(const float wL[{n}], 
                  const float wR[{n}], 
                  const float vn[{dim}], 
                  float flux[{n}])
"""

C99_SRC_BEAM_BLOCK = """
void src_beam(const float t, const float x[{dim}], float w[{n}])
{{
    float norm;
    float c0;
    float eps = 1e-8F;
    {xyz_blk}
    
    // Source values
{val_blk}
}}
"""

C99_SRC_3D_BEAM_XYZ_BLOCK = """
    // Spatial coefficient for {name}
    c0 = - 0.5F  / ({sigma_xyz:<6.8f}F * {sigma_xyz:<6.8f}F);
    
    norm = (x[0] - {x:<6.8f}F) * (x[0] - {x:<6.8f}F) + 
           (x[1] - {y:<6.8f}F) * (x[1] - {y:<6.8f}F) +
           (x[2] - {z:<6.8f}F) * (x[2] - {z:<6.8f}F);

    float p{beam_idx} = eps + exp(c0 * norm);
"""

C99_SRC_2D_BEAM_XYZ_BLOCK = """
    // Spatial coefficient for {name}
    c0 = - 0.5F  / ({sigma_xyz:<6.8f}F * {sigma_xyz:<6.8f}F);
    
    norm = (x[0] - {x:<6.8f}F) * (x[0] - {x:<6.8f}F) + 
           (x[1] - {y:<6.8f}F) * (x[1] - {y:<6.8f}F);

    float p{beam_idx} = eps + exp(c0 * norm);
"""


def check_dim(dim: int) -> int:
    """
    Check the spatial dimension (dim) of the Pn approximation.

    Args:
        dim (int): Spatial dimension of the Pn approximation.

    Returns:
        int: Validated spatial dimension.

    Raises:
        ValueError: If the dimension is not 1, 2, or 3.
    """
    valid_dims = [1, 2, 3]

    if dim not in valid_dims:
        raise ValueError("Spatial dimension must be 1, 2, or 3")

    return dim


def check_order(order: int) -> np.int64:
    """
    Check the order of the Pn approximation.

    Args:
        order (int): Order of the Pn approximation.

    Returns:
        np.int64: Validated order as a NumPy int64 scalar.

    Raises:
        ValueError: If the order is not a positive integer.
    """
    order = np.int64(order)

    if order <= 0:
        raise ValueError("Order must be a positive integer")

    return order


def check_n(n: int) -> np.int64:
    """
    Check the number of basis functions in the Pn approximation.

    Args:
        n (int): Number of basis functions of the Pn approximation.

    Returns:
        np.int64: Validated number of basis functions as a NumPy int64
        scalar.

    Raises:
        ValueError: If the number of basis functions is not a positive integer.
    """
    n = np.int64(n)

    if n < 0:
        raise ValueError("Number of basis functions must be a positive integer")

    return n


def check_l(l: int) -> np.int64:
    """
    Check the degree of a spherical harmonic.

    Args:
        l (int): Degree of the spherical harmonic.

    Returns:
        np.int64: Validated degree of the spherical harmonic as a NumPy int64
        scalar.

    Raises:
        ValueError: If the degree of the spherical harmonic is not a positive
        integer.
    """
    l = np.int64(l)

    if l < 0:
        raise ValueError(
            "Degree of the spherical harmonic must be a positive or null integer"
        )

    return l


def check_lm(l: int, m: int) -> Tuple[np.int64, np.int64]:
    """
    Check the order of a spherical harmonic.

    Args:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        np.int64: Validated order of the spherical harmonic as a NumPy int64
        scalar.

    Raises:
        ValueError: If the order of the spherical harmonic is not -l <= m <= +l
    """
    l = check_l(l)

    m = np.int64(m)

    if not (-l <= m and m <= l):
        raise ValueError("Order of the spherical harmonic must be  -l <= m <= +l")

    return l, m


def pn_get_n_basis_func(dim: int, order: int) -> np.int64:
    """
    Calculates the number of basis functions for a given dimension and order.

    Args:
        dim (int): Spatial dimension of the Pn approximation.
        order (int): Order of the Pn approximation.

    Returns:
        int: Number of basis functions of the Pn approximation.
    """

    dim = check_dim(dim)
    order = check_order(order)

    if dim == 1:
        n = order + 1

    elif dim == 2:
        n = (order * order) / 2 + (3 * order) / 2 + 1

    elif dim == 3:
        n = (order + 1) * (order + 1)

    n = check_n(n)

    return n


def pn_get_lm(dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the linear index (when serially numbering all basis functions,
    depending on symmetry assumptions (dim)) to a tuple with the degree and
    order of the corresponding real spherical harmonic (S_l^m: l: degree, m:
    order).

    Args:
        n (int): Number of basis functions of the Pn approximation.
        dim (int):  Spatial dimension of the Pn approximation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays: l
        (degree) and m (order).

    Examples:

      |  3D (-l <= m <= l) | 2D (l + m even) |  1D (m = 0) |
      |--------------------|-----------------|-------------|
      |  n: (l,  m)        | n: (l,  m)      |  n: (l,  m) |
      |  0: (0,  0)        | 0: (0,  0)      |  0: (0,  0) |
      |  1: (1, -1)        | 1: (1, -1)      |             |
      |  2: (1,  0)        |                 |  1: (1,  0) |
      |  3: (1,  1)        | 2: (1,  1)      |             |
      |  4: (2, -2)        | 3: (2, -2)      |             |
      |  5: (2, -1)        |                 |             |
      |  6: (2,  0)        | 4: (2,  0)      |  2: (2,  0) |
      |  7: ...            |                 |             |
    """

    dim = check_dim(dim)
    n = check_n(n)

    lin_idx = np.arange(n, dtype=np.int64)

    if dim == 1:
        l = lin_idx
        m = np.zeros(lin_idx.size, dtype=np.int64)

    elif dim == 2:
        l = np.ceil(-3 / 2 + np.sqrt(9 / 4 + 2 * lin_idx))
        m = 2 * (lin_idx - l * (l + 1) / 2) - l

    elif dim == 3:
        l = np.floor(np.sqrt(lin_idx))
        m = lin_idx - l * l - l

    return l, m


def pn_get_lin_idx(dim: int, l: int, m: int) -> np.int64:
    """
    Convert a tuple with degree and order of the corresponding real spherical
    harmonic (S_l^m: l: degree, m: order) to the linear index as a basis
    function (when serially numbering all basis functions, depending on symmetry
    assumptions (Spatial dimension)).

    Args:
        dim (int): Spatial dimension of the Pn approximation.
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        int: Index of the associated basis function.
    """

    idx = np.int64(0)

    dim = check_dim(dim)
    l, m = check_lm(l, m)

    if dim == 1:
        idx = np.int64(l)

    elif dim == 2:
        idx = np.int64((l * (l + 1) / 2 + (m + l) / 2))

    elif dim == 3:
        idx = np.int64((l * l + m + l))

    return idx


def pn_get_idx_to_delete(dim, order):
    # Compute 3D l_lst and m_lst
    n = pn_get_n_basis_func(3, order)
    l_lst, m_lst = pn_get_lm(3, n)

    # Reduction as no dependence on phi
    if dim == 1:
        idx_to_keep = m_lst == 0

    # Reduction as no dependence on mu
    if dim == 2:
        idx_to_keep = np.mod(l_lst + m_lst, 2) == 0

    idx_to_delete = np.where(np.logical_not(idx_to_keep))

    return idx_to_delete


def pn_apply_reduction(dim, order, mat):
    idx_to_delete = pn_get_idx_to_delete(dim, order)

    mat_red = np.delete(mat, idx_to_delete, axis=0)
    mat_red = np.delete(mat_red, idx_to_delete, axis=1)

    return mat_red


def alm(l: int, m: int) -> sp.Expr:
    """
    Builds the coefficient a_lm used in Pn matrices construction.

    Args:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        sp.Expr: The symbolic expression of the coefficient a_lm.

    Remark: See Theorem (7.1.14) equation (7.17) in
            @phdthesis{meltz:tel-01280237,
            TITLE = {{Analyse math{\'e}matique et num{\'e}rique de syst{\`e}mes
                        d'hydrodynamique compressible et de photonique en
                        coordonn{\'e}es polaires}},
            AUTHOR = {Meltz, Bertrand},
            URL = {https://theses.hal.science/tel-01280237},
            NUMBER = {2015SACLS062},
            SCHOOL = {{Universit{\'e} Paris-Saclay}},
            YEAR = {2015},
            MONTH = Nov,
            KEYWORDS = {High-Order Schemes ; Methods of Moments ; Photonics ;
                        Curvilinear Coordinates ; Numerical Analysis ;
                        Hydrodynamics ; Hydrodynamique ; Sch{\'e}ma d'ordre
                        {\'e}lev{\'e} ; Photonique ; Analyse num{\'e}rique ;
                        M{\'e}thode aux moments ; Coordonn{\'e}es curvilignes},
            TYPE = {Theses},
            PDF = {https://theses.hal.science/tel-01280237/file/71383_2015_diffusion.pdf},
            HAL_ID = {tel-01280237},
            HAL_VERSION = {v1},
            }
    """
    l, m = check_lm(l, m)

    l = sp.Integer(l)
    m = sp.Integer(m)

    return sp.sqrt(((l - m) * (l + m)) / ((2 * l + 1) * (2 * l - 1)))


def blm(l: int, m: int) -> sp.Expr:
    """
    Builds the coefficient b_lm used in Pn matrices construction.

    Args:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        sp.Expr: The symbolic expression of the coefficient b_lm.

    Remark: See Theorem (7.1.14) equation (7.17) in
            @phdthesis{meltz:tel-01280237,
            TITLE = {{Analyse math{\'e}matique et num{\'e}rique de syst{\`e}mes
                        d'hydrodynamique compressible et de photonique en
                        coordonn{\'e}es polaires}},
            AUTHOR = {Meltz, Bertrand},
            URL = {https://theses.hal.science/tel-01280237},
            NUMBER = {2015SACLS062},
            SCHOOL = {{Universit{\'e} Paris-Saclay}},
            YEAR = {2015},
            MONTH = Nov,
            KEYWORDS = {High-Order Schemes ; Methods of Moments ; Photonics ;
                        Curvilinear Coordinates ; Numerical Analysis ;
                        Hydrodynamics ; Hydrodynamique ; Sch{\'e}ma d'ordre
                        {\'e}lev{\'e} ; Photonique ; Analyse num{\'e}rique ;
                        M{\'e}thode aux moments ; Coordonn{\'e}es curvilignes},
            TYPE = {Theses},
            PDF = {https://theses.hal.science/tel-01280237/file/71383_2015_diffusion.pdf},
            HAL_ID = {tel-01280237},
            HAL_VERSION = {v1},
            }
    """
    l, m = check_lm(l, m)

    l = sp.Integer(l)
    m = sp.Integer(m)

    return sp.sqrt(((l + m - 1) * (l + m)) / ((2 * l + 1) * (2 * l - 1)))


def delta(i: int, j: int) -> sp.Integer:
    """
    Returns the value of the Kronecker delta symbol.

    Args:
        i (sp.Integer): The first index.
        j (sp.Integer): The second index.

    Returns:
        sp.Integer: The value of the Kronecker delta symbol.

    """
    return sp.Integer(sp.KroneckerDelta(i, j))


def pn_get_jx_ij(li: int, mi: int, lj: int, mj: int):
    """
    Calculates the entries of the Jx angular matrix. See (equation 7.20) in the
    PhD thesis "meltz:tel-01280237".

    Args:
        li (int)
        mi (int)
        lj (int)
        mj (int)

    Returns:
        sp.Expr: The Jx_{i,j} matrix entry.
    """

    c1 = (1 / 2) * sp.sign(mi + 0) * (1 + (sp.sqrt(2) - 1) * delta(mi, 1))
    c2 = (1 / 2) * sp.sign(mi + 1) * (1 + (sp.sqrt(2) - 1) * delta(mi, 0))

    # Note: Added dummy signs to improve readability
    d1 = -delta(lj, li - 1) * delta(mj, mi - 1) * blm(li, +mi)
    d1 += delta(lj, li + 1) * delta(mj, mi - 1) * blm(lj, -mj)

    d2 = +delta(lj, li - 1) * delta(mj, mi + 1) * blm(li, -mi)
    d2 -= delta(lj, li + 1) * delta(mj, mi + 1) * blm(lj, +mj)

    return c1 * d1 + c2 * d2


def pn_get_jy_ij(li: int, mi: int, lj: int, mj: int):
    """
    Calculates the entries of the Jy angular matrix. See (equation 7.20) in the
    PhD thesis "meltz:tel-01280237".

    Args:
        li (int)
        mi (int)
        lj (int)
        mj (int)

    Returns:
        sp.Expr: The Jy_{i,j} matrix entry.
    """

    c1 = (1 / 2) * sp.sign(mi) * (1 - delta(mi, 1))

    c2 = (
        (1 / 2)
        * sp.sign(mi + 1 / 2)
        * (1 + (sp.sqrt(2) - 1) * (delta(mi, 0) + delta(mi, -1)))
    )

    # Note: Added dummy signs to improve readability
    d1 = +delta(lj, li - 1) * delta(mj, -(mi - 1)) * blm(li, +mi)
    d1 -= delta(lj, li + 1) * delta(mj, -(mi - 1)) * blm(lj, +mj)

    d2 = +delta(lj, li - 1) * delta(mj, -(mi + 1)) * blm(li, -mi)
    d2 -= delta(lj, li + 1) * delta(mj, -(mi + 1)) * blm(lj, -mj)

    return c1 * d1 + c2 * d2


def pn_get_jz_ij(li: int, mi: int, lj: int, mj: int):
    """
    Calculates the entries of the Jz angular matrix. See (equation 7.18) in the
    PhD thesis "meltz:tel-01280237".

    Args:
        li (int)
        mi (int)
        lj (int)
        mj (int)

    Returns:
        sp.Expr: The Jz_{i,j} matrix entry.
    """

    # Note: Added dummy signs to improve readability
    c1 = +delta(lj, li - 1) * delta(mj, mi) * alm(li, mi)
    c1 += delta(lj, li + 1) * delta(mj, mi) * alm(lj, mj)

    return c1


def pn_get_3d_jx_jy_jz(order: int) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Calculates the angular matrices Jx, Jy, and Jz of the 3D Pn approximation.

    Args:
        order (int): Order of the Pn approximation.

    Returns:
        Tuple[sp.Matrix, sp.Matrix, sp.Matrix]: Angular matrices Jx, Jy, and Jz
        of the Pn approximation in 3D.
    """

    dim = 3
    order = check_order(order)
    n = pn_get_n_basis_func(dim, order)

    jx = sp.zeros(n, n)
    jy = sp.zeros(n, n)
    jz = sp.zeros(n, n)

    for li in range(order + 1):
        for mi in range(-li, li + 1):
            for lj in range(order + 1):
                for mj in range(-lj, lj + 1):
                    i = pn_get_lin_idx(dim, li, mi)
                    j = pn_get_lin_idx(dim, lj, mj)

                    jx[i, j] = pn_get_jx_ij(li, mi, lj, mj)
                    jy[i, j] = pn_get_jy_ij(li, mi, lj, mj)
                    jz[i, j] = pn_get_jz_ij(li, mi, lj, mj)

    return jx, jy, jz


def pn_get_2d_jx_jy(order: int) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Calculates the angular matrices Jx, Jy, of the 2D Pn approximation.

    Args:
        order (int): Order of the Pn approximation.

    Returns:
        Tuple[sp.Matrix, sp.Matrix] : Angular matrices Jx, Jy of the Pn
        approximation in 2D.
    """

    dim = 3
    order = check_order(order)
    n = pn_get_n_basis_func(dim, order)

    jx = sp.zeros(n, n)
    jy = sp.zeros(n, n)

    # cnorm = sp.sqrt(1 - sp.cos(sp.pi / 4) * sp.cos(sp.pi / 4))
    cnorm = 1

    for li in range(order + 1):
        for mi in range(-li, li + 1):
            for lj in range(order + 1):
                for mj in range(-lj, lj + 1):
                    i = pn_get_lin_idx(dim, li, mi)
                    j = pn_get_lin_idx(dim, lj, mj)

                    jx[i, j] = pn_get_jx_ij(li, mi, lj, mj) / cnorm
                    jy[i, j] = pn_get_jy_ij(li, mi, lj, mj) / cnorm

    return jx, jy


def pn_get_c99_code_num_flux_rus(dim: int, flux: sp.Expr) -> str:
    """
    Generates a C99 function code that computes the numerical flux of Rusanov
    of Pn approximation.

    Args:
        dim (int): Spatial dimension of the Pn approximation.
        order (int) : Order of the Pn approximation

    Returns:
        str: C99 function code that computes the numerical flux of Rusanov.
    """

    from sympy.codegen.ast import real, float32, Assignment
    from sympy.printing.c import C99CodePrinter

    # Convert symbolic expression to numeric
    flux = sp.nsimplify(flux, tolerance=1e-6, rational=False)
    flux = flux.evalf()

    # Create a C99 code printer with type aliases set to 'float32'
    p = C99CodePrinter(settings={"type_aliases": {real: float32}})

    # Apply common subexpression elimination algorithm
    sub_exprs, simplified = sp.cse(flux)

    # Write C99 function header
    header = C99_NUM_FLUX_RUS_HEADER.format(n=flux.shape[0], dim=dim)

    # Write C99 function's body internal variables
    code = "{\n"
    code += "\n".join(
        [
            "const float {}".format(p._print(Assignment(var, sub_expr)))
            for var, sub_expr in sub_exprs
        ]
    )

    M = sp.MatrixSymbol("flux", *flux.shape)

    # Write numerical flux expression
    code += "\n{}\n".format(p._print(Assignment(M, simplified[0])))

    # Write closing brace for the function's body
    code += "}\n"

    code = header + p.indent_code(code)
    # Indent the generated code
    return code


def pn_get_expr_num_flux_rus(order: int) -> Tuple[str, str]:
    """
    Compute the numerical flux of Rusanov for 2D and 3D Pn approximation.

    Args:
        order (int) : Order of the Pn approximation

    Returns:
        Tuple[str, str]: C99 function code that computes the numerical flux of
                         Rusanov (2D and 3D)
    """

    # Compute 3D Rusanov numerical flux
    dim = 3
    dim = check_dim(dim)

    n = pn_get_n_basis_func(dim, order)
    jx, jy, jz = pn_get_3d_jx_jy_jz(order)

    # Set maximum wave's speed
    vmax = 1

    wL = sp.Matrix([sp.symbols("{}[{}]".format("wL", k)) for k in range(n)])
    wR = sp.Matrix([sp.symbols("{}[{}]".format("wR", k)) for k in range(n)])
    vn = sp.Matrix([sp.symbols("{}[{}]".format("vn", k)) for k in range(dim)])

    flux_3d = 0.5 * (
        (vn[0] * jx * (wL + wR) + vn[1] * jy * (wL + wR) + vn[2] * jz * (wL + wR))
        - vmax * (wR - wL)
    )

    # Compute 2D Rusanov numerical flux
    dim = 2
    dim = check_dim(dim)

    n = pn_get_n_basis_func(dim, order)

    jx = sp.Matrix(pn_apply_reduction(dim, order, jx))
    jy = sp.Matrix(pn_apply_reduction(dim, order, jy))

    wL = sp.Matrix([sp.symbols("{}[{}]".format("wL", k)) for k in range(n)])
    wR = sp.Matrix([sp.symbols("{}[{}]".format("wR", k)) for k in range(n)])
    vn = sp.Matrix([sp.symbols("{}[{}]".format("vn", k)) for k in range(dim)])

    flux_2d = 0.5 * (
        (vn[0] * jx @ (wL + wR) + vn[1] * jy @ (wL + wR)) - vmax * (wR - wL)
    )

    return flux_2d, flux_3d


def pn_src_beam(th, ph, dim, intensity, th_0, ph_0, sigma):
    th2 = (th - th_0) * (th - th_0)

    ph2 = (ph - ph_0) * (ph - ph_0)

    # WARNING TO CONFIRM: In two dimensions there is no phi
    if dim == 2:
        ph2 = 0

    return intensity * np.exp(-(th2 + ph2) / (sigma * sigma))


def pn_src_proj_3d_real(th, ph, l, m, src_func, *args):
    # WARNING: In scipy args m and l are reversed at function call
    ylm = sc.special.sph_harm(m, l, th, ph)
    return np.real(np.conj(ylm) * src_func(th, ph, *args))


def pn_src_proj_2d_real(th, l, m, src_func, *args):
    # Defined 2D spherical harmonics have phi=pi/2
    # WARNING: In scipy args m and l are reversed at function call
    ph = 0.5 * np.pi
    ylm = sc.special.sph_harm(m, l, th, ph)
    return np.real(np.conj(ylm) * src_func(th, ph, *args))


# Returns source coef for the approximated Gaussian beam (3D)
def pn_get_src_proj(order: int, src_func, **kwargs) -> np.array:
    # Unpack extra keyworded arguments for source function call
    args = tuple(k for k in kwargs.values())

    # Buffer for two-dimensional source projection
    dim = 2
    args_2d = (dim, *args)
    src_proj_2d = np.zeros(pn_get_n_basis_func(dim, order), dtype=np.float32)

    # Buffer for three-dimensional source projection
    dim = 3
    args_3d = (dim, *args)
    src_proj_3d = np.zeros(pn_get_n_basis_func(dim, order), dtype=np.float32)

    for l in range(order + 1):
        for m in range(-l, l + 1):
            # Compute 2D source projection
            dim = 2
            i = pn_get_lin_idx(dim, l, m)

            src_proj_2d[i], _ = sc.integrate.quad(
                pn_src_proj_2d_real,
                0.0,
                2.0 * np.pi,
                args=(l, m, src_func, *args_2d),
            )

            # Compute 3D source projection
            dim = 3
            i = pn_get_lin_idx(dim, l, m)
            src_proj_3d[i], _ = sc.integrate.dblquad(
                pn_src_proj_3d_real,
                0.0,
                2 * np.pi,
                0.0,
                np.pi,
                args=(l, m, src_func, *args_3d),
            )

    # Replace small values with 0
    src_proj_2d[np.abs(src_proj_2d) < 1e-6] = 0.0
    src_proj_3d[np.abs(src_proj_3d) < 1e-6] = 0.0

    src_dict = {"2D": src_proj_2d, "3D": src_proj_3d}

    return src_dict


def pn_get_c99_code_src_beam(dim, order, src_lst: list[dict]) -> str:
    n = pn_get_n_basis_func(dim, order)
    # indent = 8 * " "

    # code = C99_SRC_BEAM_HEADER.format(dim=dim, n=n)

    vals_lst = []

    nb_src = len(src_lst)

    # Spatial C99 block
    if dim == 2:
        tag = "2D"
        xyz_tpl = C99_SRC_2D_BEAM_XYZ_BLOCK

    else:
        tag = "3D"
        xyz_tpl = C99_SRC_3D_BEAM_XYZ_BLOCK

    xyz_blk = "".join(
        [
            xyz_tpl.format(
                name=src.get("name"),
                sigma_xyz=src.get("sigma_xyz"),
                x=src.get("x"),
                y=src.get("y"),
                z=src.get("z"),
                beam_idx=k,
            )
            for k, src in enumerate(src_lst)
        ]
    )

    # Source value C99 block
    get_sign = lambda num: "+" if num > 0 else "-"

    val_blk = ""
    for j in range(n):
        line = "w[{}] = ".format(j)

        for i in range(len(src_lst)):
            t0 = src_lst[i].get("values").get(tag)[j]

            # Absolute value is taken since we reinject the sign up front
            line += " {} {:<6.8f}F * p{}".format(get_sign(t0), abs(t0), i)

        val_blk += "{}{};\n".format(1 * "    ", line)

    # Complete function code
    code = C99_SRC_BEAM_BLOCK.format(
        dim=dim,
        n=n,
        xyz_blk=xyz_blk,
        val_blk=val_blk,
    )

    # Write brace closing function's body
    code += "\n\n"

    return code

def pprint_src(order, src_lst):
    n_max_3d = pn_get_n_basis_func(3, order)
    n_max_2d = pn_get_n_basis_func(2, order)

    col_n_3d = [k for k in range(n_max_3d)]
    col_n_2d = [k for k in range(n_max_2d)]
    col_l_3d, col_m_3d = pn_get_lm(3, n_max_3d)
    col_l_2d, col_m_2d = pn_get_lm(2, n_max_2d)


    # Cast to int for display
    col_l_2d = [int(val) for val in col_l_2d]
    col_m_2d = [int(val) for val in col_m_2d]
    
    col_l_3d = [int(val) for val in col_l_3d]
    col_m_3d = [int(val) for val in col_m_3d]

    # 2D Base array for beam coefficient
    tab_2d = PrettyTable()
    tab_2d.add_column("N", col_n_2d, align="r", valign="t")
    tab_2d.add_column("l", col_l_2d, align="r", valign="t")
    tab_2d.add_column("m", col_m_2d, align="r", valign="t")
    
    # 3D Base array for beam coefficient
    tab_3d = PrettyTable()
    tab_3d.add_column("N", col_n_3d, align="r", valign="t")
    tab_3d.add_column("l", col_l_3d, align="r", valign="t")
    tab_3d.add_column("m", col_m_3d, align="r", valign="t")

    for src in src_lst:
        col_vals_2d = vals = src.get("values").get("2D").tolist()
        # Set int/float format
        col_vals_2d = ["{:<6.8f}".format(val) for val in col_vals_2d]

        tab_2d.add_column(src.get("name"), col_vals_2d, align="r", valign="t")
    
    print("# 2D coefficients:")
    print(tab_2d)

    # Build pretty table for 3D coefficients
    for src in src_lst:
        col_vals_3d = vals = src.get("values").get("3D").tolist()

        # Set float format
        col_vals_3d = ["{:<6.8f}".format(val) for val in col_vals_3d]

        tab_3d.add_column(
            src.get("name"),
            col_vals_3d,
            align="r",
            valign="t",
        )
        
    print("# 3D coefficients:")
    print(tab_3d)

def pn_build_c99_code(order: int, src_lst: list[dict]):
    # Generate Rusanov numerical flux
    flux_2d, flux_3d = pn_get_expr_num_flux_rus(order)

    # Write include guard macro
    code = "#ifndef P{order}_CL\n#define P{order}_CL\n\n".format(
        order=order,
    )

    # Write 2D selector macro
    code += "#ifdef IS_2D\n"

    # Write 2D source function code
    code += pn_get_c99_code_src_beam(2, order, src_lst)

    # Write 2D Rusanov numerical flux C99 function code
    code += pn_get_c99_code_num_flux_rus(2, flux_2d)

    # Write 3D selector macro
    code += "\n#else\n"

    # Write 3D source function code
    code += pn_get_c99_code_src_beam(3, order, src_lst)

    # Write 3D IRusanov numerical flux C99 function code
    code += pn_get_c99_code_num_flux_rus(3, flux_3d)

    # End dimension selector macro
    code += "\n#endif\n"

    # End include guard macro
    code += "#endif\n"

    return code


if __name__ == "__main__":
    # PN order
    order = 1

    # Velocity part intensity
    intensity = 1.0

    # Velocity part sigma
    sig = 0.1

    src_lst = [
        {
            "name": "beam_0",
            "values": pn_get_src_proj(
                order,
                pn_src_beam,
                intensity=intensity,
                th_0=0,
                ph_0=0,
                sigma=sig,
            ),
            # Beam spatial position
            "x": 0.25,
            "y": 0.5,
            "z": 0.5,
            # Beam spatial tickness
            "sigma_xyz": 0.005,
        },
        {
            "name": "beam_1",
            "values": pn_get_src_proj(
                order,
                pn_src_beam,
                intensity=intensity,
                th_0=0,
                ph_0=0,
                sigma=sig,
            ),
            # Beam spatial position
            "x": 0.5,
            "y": 0.75,
            "z": 0.5,
            # Beam spatial tickness
            "sigma_xyz": 0.005,
        },
    ]

# Print coefficients
pprint_src(order, src_lst)

code = pn_build_c99_code(order, src_lst)
print(code)
