"""Arblib wrapper using ctypes."""

import ctypes
import importlib.resources
import math
import platform
from typing import Any, Tuple, Union

# pylint: disable=protected-access

if platform.system() == "Windows":
    with importlib.resources.path(
        "tmlt.core.ext.lib", "libarb.dll"
    ) as _arb_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libflint.dll.a"
    ) as _flint_path:
        arblib = ctypes.WinDLL(str(_arb_path))  # type: ignore
        flintlib = ctypes.WinDLL(str(_flint_path))  # type: ignore
elif platform.system() == "Linux":
    with importlib.resources.path(
        "tmlt.core.ext.lib", "libgmp.so.10.4.1"
    ) as _gmp_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libmpfr.so.6.1.0"
    ) as _mpfr_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libflint.so.17.0.0"
    ) as _flint_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libarb.so.2.14.0"
    ) as _arb_path:
        ctypes.CDLL(str(_gmp_path))
        ctypes.CDLL(str(_mpfr_path))
        flintlib = ctypes.CDLL(str(_flint_path))
        arblib = ctypes.CDLL(str(_arb_path))
elif platform.system() == "Darwin":
    with importlib.resources.path(
        "tmlt.core.ext.lib", "libgmp.10.dylib"
    ) as _gmp_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libmpfr.6.dylib"
    ) as _mpfr_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libarb-2.14.0.dylib"
    ) as _arb_path, importlib.resources.path(
        "tmlt.core.ext.lib", "libflint-17.dylib"
    ) as _flint_path:
        # NOTE: Below, loading with mode=`RTLD_GLOBAL` makes symbols in GMP available
        # for loading MPFR.
        ctypes.CDLL(str(_gmp_path), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(_mpfr_path))
        flintlib = ctypes.CDLL(str(_flint_path))
        arblib = ctypes.CDLL(str(_arb_path))
else:
    raise RuntimeError(
        "Unrecognized platform. Expected platform.system() to be one of"
        f" 'Windows', 'Linux', or 'Darwin' not ({platform.system()})."
    )


class _PtrStruct(ctypes.Structure):
    """Wrapper type for `mantissa_ptr_struct`.

    ... highlight :: c
    ... code-block :: c

        typedef struct
        {
            mp_size_t alloc;
            mp_ptr d; // NOTE: typedef mp_limb_t * mp_ptr
        }
        mantissa_ptr_struct;
    """

    _fields_ = [("alloc", ctypes.c_int), ("d", ctypes.POINTER(ctypes.c_ulong))]


class _NoPtrStruct(ctypes.Structure):
    """Wrapper type for `mantissa_ptr_struct`.

    ... highlight :: c
    ... code-block :: c

        typedef struct
        {
            mp_limb_t d[ARF_NOPTR_LIMBS];
        }
        mantissa_noptr_struct;

    """

    _fields_ = [("d", ctypes.c_ulong * 2)]


class _MantissaStruct(ctypes.Union):
    """Wrapper type for `mantissa_struct`.

    ... highlight :: c
    ... code-block :: c

        typedef union
        {
            mantissa_noptr_struct noptr;
            mantissa_ptr_struct ptr;
        }
        mantissa_struct;

    """

    _fields_ = [("noptr", _NoPtrStruct), ("ptr", _PtrStruct)]


class _MagStruct(ctypes.Structure):
    """Wrapper type for `mag_struct`.

    This holds an unsigned floating point number with a fixed-precision (30 bits)
    mantissa and an arbitrary precision (integral) exponent.

    ... highlight :: c
    ... code-block :: c

        typedef struct
        {
            fmpz exp;
            mp_limb_t man;
        }
        mag_struct;
    """

    _fields_ = [("exp", ctypes.c_long), ("man", ctypes.c_int)]


class _ArfStruct(ctypes.Structure):
    """Wrapper type for `arf_struct`.

    This holds an arbitrary precision floating point number.

    ... highlight :: c
    ... code-block :: c

        typedef struct
        {
            fmpz exp;
            mp_size_t size;
            mantissa_struct d;
        }
        arf_struct;
    """

    _fields_ = [("exp", ctypes.c_long), ("size", ctypes.c_int), ("d", _MantissaStruct)]


class _ArbStruct(ctypes.Structure):
    """Wrapper type for `arb_struct`.

    ... highlight :: c
    ... code-block :: c

        typedef struct
        {
            arf_struct mid;
            mag_struct rad;
        }
        arb_struct;
    """

    _fields_ = [("mid", _ArfStruct), ("rad", _MagStruct)]


class Arb:
    """An interval on the real line represented by a midpoint and a radius.

    Note: Arbs with radius=0 are referred to as *exact* arbs.
    """

    def __init__(self, ptr):
        """Construct a ball represented exactly.

        NOTE: Do not use directly. Use a `from_` constructor instead.
        """
        self._ptr = ptr

    @staticmethod
    def from_midpoint_radius(
        mid: Union["Arb", int, float, Tuple[int, int]],
        rad: Union["Arb", int, float, Tuple[int, int]],
    ) -> "Arb":
        """Constructs an Arb by specifying its midpoint and radius."""
        mid_arb = mid if isinstance(mid, Arb) else _get_exact_arb(mid)
        rad_arb = rad if isinstance(rad, Arb) else _get_exact_arb(rad)

        if not mid_arb.is_exact() or not rad_arb.is_exact():
            raise ValueError("Midpoint/radius must be exact Arbs.")
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        arblib.arb_set(x, mid_arb._ptr)
        arblib.arb_add_error(x, rad_arb._ptr)
        return Arb(x)

    @staticmethod
    def from_int(val: int) -> "Arb":
        """Constructs an exact Arb with midpoint `val`."""
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        fmpz_pointer = _int_to_fmpz_t(val)
        arblib.arb_set_fmpz(x, fmpz_pointer)
        return Arb(x)

    @staticmethod
    def from_float(val: float) -> "Arb":
        """Construct an exact Arb with midpoint `val`."""
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        arblib.arb_set_d(x, ctypes.c_double(float(val)))
        return Arb(x)

    @staticmethod
    def from_man_exp(man: int, exp: int) -> "Arb":
        """Constructs an exact arb with midpoint specified as mantissa and exponent."""
        x = ctypes.pointer(_ArbStruct())
        mid = ctypes.pointer(_ArfStruct())
        arblib.arb_init(x)
        arblib.arf_init(mid)
        arblib.arf_set_fmpz_2exp(mid, _int_to_fmpz_t(man), _int_to_fmpz_t(exp))
        arblib.arb_set_arf(x, mid)
        return Arb(x)

    def __lt__(self, other: Any) -> bool:
        """Returns True if self is less than other.

        This returns True if each value in the interval represented by `self` is less
        than every value in `other`.

        Note: If the midpoint is NaN, this returns False.
        """
        if other.__class__ != self.__class__:
            return NotImplemented
        return arblib.arb_lt(self._ptr, other._ptr) > 0

    def __le__(self, other: Any) -> bool:
        """Returns True if self is less than or equal to other.

        This returns True if each value in the interval represented by `self` is less
        than or equal to every value in `other`.

        Note: If the midpoint is NaN, this returns False.
        """
        if other.__class__ != self.__class__:
            return NotImplemented
        return arblib.arb_le(self._ptr, other._ptr) > 0

    def __ne__(self, other: Any) -> bool:
        """Returns True if self is not equal to other.

        This returns True if no value in the interval represented by `self` is equal
        to a value in `other`.

        Note: If the midpoint is NaN, this returns False.
        """
        return arblib.arb_ne(self._ptr, other._ptr) > 0

    def __gt__(self, other: Any) -> bool:
        """Returns True if self is greater than other.

        This returns True if each value in the interval represented by `self` is
        greater than every value in `other`.

        Note: If the midpoint is NaN, this returns False.
        """
        if other.__class__ != self.__class__:
            return NotImplemented
        return arblib.arb_gt(self._ptr, other._ptr) > 0

    def __ge__(self, other: Any) -> bool:
        """Returns True if self is greater than or equal to other.

        This returns True if each value in the interval represented by `self` is
        greater than or equal to every value in `other`.

        Note: If the midpoint is NaN, this returns False.
        """
        if other.__class__ != self.__class__:
            return NotImplemented
        return arblib.arb_ge(self._ptr, other._ptr) > 0

    def __eq__(self, other: Any) -> bool:
        """Returns True if self is equal to other.

        This returns True if each value in the interval represented by `self` is
        greater than every value in `other`. If midpoint is NaN, this returns False.
        """
        if other.__class__ != self.__class__:
            return NotImplemented
        return arblib.arb_eq(self._ptr, other._ptr) > 0

    def __neg__(self) -> "Arb":
        """Returns -1 * self`."""
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        arblib.arb_neg(x, self._ptr)
        return Arb(x)

    def is_exact(self) -> bool:
        """Returns True if radius is 0."""
        return bool(arblib.arb_is_exact(self._ptr))

    def is_finite(self) -> bool:
        """Returns True if self is finite."""
        return bool(arblib.arb_is_finite(self._ptr))

    def is_nan(self) -> bool:
        """Returns True if midpoint is NaN."""
        return bool(arblib.arf_is_nan(ctypes.byref(self._ptr.contents.mid)))

    def __float__(self) -> float:
        """Returns a float approximating the midpoint of `self`."""
        arblib.arf_get_d.restype = ctypes.c_double
        # 4 below corresponds to ARB_RND_NEAR
        return arblib.arf_get_d(ctypes.byref(self._ptr.contents.mid), 4)

    def to_float(self, prec: int = 64) -> float:
        """Returns the only floating point number contained in `self`.

        If more than one float lies in the interval represented by `self`,
        this raises an error.
        """
        if not self.is_nan() and self.is_finite():
            l_man, l_exp = self.lower(prec).man_exp()
            u_man, u_exp = self.upper(prec).man_exp()
            lower_float = math.ldexp(int(l_man), int(l_exp))
            upper_float = math.ldexp(int(u_man), int(u_exp))
            if lower_float == upper_float:
                return lower_float
        raise ValueError("Arb contains more than one float.")

    def man_exp(self) -> Tuple[int, int]:
        """Returns a pair (Mantissa, Exponent) if exact.

        If `self` is not exact, this raises an error.
        """
        if not self.is_exact():
            raise ValueError("Arb must be exact to obtain (man, exp) representation.")
        x = self._ptr.contents.mid
        man_ptr = ctypes.pointer(ctypes.c_long())
        exp_ptr = ctypes.pointer(ctypes.c_long())
        arblib.arf_get_fmpz_2exp(man_ptr, exp_ptr, ctypes.byref(x))
        return _fmpz_t_to_int(man_ptr), _fmpz_t_to_int(exp_ptr)

    def lower(self, prec: int) -> "Arb":
        """Returns an exact Arb representing the smallest value in `self`."""
        x = ctypes.pointer(_ArbStruct())
        mid = ctypes.pointer(_ArfStruct())
        arblib.arf_init(mid)
        arblib.arb_get_lbound_arf(mid, self._ptr, prec)
        arblib.arb_init(x)
        arblib.arb_set_arf(x, mid)
        return Arb(x)

    def upper(self, prec: int) -> "Arb":
        """Returns an exact Arb representing the largest value in `self`."""
        x = ctypes.pointer(_ArbStruct())
        mid = ctypes.pointer(_ArfStruct())
        arblib.arf_init(mid)
        arblib.arb_get_ubound_arf(mid, self._ptr, prec)
        arblib.arb_init(x)
        arblib.arb_set_arf(x, mid)
        return Arb(x)

    def midpoint(self) -> "Arb":
        """Returns the midpoint of `self`."""
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        arblib.arb_get_mid_arb(x, self._ptr)
        return Arb(x)

    def radius(self) -> "Arb":
        """Returns the radius of `self`."""
        x = ctypes.pointer(_ArbStruct())
        arblib.arb_init(x)
        arblib.arb_get_rad_arb(x, self._ptr)
        return Arb(x)

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is contained in the interval represented by `self`."""
        if isinstance(value, Arb):
            return arblib.arb_contains(self._ptr, value._ptr) != 0
        return False

    def __str__(self) -> str:
        """String representation."""
        arblib.arb_get_str.restype = ctypes.c_char_p
        arb_str = arblib.arb_get_str(self._ptr, 8, 1)
        return arb_str.decode("UTF-8")

    def __del__(self):
        """Cleanup."""
        arblib.arb_clear(self._ptr)


def arb_sub(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns `x - y`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_sub(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_add(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns sum of `x` and `y`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_add(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_mul(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns product of `x` and `y`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_mul(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_div(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns quotient of `x` divided by `y`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_div(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_log(x: Arb, prec: int) -> Arb:
    """Returns `log(x)`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_log(z, x._ptr, prec)
    return Arb(z)


def arb_max(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns max of `x` and `y`.

    If `x` and `y` represent intervals [x_l, x_u] and [y_l, y_u], this returns an
    interval containing [max(x_l, y_l), max(x_u, y_u)].
    """
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_max(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_min(x: Arb, y: Arb, prec: int) -> Arb:
    """Returns min of `x` and `y`.

    If `x` and `y` represent intervals [x_l, x_u] and [y_l, y_u], this returns an
    interval containing [min(x_l, y_l), min(x_u, y_u)].
    """
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_min(z, x._ptr, y._ptr, prec)
    return Arb(z)


def arb_abs(x: Arb) -> Arb:
    """Returns absolute value of x."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_abs(z, x._ptr)
    return Arb(z)


def arb_sgn(x: Arb) -> Arb:
    """Sign function.

    If x contains both zero and nonzero numbers, this returns an Arb with mid=0
    and rad=1.
    """
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_sgn(z, x._ptr)
    return Arb(z)


def arb_erfinv(x: Arb, prec: int) -> Arb:
    """Inverse error function."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_hypgeom_erfinv(z, x._ptr, prec)
    return Arb(z)


def arb_sqrt(x: Arb, prec: int) -> Arb:
    """Returns square root of `x`."""
    z = ctypes.pointer(_ArbStruct())
    arblib.arb_init(z)
    arblib.arb_sqrt(z, x._ptr, prec)
    return Arb(z)


def _int_to_fmpz_t(val: int) -> "ctypes._PointerLike":
    """Returns pointer to a C arbitrary precision integer from a python integer.

    Args:
        val: Integer to convert.
    """
    fmpz_pointer = ctypes.pointer(ctypes.c_long())
    s = "%x" % int(val)
    val_c_string = ctypes.c_char_p(s.encode("ascii"))
    flintlib.fmpz_set_str(fmpz_pointer, val_c_string, 16)
    return fmpz_pointer


def _fmpz_t_to_int(ptr: "ctypes._PointerLike") -> int:
    """Returns python integer from an fmpz_t."""
    flintlib.fmpz_get_str.restype = ctypes.c_char_p
    c_str = flintlib.fmpz_get_str(None, 10, ptr)
    return int(c_str)


def _get_exact_arb(val: Union[int, float, Tuple[int, int]]) -> Arb:
    """Returns an exact Arb with midpoint=`val`."""
    if isinstance(val, int):
        return Arb.from_int(val)
    if isinstance(val, float):
        return Arb.from_float(val)
    assert (
        len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], int)
    ), f"Invalid mantissa, exponent tuple : ({val}). Expected a pair of integers"
    return Arb.from_man_exp(val[0], val[1])
