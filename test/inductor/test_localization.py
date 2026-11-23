# Owner(s): ["module: inductor"]
"""
Tests for torch.cuda.localize / _LocalizedTensor.

These tests exercise the tensor-subclass-based localization path:
``torch.cuda.localize(module)`` rewrites every ``nn.Linear``'s weight (and
bias) into a ``_LocalizedTensor`` whose halves live in per-locality-domain
green-context-backed memory pools.  ``__torch_function__`` then forks each
``F.linear`` call into per-branch matmuls on green streams, with a final
``aten.cat`` joining the per-branch outputs.

Pointwise tails (relu, bias-add, gelu, ...) consume the concatenated output.
"""

import contextlib
import functools
import unittest
import warnings
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


def _requires_cuda(test_func):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(
        test_func
    )


def _requires_green_contexts(test_func):
    """Skip test if localized green contexts are not supported."""

    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from torch.cuda.green_contexts import (
                GreenContext,
                is_localization_supported,
            )

            if not is_localization_supported(0):
                raise unittest.SkipTest(
                    "Localized green contexts not supported on this device"
                )
            (gc,) = GreenContext.split(locality_domain_ids=(0,), device_id=0)
            del gc
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"Green contexts not supported: {e}") from e
        return test_func(*args, **kwargs)

    return wrapper


def _cublas_version_int():
    """Return cuBLAS version as a single int (MMmmpp), or None if unavailable.

    Used to gate tests around bugs fixed in specific cuBLAS releases.
    13.4.1 returns 130401; 13.6.0 returns 130600.
    """
    import ctypes

    try:
        cublas = ctypes.CDLL("libcublas.so.13")
    except OSError:
        return None
    cublas.cublasCreate_v2.restype = ctypes.c_int
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas.cublasDestroy_v2.restype = ctypes.c_int
    cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublas.cublasGetVersion_v2.restype = ctypes.c_int
    cublas.cublasGetVersion_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    # cublasCreate needs a current CUDA context.
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    h = ctypes.c_void_p(0)
    if cublas.cublasCreate_v2(ctypes.byref(h)) != 0:
        return None
    v = ctypes.c_int(0)
    rc = cublas.cublasGetVersion_v2(h.value, ctypes.byref(v))
    cublas.cublasDestroy_v2(h)
    return v.value if rc == 0 else None


def _requires_cublas_tensorop_on_green_streams(test_func):
    """Skip if cuBLAS's TensorOp GEMM path fails on green-context-backed
    streams.  Reproducible in <10 lines: create a green context, get its
    stream, call ``F.linear`` with fp16 / bf16 inputs on the stream.

    Some cuBLAS versions return ``CUBLAS_STATUS_EXECUTION_FAILED``
    from ``cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`` when the
    stream is bound to a green context, regardless of whether the green
    context is currently pushed.  The fp32 codepath (non-TensorOp)
    works fine on the same stream.

    Fixed in cuBLAS 13.6.
    """

    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        v = _cublas_version_int()
        if v is not None and v < 130600:
            raise unittest.SkipTest(
                f"cuBLAS {v} has a TensorOp-on-green-stream bug; upgrade cuBLAS"
            )
        return test_func(*args, **kwargs)

    return wrapper


def _is_localized_tensor(t):
    from torch.cuda.localized_tensor import _LocalizedTensor

    return isinstance(t, _LocalizedTensor)


# ---------------------------------------------------------------------------
# Helper modules.  Each holds its weights inside ``nn.Linear``s so
# ``torch.cuda.localize`` can replace them with ``_LocalizedTensor``
# parameters.
# ---------------------------------------------------------------------------


class _OneLinear(nn.Module):
    """Single linear over its input."""

    def __init__(self, in_f, out_f, bias=False, dtype=torch.float32):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.fc(x)


class _LinearReLU(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return torch.relu(self.fc(x))


class _LinearLeakyReLU(nn.Module):
    def __init__(self, in_f, out_f, slope=0.01, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)
        self.slope = slope

    def forward(self, x):
        return F.leaky_relu(self.fc(x), negative_slope=self.slope)


class _TwoSequentialLinears(nn.Module):
    def __init__(self, k, n, p, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _TwoIndependentLinears(nn.Module):
    """y1 = fc1(x); y2 = fc2(x); return y1 + y2."""

    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_f, out_f, bias=bias)
        self.fc2 = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.fc1(x) + self.fc2(x)


class _LinearBiasSin(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x):
        return torch.sin(self.fc(x))


class _AttentionQKV(nn.Module):
    """Per-head MHA forward with three independent Q/K/V projections.

    The Q/K/V projections become nn.Linear and get localized; the
    rest of the attention (view/transpose/bmm/softmax) runs unchanged
    on the main stream.
    """

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, batch, seq_len):
        q = (
            self.q(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = q.reshape(batch * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(batch * self.num_heads, seq_len, self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)
        scores = torch.softmax(scores, dim=-1)
        out = torch.bmm(scores, v)
        out = out.view(batch, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        return out.contiguous().view(batch * seq_len, self.hidden)


class _FFNResidualLN(nn.Module):
    def __init__(self, hidden, ffn_dim):
        super().__init__()
        self.hidden = hidden
        self.ffn = nn.Linear(hidden, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x):
        out = self.ffn(x)
        out = self.down(out) + x
        return F.layer_norm(out, (self.hidden,))


class _FusedQKV(nn.Module):
    """Single nn.Linear for fused QKV; the rest is unchanged."""

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)

    def forward(self, x, batch, seq_len):
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = (
            q.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        k = (
            k.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        v = (
            v.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        scores = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5), dim=-1
        )
        out = torch.bmm(scores, v)
        out = out.view(batch, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        return out.contiguous().view(batch * seq_len, self.hidden)


class _NSequentialLinears(nn.Module):
    """N stacked square nn.Linears with no activations in between."""

    def __init__(self, dims, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x


class _LinearGELULinear(nn.Module):
    def __init__(self, k, n, p, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LinearHeadEscapes(nn.Module):
    """Returns intermediate y AND a final tail."""

    def __init__(self, k, n, p, q, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)
        self.fc3 = nn.Linear(p, q, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        z = self.fc2(y)
        return y, self.fc3(z)


class _LinearHeadEscapes2(nn.Module):
    """Two stacked linears; the head's output is returned alongside the tail."""

    def __init__(self, k, n, p):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=False)
        self.fc2 = nn.Linear(n, p, bias=False)

    def forward(self, x):
        head = self.fc1(x)
        tail = self.fc2(head)
        return head, tail


class _MultiConsumerHead(nn.Module):
    """One linear's output feeds two downstream linears that get summed."""

    def __init__(self, k, n, p, q, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)
        self.fc3 = nn.Linear(n, q, bias=bias)

    def forward(self, x):
        mm1 = self.fc1(x)
        return self.fc2(mm1) + self.fc3(mm1)


class _SharedLinearTwoCalls(nn.Module):
    """One nn.Linear called on two different inputs.  The same
    _LocalizedTensor weight is read from both call sites; numerics must
    match eager."""

    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x1, x2):
        return self.fc(x1) + self.fc(x2)


class _LinearMergeInnerSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 8, bias=False)

    def forward(self, x):
        y = self.fc(x)  # [6, 8]
        y = y.view(6, 2, 4)  # 1-to-many
        return y.view(12, 4)  # merge with split inside


# ---------------------------------------------------------------------------
# Drive helpers.
# ---------------------------------------------------------------------------


def _eager_and_localized(module_cls, ctor_args, forward_args, dtype=torch.float32):
    """Build two copies of the module, localize one, and return
    (eager_out, loc_out, localized_module).

    ``forward_args`` is a list of inputs (CUDA tensors / non-tensor scalars)
    passed positionally to forward().  Both module copies start from the
    same RNG seed so their parameters are bit-identical before localization.
    """
    torch.manual_seed(0)
    eager = module_cls(*ctor_args).to(device="cuda", dtype=dtype).eval()
    torch.manual_seed(0)
    loc = module_cls(*ctor_args).to(device="cuda", dtype=dtype).eval()
    with torch.inference_mode():
        torch.cuda.localize(loc)
    with torch.inference_mode():
        y_eager = eager(*forward_args)
        y_loc = loc(*forward_args)
        torch.cuda.synchronize()
    return y_eager, y_loc, loc


# ---------------------------------------------------------------------------
# Tests: parameter-shape / type contract
# ---------------------------------------------------------------------------


class TestLocalizeContract(TestCase):
    """The user-facing contract of torch.cuda.localize on an nn.Module."""

    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_localize_replaces_linear_weight_with_subclass(self):
        """nn.Linear.weight should become a _LocalizedTensor after localize."""
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m.fc.weight))
        self.assertTrue(_is_localized_tensor(m.fc.bias))
        # Public-shape contract: from outside, the parameter still looks
        # like (out_features, in_features).
        self.assertEqual(tuple(m.fc.weight.shape), (128, 64))
        self.assertEqual(tuple(m.fc.bias.shape), (128,))

    @_requires_green_contexts
    def test_inplace_update_of_localized_param_raises(self):
        m = _OneLinear(64, 128).cuda()
        sd = {k: torch.zeros_like(v) for k, v in m.state_dict().items()}
        with torch.inference_mode():
            torch.cuda.localize(m)
            with self.assertRaisesRegex(NotImplementedError, "in place"):
                m.fc.weight.copy_(torch.zeros(128, 64, device="cuda"))
            with self.assertRaisesRegex(RuntimeError, "in place"):
                m.load_state_dict(sd)
            with self.assertRaisesRegex(NotImplementedError, "in place"):
                torch._foreach_add_([m.fc.weight], 1.0)
            with self.assertRaisesRegex(NotImplementedError, "in place"):
                torch.mul(torch.ones(128, 64, device="cuda"), 0, out=m.fc.weight)
            m(torch.randn(4, 64, device="cuda"))

    @_requires_green_contexts
    def test_localize_idempotent(self):
        """Re-localizing should be a no-op."""
        m = _OneLinear(64, 128).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        w_id = id(m.fc.weight)
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertEqual(id(m.fc.weight), w_id)

    @_requires_green_contexts
    def test_localize_returns_module_for_chaining(self):
        m = _OneLinear(64, 128).cuda()
        with torch.inference_mode():
            ret = torch.cuda.localize(m)
        self.assertIs(ret, m)

    @_requires_cuda
    def test_localize_rejects_weight_norm(self):
        from torch.nn.utils.parametrizations import weight_norm

        with torch.inference_mode():
            m = nn.Sequential(
                nn.Linear(8, 8, device="cuda"),
                weight_norm(nn.Linear(8, 8, device="cuda")),
            )
            with self.assertRaisesRegex(NotImplementedError, "parametrized weight"):
                torch.cuda.localize(m)
        self.assertFalse(_is_localized_tensor(m[0].weight))

    def test_rejects_graph_module(self):
        gm = torch.fx.symbolic_trace(_OneLinear(64, 128))
        with torch.inference_mode():
            with self.assertRaisesRegex(
                NotImplementedError, "does not support torch.fx.GraphModule"
            ):
                torch.cuda.localize(gm)

    @_requires_green_contexts
    def test_rejects_odd_output_dim(self):
        """Splitting an odd output dim must error rather than silently
        miscompute, before any layer is localized."""
        m = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 17)).cuda()
        with self.assertRaisesRegex(ValueError, "divisible"):
            with torch.inference_mode():
                torch.cuda.localize(m)
        self.assertFalse(_is_localized_tensor(m[0].weight))

    @_requires_cuda
    def test_localize_skips_and_warns_once_when_unsupported(self):
        """On devices without locality-domain support, localize must leave
        the module untouched and warn once per process instead of raising.
        Runs on any CUDA device: support is mocked away."""
        from torch._dynamo.utils import warn_once_cache

        m = _OneLinear(64, 128, bias=True).cuda()
        warn_once_cache.clear()
        with (
            mock.patch(
                "torch.cuda.green_contexts.is_localization_supported",
                return_value=False,
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            with torch.inference_mode():
                ret = torch.cuda.localize(m)
            with torch.inference_mode():
                torch.cuda.localize(m)  # second call: no second warning

        self.assertIs(ret, m)
        self.assertFalse(_is_localized_tensor(m.fc.weight))
        self.assertFalse(_is_localized_tensor(m.fc.bias))
        relevant = [w for w in caught if "localization" in str(w.message)]
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0].filename, __file__)
        # The unlocalized module must still work.
        x = torch.randn(4, 64, device="cuda")
        y = m(x)
        self.assertEqual(tuple(y.shape), (4, 128))


# ---------------------------------------------------------------------------
# Tests: localizable-op registry
# ---------------------------------------------------------------------------


class TestLocalizableOpRegistry(TestCase):
    """``_LOCALIZABLE_OPS`` drives every localization decision: which ops
    fork across branches, and which module types ``localize`` rewrites.
    Nothing keys on ``linear`` specifically."""

    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_linear_is_registered_not_special_cased(self):
        """F.linear is an ordinary registry entry, carrying the arg names
        needed to match keyword args and the module type localize walks."""
        from torch.cuda.localized_tensor import (
            _LOCALIZABLE_OPS,
            _register_default_localizable_ops,
        )

        _register_default_localizable_ops()
        meta = _LOCALIZABLE_OPS[F.linear]
        self.assertEqual(meta.localized_args, (1, 2))
        self.assertEqual(meta.arg_names, ("input", "weight", "bias"))
        self.assertIs(meta.module_type, nn.Linear)
        # aten.linear is the exported shape; its names come from the schema.
        aten_meta = _LOCALIZABLE_OPS[torch.ops.aten.linear.default]
        self.assertEqual(aten_meta.arg_names, ("input", "weight", "bias"))

    @_requires_green_contexts
    def test_localized_args_may_be_passed_by_keyword(self):
        """A localized position is resolved whether the caller passed it
        positionally, by keyword, or omitted it entirely."""
        m = _OneLinear(64, 128, bias=True).cuda()
        x = torch.randn(32, 64, device="cuda")
        with torch.inference_mode():
            torch.cuda.localize(m)
            w, b = m.fc.weight, m.fc.bias
            positional = F.linear(x, w, b)
            for by_keyword in (
                F.linear(x, w, bias=b),
                F.linear(x, weight=w, bias=b),
                F.linear(input=x, weight=w, bias=b),
            ):
                self.assertEqual(by_keyword, positional, atol=0, rtol=0)
            # Omitted entirely must mean "no bias", not "a bias of zeros by
            # accident" -- match an explicit None, and differ from the biased call.
            self.assertEqual(F.linear(x, w), F.linear(x, w, None), atol=0, rtol=0)
            self.assertFalse(torch.equal(F.linear(x, w), positional))

    @_requires_green_contexts
    def test_keyword_localized_arg_matches_unlocalized(self):
        """Numerics, not just self-consistency: a keyword-passed localized
        weight must match the same call on the reified tensor."""
        m = _OneLinear(64, 128, bias=True).cuda()
        ref_w = m.fc.weight.detach().clone()
        ref_b = m.fc.bias.detach().clone()
        x = torch.randn(32, 64, device="cuda")
        expected = F.linear(x, ref_w, bias=ref_b)

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = F.linear(x, weight=m.fc.weight, bias=m.fc.bias)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)


class _ScaledLinear(nn.Module):
    """Module whose stock forward calls a registered custom op."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return torch.ops.test_localization.scaled_linear.default(
            x, self.weight, bias=self.bias, scale=2.0
        )


class TestLocalizableOpRegistryCustomOp(TestCase):
    """A downstream op registered with ``_register_localizable_op`` gets the
    same treatment the built-in linear does, including through kwargs and
    the module walk."""

    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        from torch.cuda.localized_tensor import (
            _LOCALIZABLE_OPS,
            _register_localizable_op,
        )

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self._lib = stack.enter_context(
            torch.library._scoped_library("test_localization", "FRAGMENT")
        )
        self._lib.define(
            "scaled_linear(Tensor input, Tensor weight, Tensor? bias=None, "
            "float scale=1.0) -> Tensor"
        )

        def _scaled_linear(input, weight, bias=None, scale=1.0):
            return F.linear(input, weight, bias) * scale

        self._lib.impl("scaled_linear", _scaled_linear, "CUDA")
        self._op = torch.ops.test_localization.scaled_linear.default
        _register_localizable_op(
            self._op, localized_args=(1, 2), module_type=_ScaledLinear
        )
        self._registry = _LOCALIZABLE_OPS
        # The registry is process-global; leaving this op in it would leak
        # into every later test's module walk.
        self.addCleanup(self._registry.pop, self._op, None)

    @_requires_green_contexts
    def test_arg_names_derived_from_schema(self):
        meta = self._registry[self._op]
        self.assertEqual(meta.arg_names, ("input", "weight", "bias", "scale"))

    @_requires_green_contexts
    def test_custom_op_forks_with_keyword_args(self):
        """The op's bias arrives by keyword and its non-localized ``scale``
        must reach every branch untouched."""
        m = _ScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = m(x)

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = m(x)
        self.assertTrue(_is_localized_tensor(m.weight))
        self.assertTrue(_is_localized_tensor(m.bias))
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_module_walk_is_registry_driven(self):
        """localize() rewrites a registered module type it has never heard
        of, and leaves an unregistered one alone."""

        class _Unregistered(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(128, 64))
                nn.init.normal_(self.weight)

            def forward(self, x):
                return x @ self.weight.t()

        m = nn.Sequential(_ScaledLinear(64, 128), _Unregistered()).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m[0].weight))
        self.assertFalse(_is_localized_tensor(m[1].weight))

    @_requires_green_contexts
    def test_subclass_overriding_forward_is_skipped(self):
        """The stock-forward guard is keyed on the registered module type,
        so an override still opts the layer out."""

        class _Overridden(_ScaledLinear):
            def forward(self, x):
                return F.linear(x, self.weight, self.bias)

        m = _Overridden(64, 128).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertFalse(_is_localized_tensor(m.weight))


class _ChannelScaledLinear(nn.Module):
    """Linear with a per-output-channel scale -- the shape sliced_args exists
    for.  ``scale`` is a buffer, not a parameter, so localize() leaves it
    whole and the dispatch has to split it to match the weight's rows."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        self.register_buffer("scale", torch.empty(out_f))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)
        nn.init.uniform_(self.scale, 0.5, 1.5)

    def forward(self, x):
        return torch.ops.test_localization.channel_scaled_linear.default(
            x, self.weight, self.bias, self.scale
        )


class TestLocalizableOpSlicedArgs(TestCase):
    """sliced_args positions are split per-branch along dim 0 but left in the
    default pool, for per-output-channel side tensors like quantization
    scales.  No built-in op registers any -- nn.Linear has none -- so this is
    the only coverage of that path."""

    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        from torch.cuda.localized_tensor import _LOCALIZABLE_OPS

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self._lib = stack.enter_context(
            torch.library._scoped_library("test_localization", "FRAGMENT")
        )
        self._lib.define(
            "channel_scaled_linear(Tensor input, Tensor weight, Tensor? bias=None, "
            "Tensor? scale=None, Tensor[]? extra=None, float bogus=1.0) -> Tensor"
        )

        def _channel_scaled_linear(
            input, weight, bias=None, scale=None, extra=None, bogus=1.0
        ):
            out = F.linear(input, weight, bias)
            if scale is not None:
                out = out * scale
            for e in extra or ():
                out = out * e
            return out

        self._lib.impl("channel_scaled_linear", _channel_scaled_linear, "CUDA")
        self._op = torch.ops.test_localization.channel_scaled_linear.default
        self._register(sliced_args=(3, 4))
        self._registry = _LOCALIZABLE_OPS
        self.addCleanup(_LOCALIZABLE_OPS.pop, self._op, None)

    def _register(self, *, sliced_args):
        from torch.cuda.localized_tensor import _register_localizable_op

        with warnings.catch_warnings():
            # Re-registering to vary sliced_args warns by design.
            warnings.simplefilter("ignore")
            _register_localizable_op(
                self._op,
                localized_args=(1, 2),
                sliced_args=sliced_args,
                module_type=_ChannelScaledLinear,
            )

    @_requires_green_contexts
    def test_tensor_sliced_arg_matches_unlocalized(self):
        """A per-channel scale must be split to line up with the weight rows
        each branch owns; leaving it whole would multiply every branch by the
        full-width scale and silently corrupt the result."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = m(x)

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = m(x)
        self.assertTrue(_is_localized_tensor(m.weight))
        # The scale is a buffer, so it stays whole and gets sliced per call.
        self.assertFalse(_is_localized_tensor(m.scale))
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_sliced_arg_may_be_none(self):
        """An explicit None in a sliced position is replicated to every
        branch rather than sliced."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = self._op(x, m.weight, m.bias, None)

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = self._op(x, m.weight, m.bias, None)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_sliced_arg_list_of_tensors(self):
        """A list in a sliced position is sliced element-wise and transposed
        so each branch gets a list of its own slices."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        extra = [torch.rand(128, device="cuda") + 0.5 for _ in range(3)]
        expected = self._op(x, m.weight, m.bias, m.scale, extra)

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = self._op(x, m.weight, m.bias, m.scale, extra)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_sliced_arg_empty_list(self):
        """An empty list has nothing to slice, so each branch gets an empty
        list rather than a slice of nothing."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = self._op(x, m.weight, m.bias, m.scale, [])

        with torch.inference_mode():
            torch.cuda.localize(m)
            got = self._op(x, m.weight, m.bias, m.scale, [])
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_unsliceable_sliced_arg_raises(self):
        """Registering a position that cannot be split must report which arg
        and what it got, not fail somewhere downstream."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        with torch.inference_mode():
            torch.cuda.localize(m)
            self._register(sliced_args=(5,))  # 'bogus', a float
            with self.assertRaisesRegex(TypeError, "must be None, Tensor, or list"):
                self._op(x, m.weight, m.bias, m.scale, [], 2.0)


# ---------------------------------------------------------------------------
# Tests: runtime numerics
# ---------------------------------------------------------------------------


class TestLocalizationRuntimeNumerics(TestCase):
    """The localized forward must match the eager forward across common
    model patterns."""

    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_basic_linear(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_linear_with_pointwise_relu(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearReLU, (64, 128), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_linear_with_leaky_relu(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearLeakyReLU, (64, 128), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_two_sequential_linears(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_TwoSequentialLinears, (64, 128, 64), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_two_independent_linears(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_TwoIndependentLinears, (64, 128), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_linear_bias_sin(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearBiasSin, (64, 128), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_attention_qkv(self):
        batch, seq_len, hidden, num_heads, head_dim = 2, 8, 64, 4, 16
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _AttentionQKV,
            (hidden, num_heads, head_dim),
            (x, batch, seq_len),
        )
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_ffn_residual_layernorm(self):
        batch, seq_len, hidden, ffn_dim = 2, 8, 64, 128
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_FFNResidualLN, (hidden, ffn_dim), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_fused_qkv_split(self):
        batch, seq_len, hidden, num_heads, head_dim = 2, 8, 64, 4, 16
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _FusedQKV,
            (hidden, num_heads, head_dim),
            (x, batch, seq_len),
        )
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: chain numerics (multiple sequential linears).  Each layer's
# fork-join is self-contained, so chains must match eager layer by layer.
# ---------------------------------------------------------------------------


class TestLocalizationChainNumerics(TestCase):
    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_chain_2(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_NSequentialLinears, ([64, 48, 32],), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_mm_gelu_mm(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearGELULinear, (64, 64, 32), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_3(self):
        x = torch.randn(16, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _NSequentialLinears, ([64, 64, 64, 32],), (x,)
        )
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_4(self):
        x = torch.randn(8, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _NSequentialLinears, ([64, 64, 64, 64, 32],), (x,)
        )
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_head_escapes(self):
        x = torch.randn(16, 64, device="cuda")
        y_e_pair, y_l_pair, _ = _eager_and_localized(
            _LinearHeadEscapes, (64, 64, 48, 32), (x,)
        )
        for ye, yl in zip(y_e_pair, y_l_pair):
            self.assertEqual(yl, ye, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_head_escapes_two_layer(self):
        torch.manual_seed(0)
        x = torch.randn(32, 64, device="cuda")
        y_e_pair, y_l_pair, _ = _eager_and_localized(
            _LinearHeadEscapes2, (64, 128, 96), (x,)
        )
        for ye, yl in zip(y_e_pair, y_l_pair):
            self.assertEqual(yl, ye, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_chain_multiple_consumers(self):
        x = torch.randn(16, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_MultiConsumerHead, (64, 64, 32, 32), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_view_merge_inner_split(self):
        x = torch.randn(6, 3, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearMergeInnerSplit, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: pointwise / view tails after a localized linear.  The tail runs on
# the main stream after the cat.
# ---------------------------------------------------------------------------


class TestLocalizationPackingNumerics(TestCase):
    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_linear_with_bias(self):
        """linear(x) + b: bias is now part of nn.Linear, no separate add."""
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128, True), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_linear_bias_relu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return torch.relu(self.fc(x))

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_packing_stops_at_reduction(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return self.fc(x).sum()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_green_contexts
    def test_packing_through_unsqueeze(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).unsqueeze(0)

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_packing_through_transpose(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).t()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_packing_through_contiguous(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return self.fc(x).contiguous()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_linear_bias_gelu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return F.gelu(self.fc(x))

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_view_after_linear(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).reshape(-1)

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: shared-weight nn.Linear used at multiple call sites.
# ---------------------------------------------------------------------------


class TestLocalizationSharedWeight(TestCase):
    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_linear_called_twice(self):
        """One nn.Linear, two inputs.  The same _LocalizedTensor weight is
        read by two F.linear sites; both should produce eager-equivalent
        output."""
        x1 = torch.randn(32, 64, device="cuda")
        x2 = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_SharedLinearTwoCalls, (64, 128), (x1, x2))
        self.assertEqual(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_green_contexts
    def test_tied_embedding_lm_head(self):
        """head.weight = embed.weight: the tie must survive localize so
        both call sites see the same _LocalizedTensor."""

        class TiedGPT(nn.Module):
            def __init__(self, vocab=64, dim=32):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.head.weight = self.embed.weight

            def forward(self, x):
                return self.head(self.embed(x))

        torch.manual_seed(0)
        eager = TiedGPT().cuda().eval()
        torch.manual_seed(0)
        loc = TiedGPT().cuda().eval()
        with torch.inference_mode():
            torch.cuda.localize(loc)
        self.assertIs(loc.head.weight, loc.embed.weight)
        self.assertTrue(_is_localized_tensor(loc.head.weight))

        x = torch.tensor([0, 1, 2, 3], device="cuda")
        with torch.inference_mode():
            self.assertEqual(loc(x), eager(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: dtype variants
# ---------------------------------------------------------------------------


class TestLocalizationDtypes(TestCase):
    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    @_requires_cublas_tensorop_on_green_streams
    def test_dtype_float16(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        y_e, y_l, _ = _eager_and_localized(
            _OneLinear, (64, 128), (x,), dtype=torch.float16
        )
        self.assertEqual(y_l, y_e, atol=5e-3, rtol=5e-3)

    @_requires_green_contexts
    @_requires_cublas_tensorop_on_green_streams
    def test_dtype_bfloat16(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        y_e, y_l, _ = _eager_and_localized(
            _OneLinear, (64, 128), (x,), dtype=torch.bfloat16
        )
        self.assertEqual(y_l, y_e, atol=5e-3, rtol=5e-3)


class TestLocalizationRequiresInferenceMode(TestCase):
    """``localize`` refuses to run outside inference mode.

    Localization is inference-only: the split halves are not set up to
    backprop through the fork/join, so a module localized in a training
    context would look trainable and silently fail to learn.  Rejecting the
    call is the contract for supported ``nn.Module`` inputs.
    """

    hw_classification = HardwareClassification.CUDA

    @_requires_green_contexts
    def test_module_path_raises_outside_inference_mode(self):
        m = _OneLinear(64, 128).cuda()
        with self.assertRaisesRegex(RuntimeError, "requires inference mode"):
            torch.cuda.localize(m)
        # The module is left untouched, so a caller can retry correctly.
        self.assertFalse(_is_localized_tensor(m.fc.weight))

    @_requires_green_contexts
    def test_no_grad_is_not_sufficient(self):
        """``no_grad`` is the easy thing to reach for and is not enough --
        it stops gradient recording but leaves inference mode off."""
        m = _OneLinear(64, 128).cuda()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "requires inference mode"):
                torch.cuda.localize(m)


if __name__ == "__main__":
    run_tests()
