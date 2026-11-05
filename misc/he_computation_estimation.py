
##### Basic script to quantify number of operations required on both a client server for

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.nn.modules.utils import _pair


POLY_DEGREE = 2**14

# BFVRNS estimator settings. Bit widths describe moduli, not storage words.
SECURITY_BITS = 128                  # Target classical security; not a proof.
CT_MODULUS_BITS = 320                # log2(Q) budget, excluding auxiliary P.
HYBRID_MODULUS_BITS = 60             # log2(P) budget for hybrid key switching.
RNS_MODULUS_BITS = 60                # Maximum bit width of each RNS prime.
PLAINTEXT_MODULUS = 65537             # Integer batching modulus t.
KEY_SWITCH_TECHNIQUE = "HYBRID"      # Or "DIGIT" for the earlier approximation.
KEY_SWITCH_DIGITS = None             # None derives a partition count from Q/P.
HE_PACKING = "im2col"                # main(): "rotations" or "im2col".
IM2COL_DUPLICATE_COLUMNS = True       # False models the script's single-copy packing.

INT_BITS = 64
INT_BYTES = math.ceil(INT_BITS/8)

INPUT_HEIGHT = 224
INPUT_WIDTH  = 224


INPUT_SHAPE = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)



def layer_to_ops(l: nn.Module, input_shape):
    """Analytical MAC count, including the batch dimension.

    Count one multiply-accumulate as one MAC. Bias additions, pooling,
    activations, dropout, and reshaping are excluded from this MAC count.
    """
    output_shape = _layer_output_shape(l, input_shape)
    if isinstance(l, nn.Conv2d):
        # Each output element takes a dot product over one input group.
        return (math.prod(output_shape) * (l.in_channels // l.groups)
                * math.prod(_pair(l.kernel_size)))
    if isinstance(l, nn.Linear):
        return math.prod(output_shape) * l.in_features
    return 0


def layer_to_nonlinear_ops(l: nn.Module, input_shape):
    """Estimate inference comparisons, separately from linear-layer MACs.

    ReLU uses one comparison per element. Max pooling uses K - 1 comparisons
    per output for a K-element window. This is exact for VGG16's unpadded
    pools; padded or partial windows can make it an upper bound. Dropout is
    inactive at inference. These counts do not estimate an HE protocol's cost.
    """
    output_shape = _layer_output_shape(l, input_shape)
    if isinstance(l, nn.ReLU):
        return math.prod(input_shape)
    if isinstance(l, nn.MaxPool2d):
        return math.prod(output_shape) * (math.prod(_pair(l.kernel_size)) - 1)
    return 0


def _layer_output_shape(l: nn.Module, input_shape):
    """Propagate VGG16 tensor shapes without running a forward pass."""
    if isinstance(l, (nn.Conv2d, nn.MaxPool2d)):
        batch, channels, height, width = input_shape
        if isinstance(l, nn.Conv2d) and channels != l.in_channels:
            raise ValueError(f"Expected {l.in_channels} input channels, got {channels}")
        kernel = _pair(l.kernel_size)
        stride = _pair(l.stride if l.stride is not None else l.kernel_size)
        padding = _pair(l.padding)
        dilation = _pair(l.dilation)
        spatial = []
        for size, k, s, p, d in zip((height, width), kernel, stride, padding, dilation):
            numerator = size + 2 * p - d * (k - 1) - 1
            ceil_mode = isinstance(l, nn.MaxPool2d) and l.ceil_mode
            out = (math.ceil(numerator / s) if ceil_mode else numerator // s) + 1
            if ceil_mode and (out - 1) * s >= size + p:
                out -= 1
            if out <= 0:
                raise ValueError(f"Invalid output size {out} for {l}")
            spatial.append(out)
        return (batch, l.out_channels if isinstance(l, nn.Conv2d) else channels,
                *spatial)
    if isinstance(l, nn.AdaptiveAvgPool2d):
        output_size = _pair(l.output_size)
        return (*input_shape[:-2], *(size if target is None else target
                                    for size, target in zip(input_shape[-2:], output_size)))
    if isinstance(l, nn.Linear):
        if input_shape[-1] != l.in_features:
            raise ValueError(f"Expected {l.in_features} input features, got {input_shape[-1]}")
        return (*input_shape[:-1], l.out_features)
    if isinstance(l, (nn.ReLU, nn.Dropout, nn.Identity)):
        return tuple(input_shape)
    raise TypeError(f"Unsupported layer: {type(l).__name__}")

def _vgg_layer_shapes(m: nn.Module, input_shape):
    """Yield named VGG layers with their analytical input/output shapes."""
    shape = tuple(input_shape)
    layers = [(f"features.{name}", layer) for name, layer in m.features.named_children()]
    layers.append(("avgpool", m.avgpool))
    layers.extend((f"classifier.{name}", layer)
                  for name, layer in m.classifier.named_children())
    for name, layer in layers:
        # VGG's forward flattens the pooled features before its classifier.
        if name == "classifier.0":
            shape = (shape[0], math.prod(shape[1:]))
        output_shape = _layer_output_shape(layer, shape)
        yield name, layer, shape, output_shape
        shape = output_shape


def model_to_ops(m: nn.Module, input_shape=INPUT_SHAPE, verbose=False):
    """Return per-layer counts and separate MAC/comparison totals for VGG.

    All counts include the batch dimension and assume inference. The result
    has three entries: ``layers`` (keyed by module name), ``total_linear_macs``
    and ``total_nonlinear_comparisons``. Each layer records its type, input and
    output shapes, linear MACs and nonlinear comparisons. Bias additions and
    average-pooling additions/divisions are excluded from both totals.
    """
    result = {
        "layers": {},
        "total_linear_macs": 0,
        "total_nonlinear_comparisons": 0,
    }
    for name, layer, shape, output_shape in _vgg_layer_shapes(m, input_shape):
        macs = layer_to_ops(layer, shape)
        comparisons = layer_to_nonlinear_ops(layer, shape)
        result["layers"][name] = {
            "type": type(layer).__name__,
            "input_shape": shape,
            "output_shape": output_shape,
            "linear_macs": macs,
            "nonlinear_comparisons": comparisons,
        }
        if verbose:
            print(f"{name:16s} {type(layer).__name__:18s} "
                  f"{str(shape):20s} -> {str(output_shape):20s} "
                  f"{macs:>15,} MACs {comparisons:>12,} comparisons")
        result["total_linear_macs"] += macs
        result["total_nonlinear_comparisons"] += comparisons
    return result


@dataclass(frozen=True)
class BFVRNSConfig:
    """Target cost parameters, NOT a validated encryption context.

    The rotation baseline uses one N/2-slot row; im2col uses all N slots.
    Modulus t must be prime and t = 1 mod 2N for full batching. The congruence is
    checked here. Actual RNS primes, noise and security require a HE library.
    """

    # Factories read the globals when constructing a config, including globals
    # changed below these definitions or interactively before invoking main().
    poly_degree: int = field(default_factory=lambda: POLY_DEGREE)
    plaintext_modulus: int = field(default_factory=lambda: PLAINTEXT_MODULUS)
    security_bits: int = field(default_factory=lambda: SECURITY_BITS)
    ciphertext_modulus_bits: int = field(default_factory=lambda: CT_MODULUS_BITS)
    hybrid_modulus_bits: int = field(default_factory=lambda: HYBRID_MODULUS_BITS)
    prime_bits: int = field(default_factory=lambda: RNS_MODULUS_BITS)
    storage_word_bytes: int = field(default_factory=lambda: INT_BYTES)
    key_switch_technique: str = field(default_factory=lambda: KEY_SWITCH_TECHNIQUE)
    key_switch_digits: int | None = field(default_factory=lambda: KEY_SWITCH_DIGITS)

    @property
    def q_prime_bits(self):
        return self._prime_widths(self.ciphertext_modulus_bits)

    @property
    def p_prime_bits(self):
        return self._prime_widths(self.hybrid_modulus_bits) if self.key_switch_technique == "HYBRID" else ()

    def _prime_widths(self, bits):
        full, remainder = divmod(bits, self.prime_bits)
        return (self.prime_bits,)*full + ((remainder,) if remainder else ())

    @property
    def rns_primes(self):
        return len(self.q_prime_bits)

    @property
    def auxiliary_primes(self):
        return len(self.p_prime_bits)

    @property
    def switch_digits(self):
        if self.key_switch_digits is not None:
            return self.key_switch_digits
        if self.key_switch_technique == "DIGIT":
            return self.rns_primes
        # Group whole Q towers into partitions whose bit sums fit the P budget.
        groups, used = 1, 0
        for bits in self.q_prime_bits:
            if used + bits > self.hybrid_modulus_bits:
                groups += 1
                used = 0
            used += bits
        return groups

    @property
    def key_basis_primes(self):
        return self.rns_primes + self.auxiliary_primes

    def __post_init__(self):
        if self.poly_degree < 2 or self.poly_degree & (self.poly_degree - 1):
            raise ValueError("poly_degree must be a power of two >= 2")
        if self.plaintext_modulus <= 2 or (self.plaintext_modulus - 1) % (2*self.poly_degree):
            raise ValueError("Batching requires plaintext_modulus = 1 mod (2*poly_degree)")
        if min(self.ciphertext_modulus_bits, self.prime_bits, self.storage_word_bytes,
               self.security_bits) <= 0:
            raise ValueError("RNS, storage and key-switch parameters must be positive")
        if self.prime_bits > 8*self.storage_word_bytes:
            raise ValueError("prime_bits does not fit storage_word_bytes")
        if self.key_switch_technique not in ("HYBRID", "DIGIT"):
            raise ValueError("key_switch_technique must be HYBRID or DIGIT")
        if self.key_switch_digits is not None and self.key_switch_digits <= 0:
            raise ValueError("key_switch_digits must be positive or None")
        if self.key_switch_technique == "HYBRID":
            if self.hybrid_modulus_bits < max(self.q_prime_bits):
                raise ValueError("HYBRID P budget must accommodate at least one Q tower")
            if self.switch_digits > self.rns_primes:
                raise ValueError("HYBRID cannot have more partitions than Q towers")
            # Match the cost model's contiguous, nearly equal tower partitions.
            partitions = [self.q_prime_bits[i*self.rns_primes//self.switch_digits:
                                            (i+1)*self.rns_primes//self.switch_digits]
                          for i in range(self.switch_digits)]
            if max(map(sum, partitions)) > self.hybrid_modulus_bits:
                raise ValueError("HYBRID P is smaller than a Q partition; increase P or digit count")


def bfvrns_operation_costs(config, packing="rotations"):
    """Editable operation-cost model; all transforms are single-prime N-point.

    Rotation-mode ciphertexts are coefficient-form at operation boundaries.
    Im2col caches input transforms across its whole multiply/accumulate pipeline.
    Weights and keys are cached in evaluation form. Costs are not exact OpenFHE
    traces. Hybrid basis conversion uses a naive CRT arithmetic proxy. Other
    RNS conversion, scale/round, randomness and memory work is counted separately.
    """
    n, r, d = config.poly_degree, config.rns_primes, config.switch_digits
    rn = r*n
    # ntt_q / intt_q: transforms modulo ONE RNS prime, not the entire basis.
    # ntt_t / intt_t: slot packing transforms modulo the plaintext modulus.
    costs = {
        "encode_input": {"intt_t": 1},
        "encode_weight": {"intt_t": 1, "ntt_q": r,
                          "rns_lift_coefficients": rn},
        "encode_bias": {"intt_t": 1, "rns_lift_coefficients": rn},
        "decode": {"ntt_t": 1},
        # Public-key encrypt: transform u, multiply by two cached public-key
        # polynomials, inverse-transform both products, add e0/e1 and Delta*m.
        "encrypt": {"ntt_q": r, "intt_q": 2*r,
                    "modular_multiplications": 2*rn, "modular_additions": 3*rn,
                    "sampled_coefficients": 3*n, "rns_lift_coefficients": rn},
        "decrypt": {"ntt_q": r, "intt_q": r,
                    "modular_multiplications": rn, "modular_additions": rn,
                    "scale_round_coefficients": rn},
        "multiply_plain": {"ntt_q": 2*r, "intt_q": 2*r,
                           "modular_multiplications": 2*rn},
        "add_ciphertext": {"modular_additions": 2*rn},
        "add_plain_bias": {"modular_additions": rn},
        # Automorphism, D full-Q digit transforms, multiply/accumulate with
        # evaluation key, inverse-transform two outputs, add untouched c0.
        "rotate": {"ntt_q": d*r, "intt_q": 2*r,
                   "modular_multiplications": 2*d*rn,
                   "modular_additions": (2*d + 1)*rn,
                   "coefficient_permutations": 2*rn,
                   "rns_digit_coefficients": d*rn},
        "keypair_generate": {"ntt_q": 2*r, "modular_multiplications": rn,
                             "modular_additions": rn, "sampled_coefficients": 3*n},
        "rotation_key_generate": {"ntt_q": (d + 1)*r,
                                  "modular_multiplications": d*rn,
                                  "modular_additions": 2*d*rn,
                                  "sampled_coefficients": 2*d*n,
                                  "coefficient_permutations": rn,
                                  "rns_digit_coefficients": d*rn},
        "comparison": {"comparisons": 1},
        "average_pool_add": {"scalar_additions": 1},
        "average_pool_divide": {"scalar_divisions": 1},
    }
    if config.key_switch_technique == "HYBRID":
        p, extended = config.auxiliary_primes, config.key_basis_primes
        en = extended*n
        # Approximate naive CRT basis conversion, in addition to the digit
        # multiply/accumulate. Sum partition widths in tower units = R.
        up_mult = n*r*extended
        up_add = n*(r-d)*extended
        down_mult = 2*n*r*p
        down_add = 2*n*r*max(p-1, 0)
        costs["rotate"] = {
            "ntt_q": d*r, "ntt_p": d*p,
            "intt_q": 2*r, "intt_p": 2*p,
            "modular_multiplications": 2*d*en + up_mult + down_mult + 2*rn,
            "modular_additions": 2*d*en + up_add + down_add + 3*rn,
            "coefficient_permutations": 2*rn,
            "rns_basis_extension_coefficients": d*en,
            "rns_moddown_coefficients": 2*rn,
        }
        costs["rotation_key_generate"] = {
            "ntt_q": (d+1)*r, "ntt_p": (d+1)*p,
            "modular_multiplications": 2*d*en,
            "modular_additions": 2*d*en,
            "sampled_coefficients": 2*d*n,
            "coefficient_permutations": en,
            "rns_digit_coefficients": d*en,
        }
    if packing == "im2col":
        # Keep the entire multiply/accumulate in evaluation form. Inputs are
        # transformed once and reused, and only final outputs are transformed back.
        costs["ciphertext_to_ntt"] = {"ntt_q": 2*r}
        costs["ciphertext_from_ntt"] = {"intt_q": 2*r}
        costs["multiply_plain"] = {"modular_multiplications": 2*rn}
        costs["im2col_copy"] = {"copied_elements": 1}
        costs["pack_slot_write"] = {"packed_slot_writes": 1}
    elif packing != "rotations":
        raise ValueError("packing must be rotations or im2col")
    return costs


def _he_phase(counts, costs, config):
    """Expand event counts by cost; keep transforms outside either owner total."""
    result = {}
    transform_total = Counter()
    transform_by_origin = {}
    arithmetic_units = ("modular_multiplications", "modular_additions",
                        "comparisons", "scalar_additions", "scalar_divisions")
    for owner in ("client", "server"):
        primitive_totals = Counter()
        transforms = Counter()
        operations = {}
        for event, count in sorted(counts[owner].items()):
            unit_cost = costs[event]
            total_cost = {unit: count*value for unit, value in unit_cost.items()}
            operations[event] = {"count": count, "per_call": dict(unit_cost),
                                 "total": total_cost}
            for unit, value in total_cost.items():
                if unit in ("ntt_q", "intt_q", "ntt_p", "intt_p", "ntt_t", "intt_t"):
                    transforms[unit] += value
                else:
                    primitive_totals[unit] += value
        result[owner] = {
            "counts": dict(counts[owner]), "operations": operations,
            "primitive_ops_excluding_transforms": dict(primitive_totals),
            "estimated_arithmetic_ops_excluding_transforms": sum(
                primitive_totals[unit] for unit in arithmetic_units),
        }
        transform_by_origin[owner] = dict(transforms)
        transform_total.update(transforms)
    n = config.poly_degree
    butterflies = (n // 2) * (n.bit_length() - 1)
    ntt_count = sum(transform_total["ntt_" + basis] for basis in ("q", "p", "t"))
    intt_count = sum(transform_total["intt_" + basis] for basis in ("q", "p", "t"))
    result["transforms"] = {
        "by_origin": transform_by_origin,
        "counts_by_modulus": dict(transform_total),
        "ntts": ntt_count, "intts": intt_count,
        "per_ntt": {"modular_multiplications": butterflies,
                    "modular_additions": 2*butterflies},
        "per_intt": {"modular_multiplications": butterflies + n,
                     "modular_additions": 2*butterflies},
        "estimated_modular_multiplications": ((ntt_count + intt_count)*butterflies
                                              + intt_count*n),
        "estimated_modular_additions": 2*(ntt_count + intt_count)*butterflies,
        "estimated_arithmetic_ops": (ntt_count + intt_count)*3*butterflies + intt_count*n,
    }
    events = Counter(counts["client"])
    events.update(counts["server"])
    result["counts"] = {
        "encodings": sum(events[event] for event in ("encode_input", "encode_weight", "encode_bias")),
        "decodings": events["decode"],
        "encryptions": events["encrypt"],
        "decryptions": events["decrypt"],
        "rotations": events["rotate"],
        "rotation_keys_generated": events["rotation_key_generate"],
        "keypairs_generated": events["keypair_generate"],
        "ciphertext_plaintext_multiplications": events["multiply_plain"],
        "ciphertext_additions": events["add_ciphertext"],
        "plaintext_bias_additions": events["add_plain_bias"],
        "ntts": ntt_count,
        "intts": intt_count,
    }
    return result


def _he_im2col_layer_counts(l, input_shape, config, duplicate_columns):
    """Paper Sec. IV-A packing: X'[M,K] @ W'[K,C], one column per CT.

    Handle each sample and convolution group separately. Use all N BFV slots;
    both rows are safe because there are no rotations. If M <= N, repeat each
    input column floor(N/M) times (capped by C) to compute output channels in
    parallel. Otherwise split columns into ceil(M/N) ciphertexts. Plaintext
    weights are cached and reused across column chunks and batch samples.
    """
    output_shape = _layer_output_shape(l, input_shape)
    if isinstance(l, nn.Conv2d):
        batch = input_shape[0]
        groups = l.groups
        rows = math.prod(output_shape[-2:])
        reduction = l.in_channels // groups * math.prod(_pair(l.kernel_size))
        channels = l.out_channels // groups
    else:
        # Independent dense samples: M=1, duplicate each scalar input across
        # output-neuron slots. Do not fuse different batch samples in one CT.
        batch = math.prod(input_shape[:-1])
        groups, rows, reduction, channels = 1, 1, l.in_features, l.out_features
    slots = config.poly_degree
    chunks = (rows + slots - 1) // slots
    copies = min(channels, slots // rows) if duplicate_columns and rows <= slots else 1
    output_blocks = (channels + copies - 1) // copies
    inputs = batch*groups*reduction*chunks
    outputs = batch*groups*output_blocks*chunks
    products = outputs*reduction
    weights = groups*reduction*output_blocks
    biases = groups*output_blocks if l.bias is not None else 0
    setup = {"client": Counter(), "server": Counter()}
    inference = {"client": Counter(), "server": Counter()}
    setup["server"].update(encode_weight=weights, encode_bias=biases)
    inference["client"].update(
        encode_input=inputs, encrypt=inputs, decrypt=outputs, decode=outputs,
        im2col_copy=batch*groups*rows*reduction if isinstance(l, nn.Conv2d) else 0,
        pack_slot_write=batch*groups*rows*reduction*copies)
    inference["server"].update(
        multiply_plain=products, add_ciphertext=outputs*(reduction-1),
        add_plain_bias=outputs if biases else 0,
        ciphertext_to_ntt=inputs, ciphertext_from_ntt=outputs)
    details = {
        "method": "im2col", "slots_per_ciphertext": slots,
        "matrix_shape_per_sample_group": {"m": rows, "k": reduction, "n": channels},
        "batch_samples": batch, "groups": groups,
        "duplicate_columns": duplicate_columns,
        "column_copies_per_ciphertext": copies,
        "ciphertexts_per_input_column": chunks,
        "output_channel_blocks": output_blocks,
        "rotation_offsets": [],
        "input_ciphertexts": inputs, "output_ciphertexts": outputs,
        "multiply_plain_results": products,
        "unique_weight_plaintexts": weights,
        "weight_plaintext_references_per_inference": products,
        "im2col_elements": batch*groups*rows*reduction,
        "input_matrix_bytes": batch*groups*rows*reduction*config.storage_word_bytes,
        "im2col_materialized_bytes": (batch*groups*rows*reduction*config.storage_word_bytes
                                      if isinstance(l, nn.Conv2d) else 0),
        "weight_matrix_elements": groups*reduction*channels,
        "output_matrix_elements": math.prod(output_shape),
        "padded_input_slots": inputs*slots - batch*groups*rows*reduction*copies,
        "unused_output_slots": outputs*slots - math.prod(output_shape),
    }
    return setup, inference, details


def _he_layer_counts(l, input_shape, config, packing="rotations", duplicate_columns=True):
    """Build dot-product baseline counts without materializing any HE tensors."""
    output_shape = _layer_output_shape(l, input_shape)
    if packing == "im2col" and isinstance(l, (nn.Conv2d, nn.Linear)):
        return _he_im2col_layer_counts(l, input_shape, config, duplicate_columns)
    setup = {"client": Counter(), "server": Counter()}
    inference = {"client": Counter(), "server": Counter()}
    details = {"method": packing, "dot_product_length": 0, "chunks_per_dot_product": 0,
               "rotation_offsets": [], "input_ciphertexts": 0,
               "output_ciphertexts": 0, "multiply_plain_results": 0}
    if isinstance(l, (nn.Conv2d, nn.Linear)):
        if isinstance(l, nn.Conv2d):
            length = l.in_channels // l.groups * math.prod(_pair(l.kernel_size))
            input_vectors = input_shape[0]*math.prod(output_shape[-2:])*l.groups
            output_channels = l.out_channels
        else:
            length = l.in_features
            input_vectors = math.prod(input_shape[:-1])
            output_channels = l.out_features
        slots = config.poly_degree // 2
        chunks = (length + slots - 1) // slots
        # Sum products across chunks BEFORE the rotation tree (linearity).
        reduction_steps = (min(length, slots) - 1).bit_length()
        offsets = [1 << step for step in range(reduction_steps)]
        outputs = math.prod(output_shape)
        inputs = input_vectors*chunks
        products = outputs*chunks
        weights = output_channels*chunks
        biases = output_channels if l.bias is not None else 0
        setup["server"].update(encode_weight=weights, encode_bias=biases)
        inference["client"].update(encode_input=inputs, encrypt=inputs,
                                    decrypt=outputs, decode=outputs)
        inference["server"].update(
            multiply_plain=products, rotate=outputs*reduction_steps,
            add_ciphertext=outputs*(chunks - 1 + reduction_steps),
            add_plain_bias=outputs if biases else 0)
        details.update(dot_product_length=length, chunks_per_dot_product=chunks,
                       rotation_offsets=offsets, input_ciphertexts=inputs,
                       output_ciphertexts=outputs, multiply_plain_results=products)
    else:
        inference["client"]["comparison"] = layer_to_nonlinear_ops(l, input_shape)
        if isinstance(l, nn.AdaptiveAvgPool2d):
            # Adaptive pooling bins may overlap for non-divisible dimensions.
            h, w = input_shape[-2:]
            oh, ow = output_shape[-2:]
            height_sum = sum(((i + 1)*h + oh - 1)//oh - i*h//oh for i in range(oh))
            width_sum = sum(((i + 1)*w + ow - 1)//ow - i*w//ow for i in range(ow))
            outputs = math.prod(output_shape)
            inference["client"].update(
                average_pool_add=math.prod(input_shape[:-2])*height_sum*width_sum - outputs,
                average_pool_divide=outputs)
    return setup, inference, details


def _he_layer_storage(details, setup, config):
    """Logical packed input/output and cached server-weight package sizes.

    Ciphertexts contain two Q-basis polynomials; cached server plaintexts
    contain one. Client input encodings have one polynomial modulo t and are
    reported separately as optional staging storage. Totals exclude staging,
    intermediate results, keys, allocator overhead and NTT workspaces, so they
    describe full retained packages rather than peak execution memory.
    """
    polynomial_bytes = config.poly_degree*config.rns_primes*config.storage_word_bytes
    ct_bytes = 2*polynomial_bytes
    inputs, outputs = details["input_ciphertexts"], details["output_ciphertexts"]
    weights, biases = setup["server"]["encode_weight"], setup["server"]["encode_bias"]
    storage = {
        "ciphertext_bytes": ct_bytes,
        "plaintext_bytes": polynomial_bytes,
        "input_ciphertexts": inputs,
        "output_ciphertexts": outputs,
        "total_ciphertexts": inputs + outputs,
        "weight_plaintexts": weights,
        "bias_plaintexts": biases,
        "total_plaintexts": weights + biases,
        "input_ciphertexts_bytes": inputs*ct_bytes,
        "output_ciphertexts_bytes": outputs*ct_bytes,
        "total_ciphertexts_bytes": (inputs + outputs)*ct_bytes,
        "cached_weight_plaintexts_bytes": weights*polynomial_bytes,
        "cached_bias_plaintexts_bytes": biases*polynomial_bytes,
        "total_plaintexts_bytes": (weights + biases)*polynomial_bytes,
        "input_plaintexts": inputs,
        "input_plaintext_bytes": config.poly_degree*config.storage_word_bytes,
        "input_plaintexts_bytes": inputs*config.poly_degree*config.storage_word_bytes,
        "weight_package_bytes_without_reuse": details.get(
            "weight_plaintext_references_per_inference", weights)*polynomial_bytes,
    }
    storage["total_layer_data_bytes"] = (storage["total_ciphertexts_bytes"]
                                          + storage["total_plaintexts_bytes"])
    return storage


def _he_model_storage_totals(layers):
    """Sum prepared packages across layers, separately identifying model input.

    These sums assume every layer package is retained independently and do not
    multiply by session inference count. Encoded weights are reused. The initial
    input is the first HE linear layer's fully prepared encrypted package.
    """
    fields = ("input_ciphertexts", "output_ciphertexts", "total_ciphertexts",
              "weight_plaintexts", "bias_plaintexts", "total_plaintexts",
              "input_ciphertexts_bytes", "output_ciphertexts_bytes",
              "total_ciphertexts_bytes", "total_plaintexts_bytes",
              "input_plaintexts", "input_plaintexts_bytes", "total_layer_data_bytes")
    totals = {"total_" + key if not key.startswith("total_") else key:
              sum(layer["storage"][key] for layer in layers.values()) for key in fields}
    first_name = next((name for name, layer in layers.items()
                       if layer["storage"]["input_ciphertexts"]), None)
    first = layers[first_name]["storage"] if first_name is not None else {}
    totals.update(
        initial_input_layer=first_name,
        initial_input_ciphertexts=first.get("input_ciphertexts", 0),
        initial_input_ciphertexts_bytes=first.get("input_ciphertexts_bytes", 0),
        storage_basis="independently retained layer packages; not peak memory; excludes keys and temporary workspace",
        plaintext_basis="cached Q-basis server weights/biases; client modulo-t input encodings reported separately",
    )
    return totals


def layer_to_he_ops(l: nn.Module, input_shape, config=None, cost_overrides=None,
                    packing="rotations", duplicate_columns=None):
    """Per-layer BFVRNS estimate. Shared key generation belongs to the model."""
    config = config or BFVRNSConfig()
    if duplicate_columns is None:
        duplicate_columns = IM2COL_DUPLICATE_COLUMNS
    costs = bfvrns_operation_costs(config, packing)
    if cost_overrides:
        costs.update(cost_overrides)
    setup, inference, details = _he_layer_counts(l, input_shape, config, packing, duplicate_columns)
    ct_bytes = 2*config.poly_degree*config.rns_primes*config.storage_word_bytes
    linear = isinstance(l, (nn.Conv2d, nn.Linear))
    return {
        "type": type(l).__name__, "input_shape": tuple(input_shape),
        "output_shape": _layer_output_shape(l, input_shape),
        "normal_linear_macs": layer_to_ops(l, input_shape),
        "normal_nonlinear_comparisons": layer_to_nonlinear_ops(l, input_shape),
        "packing": details,
        "storage": _he_layer_storage(details, setup, config),
        "communication": {
            "client_to_server_messages": int(linear),
            "server_to_client_messages": int(linear),
            "client_to_server_bytes": details["input_ciphertexts"]*ct_bytes,
            "server_to_client_bytes": details["output_ciphertexts"]*ct_bytes,
        },
        "setup": _he_phase(setup, costs, config),
        "per_inference": _he_phase(inference, costs, config),
    }


def model_to_he_ops(m: nn.Module, input_shape=INPUT_SHAPE, config=None,
                    num_inferences=1, cost_overrides=None, verbose=False,
                    packing="rotations", duplicate_columns=None):
    """Analytical BFVRNS baseline with reusable setup and client nonlinearities.

    The client packs/encrypts convolution patches or dense input chunks, reused
    across output channels. Server weights are encoded once and cached. Each
    output scalar is one ciphertext: multiply each chunk by plaintext weights,
    add chunks, then rotate/add at power-of-two offsets, then add bias. The
    client decrypts each linear layer, computes nonlinearities/average pooling,
    and repacks. This intentionally simple baseline does not pack outputs.

    Alternatively packing="im2col" uses the paper's rotation-free column
    packing (see model_to_he_im2col_ops). duplicate_columns=False selects the
    reference script's single-copy variant. The return schema is shared.

    Counts are protocol-dependent, not a generic BFVRNS performance prediction.
    Coefficient-form boundaries and configurable HYBRID or full-Q DIGIT cost
    approximations are used. NTT/iNTT totals stay outside owner arithmetic totals.
    Keygen and model encoding happen once per session, not per rotation or
    inference. Raw word-array sizes exclude serialization/network overhead.
    Float-to-integer quantization, client rescaling, wraparound, security/noise
    validation and real cryptography are not implemented by this estimator.
    """
    if not isinstance(num_inferences, int) or num_inferences < 1:
        raise ValueError("num_inferences must be a positive integer")
    config = config or BFVRNSConfig()
    if duplicate_columns is None:
        duplicate_columns = IM2COL_DUPLICATE_COLUMNS
    costs = bfvrns_operation_costs(config, packing)
    if cost_overrides:
        costs.update(cost_overrides)
    setup_counts = {"client": Counter(), "server": Counter()}
    inference_counts = {"client": Counter(), "server": Counter()}
    layers = {}
    offsets = set()
    communication = Counter()
    ciphertexts = Counter()
    for name, layer, shape, _ in _vgg_layer_shapes(m, input_shape):
        record = layer_to_he_ops(layer, shape, config, cost_overrides, packing, duplicate_columns)
        layers[name] = record
        offsets.update(record["packing"]["rotation_offsets"])
        communication.update(record["communication"])
        for key in ("input_ciphertexts", "output_ciphertexts", "multiply_plain_results"):
            ciphertexts[key] += record["packing"][key]
        for phase, target in (("setup", setup_counts), ("per_inference", inference_counts)):
            for owner in ("client", "server"):
                target[owner].update(record[phase][owner]["counts"])
        if verbose:
            print(f"{name:16s} {record['type']:18s} "
                  f"{record['packing']['input_ciphertexts']:>10,} input CTs "
                  f"{record['packing']['output_ciphertexts']:>10,} output CTs "
                  f"{record['per_inference']['server']['counts'].get('rotate', 0):>12,} rotations")
    setup_counts["client"].update(keypair_generate=1, rotation_key_generate=len(offsets))
    n, r, word = config.poly_degree, config.rns_primes, config.storage_word_bytes
    ct_bytes = 2*n*r*word
    rotation_key_bytes = (2*n*config.key_basis_primes*config.switch_digits*word
                          if offsets else 0)
    setup_key_bytes = len(offsets)*rotation_key_bytes
    # Encryption is performed by the client; the public key stays there.
    session_counts = {owner: setup_counts[owner] + Counter(
        {event: count*num_inferences for event, count in inference_counts[owner].items()})
        for owner in ("client", "server")}
    communication["total_messages"] = (communication["client_to_server_messages"]
                                        + communication["server_to_client_messages"])
    communication["total_bytes"] = (communication["client_to_server_bytes"]
                                     + communication["server_to_client_bytes"])
    ciphertexts["transmitted_ciphertexts"] = (ciphertexts["input_ciphertexts"]
                                               + ciphertexts["output_ciphertexts"])
    # Logical result events, not distinct allocations or peak live ciphertexts.
    ciphertexts["ciphertext_result_events_including_intermediates"] = (
        ciphertexts["input_ciphertexts"] + sum(
            inference_counts["server"][event]
            for event in ("multiply_plain", "rotate", "add_ciphertext", "add_plain_bias")))
    session_communication = {key: value*num_inferences for key, value in communication.items()}
    session_communication["client_to_server_messages"] += int(bool(offsets))
    session_communication["client_to_server_bytes"] += setup_key_bytes
    session_communication["total_messages"] += int(bool(offsets))
    session_communication["total_bytes"] += setup_key_bytes
    return {
        "config": {
            **asdict(config),
            "q_prime_bits": config.q_prime_bits,
            "p_prime_bits": config.p_prime_bits,
            "rns_primes": r,
            "auxiliary_primes": config.auxiliary_primes,
            "active_auxiliary_primes": config.auxiliary_primes if offsets else 0,
            "resolved_key_switch_digits": config.switch_digits,
            "security_modulus_bits": (config.ciphertext_modulus_bits
                                      + (sum(config.p_prime_bits) if offsets else 0)),
            "security_validated": False,
        },
        "num_inferences": num_inferences,
        "packing_method": packing,
        "assumptions": {
            "packing": ("im2col columns across all N slots; duplicate columns=" + str(duplicate_columns)
                        if packing == "im2col" else
                        "one dot product per ciphertext; use only one N/2-slot row"),
            "evaluation_form": ("NTT inputs once; cache weights; accumulate in NTT form; iNTT final outputs once"
                                if packing == "im2col" else "coefficient-form ciphertext operation boundaries"),
            "batching": "samples and convolution groups are packed independently",
            "nonlinearities": "decrypt/decode, client plaintext evaluation, encode/encrypt next linear input",
            "privacy": "client sees each linear output; this is not a model-private protocol",
            "key_switch_cost": (f"approximate {config.key_switch_technique} model, not exact library trace"
                                if offsets else "unused: no rotations or key switching"),
            "transforms": "single-prime N-point transforms; separate from owner totals",
            "arithmetic_ops": "unit-weight add/multiply/compare/divide proxy; not CPU cycles",
            "unpriced_work": "sampling, permutations, remaining RNS conversion/decomposition, scale/round, quantization",
            "parameters": f"target {config.security_bits}-bit classical security; security/primality/noise not validated",
            "storage": "uncompressed word arrays; traffic is cumulative, not peak memory",
        },
        "layers": layers,
        "normal_ops": model_to_ops(m, input_shape),
        "normal_storage": model_to_size(m, input_shape),
        "storage": {
            **_he_model_storage_totals(layers),
            "ciphertext_bytes": ct_bytes,
            "plaintext_bytes": n*r*word,
            "rotation_key_bytes": rotation_key_bytes,
            "all_rotation_keys_bytes": setup_key_bytes,
            "client_secret_key_bytes": n*r*word,
            "client_key_switch_secret_key_bytes": n*config.key_basis_primes*word if offsets else 0,
            "client_public_key_bytes": ct_bytes,
            "hybrid_extended_ciphertext_bytes": 2*n*config.key_basis_primes*word if offsets else 0,
            "cached_weight_plaintexts_bytes": setup_counts["server"]["encode_weight"]*n*r*word,
            "cached_bias_plaintexts_bytes": setup_counts["server"]["encode_bias"]*n*r*word,
            "total_im2col_materialized_bytes": sum(
                record["packing"].get("im2col_materialized_bytes", 0) for record in layers.values()),
        },
        "rotation_offsets": sorted(offsets),
        "ciphertexts_per_inference": dict(ciphertexts),
        "communication": {
            "setup": {"client_to_server_messages": int(bool(offsets)),
                      "server_to_client_messages": 0,
                      "client_to_server_bytes": setup_key_bytes,
                      "server_to_client_bytes": 0},
            "per_inference": dict(communication), "session": session_communication,
        },
        "setup": _he_phase(setup_counts, costs, config),
        "per_inference": _he_phase(inference_counts, costs, config),
        "session": _he_phase(session_counts, costs, config),
    }


def layer_to_he_im2col_ops(l: nn.Module, input_shape, config=None,
                           cost_overrides=None, duplicate_columns=None):
    """Rotation-free column-packed matrix multiplication estimate for one layer."""
    return layer_to_he_ops(l, input_shape, config, cost_overrides,
                           packing="im2col", duplicate_columns=duplicate_columns)


def model_to_he_im2col_ops(m: nn.Module, input_shape=INPUT_SHAPE, config=None,
                           num_inferences=1, cost_overrides=None, verbose=False,
                           duplicate_columns=None):
    """Paper Sec. IV-A im2col packing, with no rotations or rotation keys.

    M=Hout*Wout, K=(Cin/groups)*Kh*Kw, C=Cout/groups for convolutions.
    Each input ciphertext holds a column of X'[M,K], possibly split or repeated.
    For dense layers use X'[1,in_features] per sample. Ciphertext products
    are added across K, leaving independent output pixels/neurons in slots.
    Nonlinearities still run after client decrypt/decode. No HE tensors or
    im2col matrices are actually allocated. See BFVRNS_ESTIMATION.md for formulas.
    """
    return model_to_he_ops(m, input_shape, config, num_inferences, cost_overrides,
                           verbose, packing="im2col", duplicate_columns=duplicate_columns)


def layer_to_size(l: nn.Module, input_shape, activation_dtype=torch.float32):
    """Return logical tensor sizes in elements and bytes, without a forward pass.

    Parameter bytes include weights and biases at their existing dtypes.
    Activation bytes assume every element uses activation_dtype. Tensor metadata,
    allocator overhead, temporary workspaces, gradients and optimizer state are
    excluded. A layer's input/output sizes describe tensors, not new allocations.
    """
    output_shape = _layer_output_shape(l, input_shape)
    element_bytes = torch.empty((), dtype=activation_dtype).element_size()
    parameters = list(l.parameters(recurse=False))
    buffers = list(l.buffers(recurse=False))
    input_elements = math.prod(input_shape)
    output_elements = math.prod(output_shape)

    return {
        "type": type(l).__name__,
        "input_shape": tuple(input_shape),
        "output_shape": output_shape,
        "parameter_elements": sum(p.numel() for p in parameters),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "buffer_elements": sum(b.numel() for b in buffers),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
        "input_elements": input_elements,
        "input_bytes": input_elements * element_bytes,
        "output_elements": output_elements,
        "output_bytes": output_elements * element_bytes,
    }


def model_to_size(m: nn.Module, input_shape=INPUT_SHAPE,
                  activation_dtype=None, verbose=False):
    """Return per-layer and total plaintext tensor storage estimates for VGG.

    Activation dtype defaults to the first parameter's dtype (FP32 for the
    default VGG16). Parameters/buffers retain their actual dtypes and are counted
    once in model totals. Batch size affects activations, not model storage.

    total_output_bytes sums logical outputs of every layer. The
    total_bytes_with_all_outputs entry assumes independent copies of all those
    outputs plus the model input and parameters/buffers are retained. It is NOT
    peak inference memory: in-place ReLU and inference dropout can share storage,
    and inference can release intermediate tensors. Per-layer inputs are not
    added again, and flattening is a view. No HE expansion is included.
    """
    if activation_dtype is None:
        first_parameter = next(m.parameters(), None)
        activation_dtype = (first_parameter.dtype if first_parameter is not None
                            else torch.float32)
    element_bytes = torch.empty((), dtype=activation_dtype).element_size()
    parameters = list(m.parameters())
    buffers = list(m.buffers())

    result = {
        "layers": {},
        "activation_dtype": str(activation_dtype),
        "total_parameter_elements": sum(p.numel() for p in parameters),
        "total_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "total_buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
        "input_bytes": math.prod(input_shape) * element_bytes,
        "total_output_elements": 0,
        "total_output_bytes": 0,
    }

    for name, layer, shape, _ in _vgg_layer_shapes(m, input_shape):
        sizes = layer_to_size(layer, shape, activation_dtype)
        result["layers"][name] = sizes
        result["total_output_elements"] += sizes["output_elements"]
        result["total_output_bytes"] += sizes["output_bytes"]
        if verbose:
            print(f"{name:16s} {sizes['type']:18s} "
                  f"{sizes['parameter_bytes']:>15,} parameter bytes "
                  f"{sizes['buffer_bytes']:>12,} buffer bytes "
                  f"{sizes['output_bytes']:>12,} output bytes")
    result["total_model_bytes"] = (result["total_parameter_bytes"]
                                   + result["total_buffer_bytes"])
    result["total_bytes_with_all_outputs"] = (result["total_model_bytes"]
                                               + result["input_bytes"]
                                               + result["total_output_bytes"])

    return result






def print_he_storage(he):
    """Print counts and memory for every HE layer using the selected packing."""
    mib = 1024**2
    storage = he["storage"]
    print(f"\nHE storage ({he['packing_method']}; sizes in MiB):")
    print(f"Prepared initial input ({storage['initial_input_layer']}): "
          f"{storage['initial_input_ciphertexts']:,} ciphertexts, "
          f"{storage['initial_input_ciphertexts_bytes']/mib:,.2f} MiB")
    print(f"{'Layer':16s} {'Input CTs':>10s} {'Input MiB':>12s} "
          f"{'Output CTs':>11s} {'Output MiB':>12s} "
          f"{'Weight PTs':>11s} {'Bias PTs':>9s} {'PT MiB':>12s} {'Total MiB':>12s}")
    for name, layer in he["layers"].items():
        sizes = layer["storage"]
        print(f"{name:16s} {sizes['input_ciphertexts']:>10,} "
              f"{sizes['input_ciphertexts_bytes']/mib:>12,.2f} "
              f"{sizes['output_ciphertexts']:>11,} "
              f"{sizes['output_ciphertexts_bytes']/mib:>12,.2f} "
              f"{sizes['weight_plaintexts']:>11,} {sizes['bias_plaintexts']:>9,} "
              f"{sizes['total_plaintexts_bytes']/mib:>12,.2f} "
              f"{sizes['total_layer_data_bytes']/mib:>12,.2f}")
    print(f"All prepared layer inputs: {storage['total_input_ciphertexts']:,} CTs, "
          f"{storage['total_input_ciphertexts_bytes']/mib:,.2f} MiB")
    print(f"All layer input/output ciphertexts: {storage['total_ciphertexts']:,}, "
          f"{storage['total_ciphertexts_bytes']/mib:,.2f} MiB")
    print(f"All cached weight/bias plaintexts: {storage['total_plaintexts']:,}, "
          f"{storage['total_plaintexts_bytes']/mib:,.2f} MiB")
    print(f"All layer packages: {storage['total_layer_data_bytes']/mib:,.2f} MiB "
          "(excludes keys, client encoding staging and temporary workspace; not peak memory)")


def fully_analyze_model(model):
    # Weights do not affect the analytical count; avoid downloading them.
    print(model)
    print("")

    result = model_to_ops(model, verbose=True)
    print(f"Linear-layer MACs: {result['total_linear_macs']:,}")
    print(f"Nonlinear-layer comparisons: {result['total_nonlinear_comparisons']:,}")
    sizes = model_to_size(model, verbose=True)
    print(f"Model storage: {sizes['total_model_bytes']:,} bytes")
    print(f"Input storage: {sizes['input_bytes']:,} bytes")
    print(f"Sum of layer output sizes: {sizes['total_output_bytes']:,} bytes")
    print(f"Storage if all outputs are copied and retained: "
          f"{sizes['total_bytes_with_all_outputs']:,} bytes (not peak memory)")


    he = model_to_he_ops(model, packing="rotations")
    print("\nBFVRNS rotations (target parameters; see returned assumptions):")
    print(f"BFVRNS parameters: {he['config']}")
    print(f"Packing method: {he['packing_method']}")
    print(f"One-time rotation keys: {len(he['rotation_offsets']):,}")
    print(f"Per-inference communication: {he['communication']['per_inference']}")
    for owner in ("client", "server"):
        print(f"{owner.capitalize()} per-inference counts: {he['per_inference'][owner]['counts']}")
        print(f"{owner.capitalize()} arithmetic excluding transforms: "
              f"{he['per_inference'][owner]['estimated_arithmetic_ops_excluding_transforms']:,}")
    print(f"Per-inference NTTs: {he['per_inference']['transforms']['ntts']:,}")
    print(f"Per-inference iNTTs: {he['per_inference']['transforms']['intts']:,}")
    print(f"Transform arithmetic (separate): "
          f"{he['per_inference']['transforms']['estimated_arithmetic_ops']:,}")
    # print(f"Storage Stats: "
    #     f"{he['storage']}")
    print_he_storage(he)


    he = model_to_he_ops(model, packing="im2col")
    print("\nBFVRNS im2col matmul (target parameters; see returned assumptions):")
    print(f"BFVRNS parameters: {he['config']}")
    print(f"Packing method: {he['packing_method']}")
    print(f"One-time rotation keys: {len(he['rotation_offsets']):,}")
    print(f"Per-inference communication: {he['communication']['per_inference']}")
    for owner in ("client", "server"):
        print(f"{owner.capitalize()} per-inference counts: {he['per_inference'][owner]['counts']}")
        print(f"{owner.capitalize()} arithmetic excluding transforms: "
              f"{he['per_inference'][owner]['estimated_arithmetic_ops_excluding_transforms']:,}")
    print(f"Per-inference NTTs: {he['per_inference']['transforms']['ntts']:,}")
    print(f"Per-inference iNTTs: {he['per_inference']['transforms']['intts']:,}")
    print(f"Transform arithmetic (separate): "
          f"{he['per_inference']['transforms']['estimated_arithmetic_ops']:,}")
    # print(f"Storage Stats: "
    #     f"{he['storage']}")
    print_he_storage(he)

if __name__ == "__main__":
    model = vgg16(weights=None)
    fully_analyze_model(model)
