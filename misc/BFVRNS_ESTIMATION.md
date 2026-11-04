# BFVRNS analytical estimator

`he_computation_estimation.py` estimates inference protocols without
running encryption, a model forward pass, or a profiler:

```python
config = BFVRNSConfig(
    poly_degree=16384,
    plaintext_modulus=65537,
    security_bits=128,
    ciphertext_modulus_bits=320,
    hybrid_modulus_bits=60,
    prime_bits=60,
    storage_word_bytes=8,
    key_switch_technique="HYBRID",
    key_switch_digits=None,
)
he = model_to_he_ops(model, config=config, num_inferences=10)

he["layers"]["features.0"]
he["setup"]["client"]["counts"]
he["per_inference"]["server"]["operations"]["rotate"]
he["session"]["transforms"]
he["communication"]
he["storage"]
```

For the rotation-free im2col method, use:

```python
he = model_to_he_im2col_ops(model, config=config)
# Same result schema; single-copy variant of the reference script:
single_copy = model_to_he_im2col_ops(model, config=config, duplicate_columns=False)
# The original method is still available:
rotations = model_to_he_ops(model, config=config)
```

`main()` selects its method using `HE_PACKING="im2col"` (or `"rotations"`).
`IM2COL_DUPLICATE_COLUMNS=True` selects the paper's repeated-column packing.
These globals do not change the original `model_to_he_ops` default, preserving
its API. The im2col wrapper always selects im2col.

The defaults come from the global declarations near the top of the script:
`POLY_DEGREE`, `SECURITY_BITS`, `CT_MODULUS_BITS`, `HYBRID_MODULUS_BITS`,
`RNS_MODULUS_BITS`, `INT_BITS`/`INT_BYTES`, `PLAINTEXT_MODULUS`,
`KEY_SWITCH_TECHNIQUE`, and `KEY_SWITCH_DIGITS`. `BFVRNSConfig()` reads them
when it is constructed; editing the declarations above `main()` is sufficient.

The requested parameters target classical 128-bit security; they are not a
security/noise-validated context. The 320-bit Q budget is represented as prime
widths (60, 60, 60, 60, 60, 20), not six 60-bit primes. There is one 60-bit P
prime for HYBRID switching. All coefficients use 64-bit words for storage.
The combined QP bit budget is 380. Real primes are not generated, so these
widths describe an analytical budget rather than an actual modulus chain.

With HYBRID switching, security depends on QP, not Q alone, as explained in
[OpenFHE's parameter guidance](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md).
The returned config distinguishes the target from `security_validated=False`.
BFV
computes integers modulo t. The float VGG16 model supplies architecture and
shapes only; quantization, fixed-point scale management, signed interpretation,
and avoiding modular wraparound require a separate numerical design. Client
quantization/rescaling costs are not priced here.

## Prepared HE input and per-layer storage

Both methods expose counts and byte sizes together:

```python
he = model_to_he_im2col_ops(model)  # Or model_to_he_ops(model).
layer = he["layers"]["features.0"]["storage"]
layer["input_ciphertexts"], layer["input_ciphertexts_bytes"]
layer["output_ciphertexts"], layer["output_ciphertexts_bytes"]
layer["weight_plaintexts"], layer["bias_plaintexts"]
layer["total_plaintexts"], layer["total_plaintexts_bytes"]
layer["total_ciphertexts"], layer["total_ciphertexts_bytes"]
layer["total_layer_data_bytes"]

he["storage"]["initial_input_ciphertexts"]
he["storage"]["initial_input_ciphertexts_bytes"]
he["storage"]["total_input_ciphertexts"]
he["storage"]["total_input_ciphertexts_bytes"]
he["storage"]["total_plaintexts"]
he["storage"]["total_plaintexts_bytes"]
print_he_storage(he)
```

The initial input is the first HE linear layer's package after method-specific
preprocessing and packing. Thus the im2col count uses column chunks/copies,
while rotation packing uses encrypted dot-product input patches/chunks.
`total_input_ciphertexts` sums all layers' prepared encrypted inputs rather
than just the original image. Model-level `total_*` fields sum corresponding
per-layer counts/bytes; storage is not multiplied by `num_inferences`.
`main()` prints these tables for both methods.

One ciphertext occupies 2*N*R*word_bytes. Each cached server plaintext
(weights or biases) occupies N*R*word_bytes in its Q-basis representation.
Plaintext counts refer to reusable packed layouts, not scalar weight counts
or the number of multiplications. All cached biases are included.
Client modulo-t input encodings require N*word_bytes each and are reported
separately by `input_plaintexts` and `input_plaintexts_bytes`. They are staging
data for encryption and are excluded from the server plaintext totals.

Per-layer `total_layer_data_bytes` is input ciphertext bytes + output ciphertext
bytes + cached weight/bias plaintext bytes. The model total assumes all layer
packages are retained independently. These package totals exclude keys (reported
separately), client input-encoding staging, intermediate multiplication/rotation
results, allocator metadata, and NTT workspace. They do not predict peak memory
or require all intermediate result events to be materialized. Streaming can
reduce retained data. Client-only nonlinear layers report zero HE objects;
their ordinary tensor sizes remain in `normal_storage`.

## Rotation-free im2col packing

This follows Section IV-A and the evaluation-space accumulation in Section IV-B
of the supplied [paper](</home/mrcorcoran/Downloads/Performance_Modeling_and_FPGA_Acceleration_of_Homomorphic_Encrypted_Convolution.pdf>)
by Ye et al., FPL 2021, DOI 10.1109/FPL53798.2021.00027. It also supports the
single-copy layout in the supplied
[packaging script](/home/mrcorcoran/projects/PIPO-RECON/misc/feasibility/server_computation/im2col_data_gen.py).
The source files are read as references and are not imported or executed.

For one sample and convolution group, build the logical multiplication
X'[M,K] @ W'[K,C], with M=Hout*Wout, K=(Cin/groups)*Kh*Kw, and C=Cout/groups.
Unlike the dot-product baseline, ciphertext slots hold *different output
positions*, not terms of one output's dot product. Accumulating across K separate
ciphertexts sums each output's terms in the same slot, requiring no rotation.
Both N/2-slot rows can therefore be used, giving N slots per ciphertext.

Let T=ceil(M/N). For M>N, split each input column into T chunks, with one
column copy per ciphertext. For M<=N, repeat the column
D=min(C,floor(N/M)) times, with T=1. Each copy computes a different output
channel. The single-copy variant forces D=1. Let Bc=ceil(C/D), and let B be
batch samples and G be convolution groups. Counts per inference are:

| Quantity | Exact count under this layout |
|---|---:|
| Input ciphertexts / client encodings / encryptions | B*G*K*T |
| Output ciphertexts / client decryptions / decodings | B*G*Bc*T |
| Ciphertext-plaintext multiplications | B*G*Bc*T*K |
| Ciphertext additions | B*G*Bc*T*(K-1) |
| Plaintext bias additions, if bias exists | B*G*Bc*T |
| Unique cached weight encodings | G*K*Bc |
| Unique cached bias encodings, if bias exists | G*Bc |
| Rotations and rotation keys | 0 |

All shapes use actual convolution output dimensions, including stride,
padding, dilation, non-square kernels, and groups. Ceiling counts handle partial
chunks and the final incomplete output-channel block. Samples and groups are
packed independently; batch samples are not fused in spare slots. Dense layers
use M=1, K=in_features, C=out_features independently per sample, so repeated
scalar inputs compute multiple output neurons at once.

Client matrix preparation records B*G*M*K logical element copies for
convolutions and B*G*M*K*D populated input-slot writes. These are reported
as memory operations, not included in arithmetic totals. Dense inputs require
no im2col materialization. Padded slots and unused output slots are reported.
The total materialized im2col bytes sums full integer matrices for convolution
layers and is not peak memory; a streaming implementation can avoid them.

The server keeps inputs and intermediate products in NTT form: transform each
input ciphertext once (2R forward NTTs), reuse cached NTT-form weights, perform
all pointwise products/additions, then transform each completed output once
(2R inverse NTTs). Bias is added in coefficient form after that conversion.
The im2col `multiply_plain` cost therefore contains no transforms. Client
encoding/encryption/decryption/decoding costs remain included. Transforms remain
in the separate transform bucket even though their default origin is recorded.

There is no key switching, relinearization, or auxiliary P basis in this method.
The configured P budget is retained in the config, but `active_auxiliary_primes`
and all rotation-key storage are zero. The active security modulus is Q alone.
Keypair generation still occurs once; the key never leaves the client in this
protocol. Every linear layer still has two messages, followed by client-side
decrypt/decode and plaintext nonlinear computation as previously requested.

The unfinished reference script creates duplicate weight plaintext objects for
each spatial chunk, whereas its size estimate counts unique weights. This
estimator caches/reuses each distinct *layout* across chunks and samples:
`unique_weight_plaintexts` is the cached count, and
`weight_plaintext_references_per_inference` counts all uses. Per-layer
`weight_package_bytes_without_reuse` reports the size if every reference is
materialized independently, matching that duplication behavior. Equal-valued
weights across unrelated matrix entries are not deduplicated. With repeated
columns, each weight plaintext holds different weights in its output-channel
blocks; it is no longer one broadcast scalar across all N slots.

The paper's rough product count M*K*C/N omits ceiling/padding effects. This
implementation uses the explicit counts above, which matter for small images
and fully connected layers.

## Original rotation-based packing and protocol

BFV batching is arranged into two rows of N/2 slots. This estimator uses only
one row, leaving the other zero, so rotations need no row swap. It checks the
necessary congruence t = 1 mod 2N; it does not test primality or select Q primes.

For a dot product of length K, let S = N/2, C = ceil(K/S), and
J = ceil(log2(min(K, S))). Inputs/weights are zero padded. Multiply each of the
C encrypted chunks by its plaintext weight chunk, add those ciphertexts, then
perform J rotate-and-add steps at offsets 1, 2, 4, ..., 2^(J-1). Slot zero
contains the dot product. Add a plaintext bias if present.

For O output scalars, count O*C multiply-plain operations, O*J rotations, and
O*(C-1+J) ciphertext additions. One ciphertext is returned per output scalar.
No output extraction/repacking on the server is assumed.

The client prepares each convolution patch/group once and reuses its encrypted
chunks across that group's output channels. Dense input chunks are reused
across output neurons. Server weight encodings are reused across spatial
locations and batch samples. Bias encodings are reused across output positions.
There is no cross-layer or cross-inference reuse of input ciphertexts.

After **every linear layer**, the server sends its ciphertext outputs to the
client. The client decrypts/decodes them and runs ReLU, max pooling, average
pooling, and inference dropout locally before preparing the next input. This
baseline reveals intermediate linear outputs to the client; it does not provide
model-private nonlinear evaluation. It deliberately uses inefficient output
packing, so its traffic is not representative of optimized HE inference.

## Counts, setup, and communication

- `setup`: one client keypair, one rotation key per distinct offset across the
  model, and server weight/bias encodings. Keys and encodings are reused.
- `per_inference`: client encodings/encryptions/decryptions/decodings, server
  homomorphic operations, and client plaintext operations.
- `session`: setup plus `num_inferences` times inference work.
- A transfer means a **logical one-way message**, not a TCP packet or a
  ciphertext. Each linear layer contributes one input message and one output
  message. All rotation keys are sent together in one setup message. The public
  key stays with the client because the server does not encrypt in this protocol.
- Ciphertext counters distinguish encrypted inputs, returned outputs, and
  multiplication results. The additional result-event count includes logical
  intermediate results; it is not an allocation count or peak live memory.
- Word-array ciphertext bytes = 2*N*R*word_bytes, where R is the RNS prime count.
  A rotation key is approximated as D pairs of polynomials in the QP basis
  (Q only for DIGIT): 2*N*(R+A)*D*word_bytes, where A is the number of P primes.
  With the defaults R=6, A=1, D=6: ciphertexts are 1,572,864 bytes and rotation
  keys are 11,010,048 bytes each. Serialized
  wire bytes, compression, metadata, temporary allocations, and transport
  overhead are excluded. Traffic totals are cumulative bytes, not peak storage.

## Rotation-baseline operation costs and transforms

Each operation record contains `count`, `per_call`, and `total = count*per_call`.
The cost model is exposed by `bfvrns_operation_costs(config)` and can be replaced
per event through `cost_overrides={"rotate": {...}}`. An override replaces that
event's entire cost dictionary. Use this to match a chosen implementation.

The baseline assumes coefficient-form ciphertexts at operation boundaries,
evaluation-form cached weights/keys, and no transform hoisting. The default
HYBRID approximation partitions whole Q towers into D groups fitting the P
bit budget. Here D=6, one Q tower per partition. `KEY_SWITCH_DIGITS` can override
the partition count, provided each partition fits P. P must accommodate at least
one Q tower. Extra correctness/noise headroom required by a library is not
validated here. This is **not an exact OpenFHE HYBRID trace**.
The earlier full-Q digit model remains available as `KEY_SWITCH_TECHNIQUE="DIGIT"`;
it ignores P and defaults to D=R. Neither mode models rotation hoisting.
No ciphertext-ciphertext multiplication, relinearization keys, bootstrapping,
or ciphertext level dropping is assumed. HYBRID's internal QP-to-Q modulus
down conversion is included as an arithmetic approximation.

| Event | Forward NTTs modulo Q primes | Inverse NTTs modulo Q primes |
|---|---:|---:|
| Encrypt | R | 2R |
| Decrypt | R | R |
| Multiply plaintext (cached evaluation form) | 2R | 2R |
| Rotate, including approximate digit key switch | DR | 2R |
| Encode/cache a weight | R | 0 |
| Generate keypair | 2R | 0 |
| Generate one rotation key | (D+1)R | 0 |

For HYBRID, add D*A forward and 2*A inverse transforms per rotation, and
(D+1)*A forward transforms per generated rotation key. All switching-key
pointwise products/accumulations operate on E=R+A primes. The arithmetic
proxy for naive basis conversion adds N*R*E multiplications and N*(R-D)*E
additions for lifting all partitions, plus 2*N*R*A multiplications and
2*N*R*max(A-1,0) additions for reducing two components. Final P-inverse scaling
and subtraction add another 2*N*R multiplications and 2*N*R additions.
The unmodified c0 addition contributes N*R additions. These conversion formulas
are intentionally simple proxies; optimized RNS conversion will differ.

Every encoding also has one inverse N-point transform modulo t; decoding has
one forward transform modulo t. Transform totals distinguish q, p, and t, retain
their client/server origin, and remain **outside** client/server arithmetic
totals so their placement can be assigned independently.

One radix-2 N-point NTT is priced as B = (N/2)*log2(N) butterflies, each one
modular multiplication plus two modular additions. The inverse adds N modular
multiplications for normalization. Twist factors, implementation-specific
reductions, precomputation, and memory operations are excluded. A single
transform here means one polynomial modulo **one prime**, not an RNS transform
over all R primes. Operations modulo t and q are tracked separately even though
the scalar-operation proxy weights them equally, including the shorter last Q prime.

Modular add/multiply, plaintext compare/add/divide counts are summed as a rough
arithmetic proxy, not CPU instructions, bit operations, FLOPs, or time. Sampling,
coefficient permutations, remaining RNS lifts/digit conversions, and scale-and-round are
reported in their own coefficient units and left unpriced. Their implementation
costs can be supplied in event cost overrides. Consequently the arithmetic
subtotal is not a complete runtime estimate.

## References and checks

The protocol and costs above are modeling choices informed by OpenFHE's
[BFVRNS rotation example](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/rotation.cpp),
[public-key encryption implementation](https://openfhe-development.readthedocs.io/en/latest/api/program_listing_file_pke_lib_schemebase_base-pke.cpp.html),
and [performance notes on key switching](https://github.com/openfheorg/openfhe-development/blob/main/docs/static_docs/Best_Performance.md).
These sources do not establish the estimator's approximate operation counts.

Run the small-model accounting tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s misc -p 'test_he_computation_estimation.py' -v
```

The tests cover chunked dot products, the rotate-and-add algorithm, input/key
reuse, session scaling, cost expansion, transform separation, transfer bytes,
grouped convolutions, global parameter updates, Q/P limb sizing, hybrid auxiliary
transforms, and parameter checks. Im2col tests cover both packing variants,
partial output blocks, long columns, dense layers, input/weight reuse, no
rotation keys or P transforms, cached transform counts, and exact integer
slot-by-slot agreement with matrix multiplication.
