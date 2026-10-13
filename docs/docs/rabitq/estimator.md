# Estimator

Estimate squared L2 or inner-product distance from quantized codes and stored
factors. For API usage, see [Stable core API](core_api.md) or the advanced
[Quantizer examples](quantizer.md).

| Notation     | Description                            |
|----------|----------------------------------------|
| $\mathbf{o}_r$, $\mathbf{q}_r$ | The original data and query vectors, zero-padded to the working dimension. |
| $\mathbf{c}$ | The centroid in the same coordinate system as the original vectors. |
| $D$ | The working (padded) dimension, $D>1$. |
| $B$ | Total bits per dimension, including the sign bit. |
| $c_B$| A constant value $c_B=- \frac{2^B-1}{2}$. |
| $\mathbf{1}_D$ | The all-one vector of dimension $D$. |
| $P$ | Orthogonal map from the quantized coordinate system back to the original padded domain. |
| $\mathbf{o}$, $\mathbf{q}$ | Normalized residuals: $(\mathbf{o}_r-\mathbf{c})/\|\mathbf{o}_r-\mathbf{c}\|$ and $(\mathbf{q}_r-\mathbf{c})/\|\mathbf{q}_r-\mathbf{c}\|$. |
| $\mathbf{\bar o}$ | Unit-length quantized residual direction, expressed in the original padded domain. |
| $\mathbf{q}_r'$ | The query in the quantized coordinate system: $\mathbf{q}_r'=P^{-1}\mathbf{q}_r$. |
| $S_q$ | $S_q=\sum_{i=1}^{D} \mathbf{q}'_{r}[i].$ |
| $\Delta_x$ | Residual-norm scale defined below; distinct from the estimator factor `F_rescale`. |
| $\mathbf{x}_u$ | Unsigned integer codes.|

## Quantization

The derivation assumes nonzero residual norms. Zero residuals are handled by
the encoder; callers need not normalize residuals themselves.

Here $P^{-1}$ denotes the forward preprocessing transform applied by
[the rotator](rotator.md). Apply that same transform to data, centroids and
queries; no separate inverse-rotation call is needed for queries.

For unsigned codes $\mathbf{x}_u$, define

$$
\mathbf{z}=\mathbf{x}_u+c_B\mathbf{1}_D,\qquad
\mathbf{\bar o}=\frac{P\mathbf{z}}{\|\mathbf{z}\|},\qquad
\Delta_x=\frac{\|\mathbf{o}_r-\mathbf{c}\|}{\|\mathbf{z}\|}.
$$

Then the norm-preserving reconstruction satisfies

$$
\begin{align}
\mathbf{o}_r - \mathbf{c} \approx \| \mathbf{o}_r-\mathbf{c} \| \cdot \mathbf{\bar o} = \Delta_x P(\mathbf{x}_u+c_B\mathbf{1}_D)
\end{align}
$$

## Estimator

The following derivation covers squared Euclidean distance and inner product.
The public metric choices are `METRIC_L2` and `METRIC_IP`; there is no separate
cosine metric. For cosine search, normalize both data and query vectors, then
use inner product. The implementation represents inner-product distance as
$1-\langle\mathbf{o}_r,\mathbf{q}_r\rangle$, which preserves the same ranking while
keeping the nearest-neighbor convention that smaller values are better.

Here $\epsilon_0$ controls the width of the probabilistic error interval. The current
encoder uses the internal constant `kConstEpsilon = 1.9`; it is not a public
configuration parameter. These intervals support approximate pruning, not a
deterministic guarantee for every vector or query. Validate recall for the target workload.
The expressions below describe the data-code error with an unquantized query
under the random-rotation model. They do not include floating-point roundoff or
additional query/LUT quantization error. The one-bit probabilistic bound is
derived in [RaBitQ, Section 3.2](https://arxiv.org/pdf/2405.12497).

### Estimator of Euclidean distance

$$
\begin{align}
&\| \mathbf{o}_r-\mathbf{q}_r\|^2
\\=& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2 \left< \mathbf{o}_r - \mathbf{c}, \mathbf{q}_r - \mathbf{c}\right>
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \left<\mathbf{ o},\mathbf{q} \right>
\\ \approx& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \frac{\left<\mathbf{\bar o},\mathbf{q} \right>}{ \left<\mathbf{\bar o},\mathbf{o} \right>}
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\| \cdot \frac{\left<\mathbf{\bar o},\mathbf{q}_r-\mathbf{c} \right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 +2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{2}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left< \| \mathbf{o}_r-\mathbf{c}\|\cdot P^{-1}\mathbf{\bar o}, \mathbf{q}_r' \right>
\\=&\|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 +2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{2\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\end{align}
$$

The probabilistic error interval is

$$
\begin{align}
\pm \ 2 \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}
\end{align}
$$

Store the following factors (`F_` for data and `G_` for the query).

| Name (Type) of Variable | Description |
| ----------------- | ----------- |
| `F_add (float)`| $\|\mathbf{o}_r-\mathbf{c}\|^2+2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}$ |
| `F_rescale (float)`| $-2\frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}$|
| `F_error (float)`| $\ 2 \|\mathbf{o}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}$|
| `G_add (float)`| $\|\mathbf{q}_r-\mathbf{c}\|^2$ |
| `G_kBxSumq (float)`| $c_BS_q$|
| `G_error (float)`| $\| \mathbf{q}_r-\mathbf{c}\|$|

### Estimator of inner product

When inner product is used as the metric of vector search, it targets the data
vector with the *maximum* inner product with the query. The implementation
uses $1-\langle\mathbf{o}_r,\mathbf{q}_r\rangle$ so that smaller distances rank
higher-inner-product vectors first. The constant 1 does not affect ordering.

$$
\begin{align}
&1-\left< \mathbf{o}_r,\mathbf{q}_r\right>
\\=&1 -\left< \mathbf{o}_r-\mathbf{c} + \mathbf{c},\mathbf{q}_r-\mathbf{c} + \mathbf{c}\right>
\\=&1 -\left< \mathbf{q}_r,\mathbf{c}\right> -\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right> -  \left< \mathbf{o}_r-\mathbf{c},\mathbf{q}_r-\mathbf{c} \right>
\\ \approx &1 -\left< \mathbf{q}_r,\mathbf{c}\right> -\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right> +\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\end{align}
$$

The probabilistic error interval is

$$
\begin{align}
\pm \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}
\end{align}
$$

Store the following factors (`F_` for data and `G_` for the query).

| Name (Type) of Variable | Description |
| ----------------- | ----------- |
| `F_add (float)`| $1-\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right>+\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}$ |
| `F_rescale (float)`| $-\frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}$|
| `F_error (float)`| $\|\mathbf{o}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}$|
| `G_add (float)`| $-\left< \mathbf{q}_r,\mathbf{c}\right>$ |
| `G_kBxSumq (float)`| $c_BS_q$|
| `G_k1xSumq (float)`| $c_1S_q$|
| `G_error (float)`| $\| \mathbf{q}_r-\mathbf{c}\|$|

## Implementation

### Distance estimation

Both metrics use the same scoring formula with metric-specific factors.
Here `ip` is the inner product between the logical
unsigned $B$-bit code and the rotated query vector; it is a binary code only
when $B=1$.

```cpp

// Compute the estimated distance
// Note that G_add is dependent on the centroid.
float est_dist = F_add + G_add + F_rescale * (ip + G_kBxSumq);

// Compute the error bound 
// Note that G_error is dependent on the centroid.
float error_bound = F_error * G_error;

// Compute the lower and upper bounds of the estimated distance
float lb_dist = est_dist - error_bound;
float ub_dist = est_dist + error_bound;
```

### Incremental distance estimation

Split codes separate the sign bit from the remaining $B-1$ extra bits.
Score the sign code first, then refine selected candidates with the extra bits.
Each part has its own factors. The binary layout stores `F_add`,
`F_rescale`, and `F_error`; the extended layout stores `F_add_ex` and
`F_rescale_ex`. For $B>1$, the logical codes satisfy
$\mathbf{x}_u=2^{B-1}\mathbf{x}_{\mathrm{bin}}+\mathbf{x}_{\mathrm{ex}}$,
which gives the refined distance formula below.

The split-code implementation uses the binary error factor divided by
$2^{B-1}$ as its refined pruning margin. This is an approximation, not an
identity for the full-code error factor in the tables above: that factor depends
on the full-code direction's own $\langle\mathbf{\bar o},\mathbf{o}\rangle$.
The split layout does not store a separate `F_error_ex`.
The query factors are `G_add`, `G_error`, `G_kBxSumq` and `G_k1xSumq`.
Here `ip_bin` and `ip_ex` are the inner products of the sign and extra-bit
codes, respectively, with the rotated query.
These equations use exact code/query inner products. Binary filtering may instead
approximate `ip_bin` through four-bit query codes or FastScan lookup tables;
see [inner-product kernels](../kernel_ip.md). The margins do not separately bound
that additional error.

```cpp
// 1-bit dist
float est_dist = F_add + G_add + F_rescale * (ip_bin + G_k1xSumq);
float bound = F_error * G_error;
float ub_dist = est_dist + bound;
float lb_dist = est_dist - bound;

// boost to full-bit dist
float ex_scale = static_cast<float>(1 << (bits - 1));
float ex_est_dist =
    F_add_ex + G_add +
    F_rescale_ex * (ex_scale * ip_bin + ip_ex + G_kBxSumq);
float ex_bound = F_error * G_error / ex_scale;
float ex_ub_dist = ex_est_dist + ex_bound;
float ex_lb_dist = ex_est_dist - ex_bound;
```
