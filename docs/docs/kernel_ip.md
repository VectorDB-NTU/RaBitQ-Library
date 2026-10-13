# Inner-product kernels

> Inner-product kernels are selected at runtime. The library provides AVX2/FMA
> and AVX-512 implementations on x86-64, plus NEON and portable scalar
> implementations on ARM64. AVX-512 VPOPCNTDQ is used by popcount-specific
> x86 paths when available.

These kernels compute inner products of sign or multi-bit codes with a rotated
query. The notation follows the [Estimator](rabitq/estimator.md).

## The kernel for binary codes

### Single code

For a sign code $\mathbf{x}_0$, first quantize the rotated query
$\mathbf{q}_r'$ into four-bit unsigned integers using the fast RaBitQ encoder.
Then use bitwise-and `&` and `popcnt` to estimate the inner product.

$$
\mathbf{q}_r'\approx \Delta_q \mathbf{q}_u + v_q\cdot \mathbf{1}_D
$$

Let $\mathbf{q}_u^{(i)}$ be bit plane $i$ of $\mathbf{q}_u$. Then:

$$
\begin{align}
\left< \mathbf{x}_0,\mathbf{q}_r'\right>
&\approx \left< \mathbf{x}_0,\Delta_q \mathbf{q}_u + v_q\cdot \mathbf{1}_D\right>
\\&=\Delta_q \left< \mathbf{x}_0,\mathbf{q}_u\right> + v_q \cdot \mathrm{popcnt}(\mathbf{x}_0)
\\ &=\Delta_q \sum_{i=0}^{B_q-1} \left( 2^i\left< \mathbf{x}_0,\mathbf{q}_u^{(i)}\right> \right) + v_q \cdot\mathrm{popcnt}(\mathbf{x}_0)
\\ &=\Delta_q \sum_{i=0}^{B_q-1} 2^i\cdot \mathrm{popcnt}(\mathbf{x}_0\ \&\  \mathbf{q}_u^{(i)})  + v_q \cdot \mathrm{popcnt}(\mathbf{x}_0)
\end{align}
$$

### Batch code

FastScan scores batches of sign codes using query lookup tables. See the
[Faiss tutorial](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan))
for background.

For a $D$-bit binary code, we split it into $M=D/4$ segments. We prepare look-up-tables for each segment.
- $\mathrm{LUT}[m][mask]$: the inner product with $\mathbf{q}_r'$ in the $m$-th segment, i.e., the $(4m)$-th to the $(4m+3)$-th dimensions, when the code in the $m$-th segment equals to $mask$.

Based on the look-up-tables, we can compute the inner product as follows:

$$
\begin{align}
\left< \mathbf{x}_0,\mathbf{q}_r'\right>= \sum_{m=0}^{M-1}  \mathrm{LUT}[m][\mathbf{x}_{0}[4m:4m+3]]
\end{align}
$$

## The kernel for multi-bit codes

For multi-bit codes, the packed unsigned values are unpacked and accumulated
by the selected AVX2/FMA, AVX-512, NEON, or scalar kernel. The dispatch table supports zero
through eight extended bits; full RaBitQ codes therefore contain one through
nine bits per dimension.
