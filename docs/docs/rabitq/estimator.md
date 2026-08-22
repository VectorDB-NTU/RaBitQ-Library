
# Estimator
This part presents how to use the quantized vectors to estimate Euclidean distances and inner product with minimal efforts. 

| Notation     | Description                            |
|----------|----------------------------------------|
| $\mathbf{o}_r$, $\mathbf{q}_r$ | The raw data vector and the raw query vector. |
| $\mathbf{c}$ | The center vector. |
| $D$ | The dimension of vectors. |
| $B$ | The number of bits per dimension used for the quantization. |
| $c_B$| A constant value $c_B=- \frac{2^B-1}{2}$. |
| $\mathbf{1}_D$ | The all-one vector of dimension $D$. |
| $P$ | The random rotation matrix. |
| $\mathbf{\bar o}$ | The quantized normalized data vector. |
| $\mathbf{q}_r'$ | The reversely rotated raw query vector: $\mathbf{q}_r'=P^{-1}\mathbf{q}_r$. |
| $S_q$ | $S_q=\sum_{i=1}^{D} \mathbf{q}'_{r}[i].$ |
| $\Delta_x$ | The rescaling factor of a quantized vector. |
| $\mathbf{x}_u$ | The uint representation of the quantized vectors.|


## Quantization 
Recall that the quantization algorithm of RaBitQ receives a raw data vector $\mathbf{o}_r$, a center vector $\mathbf{c}$ and a sample of random rotation matrices $P$ as inputs and returns a value $\Delta_x$ and an uint vector $\mathbf{x}_u$ as output, such that 

$$
\begin{align}
\mathbf{o}_r - \mathbf{c} \approx \| \mathbf{o}_r-\mathbf{c} \| \cdot \mathbf{\bar o} = \Delta_x P(\mathbf{x}_u+c_B\mathbf{1}_D) 
\end{align}
$$

where $c_B=- \frac{2^B-1}{2}$ and $B$ is the number of bits used for the quantization. $\mathbf{1}_D$ is the all-one vector of dimension $D$. $P$ is a sample of random rotation matrices. 

## Estimator
The following derivation covers squared Euclidean distance and inner product.
The public metric choices are `METRIC_L2` and `METRIC_IP`; there is no separate
cosine metric. For cosine search, normalize both data and query vectors, then
use inner product. The implementation represents inner-product distance as
$1-\langle\mathbf{o},\mathbf{q}\rangle$, which preserves the same ranking while
keeping the nearest-neighbor convention that smaller values are better.

### Estimator of Euclidean Distance
$$
\begin{align}
&\| \mathbf{o}_r-\mathbf{q}_r\|^2 
\\=& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2 \left< \mathbf{o}_r - \mathbf{c}, \mathbf{q}_r - \mathbf{c}\right>
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \left<\mathbf{ o},\mathbf{q} \right>
\\ \approx& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \frac{\left<\mathbf{\bar o},\mathbf{q} \right>}{ \left<\mathbf{\bar o},\mathbf{o} \right>}
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 - 2\| \mathbf{o}_r-\mathbf{c}\| \cdot \frac{\left<\mathbf{\bar o},\mathbf{q}_r-\mathbf{c} \right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}
\\ =& \|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 +2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{2}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left< \| \mathbf{o}_r-\mathbf{c}\|\cdot P^{-1}\mathbf{\bar o}, \mathbf{q}_r' \right>
\\=&\|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 +2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{2\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\\ \| \mathbf{o}_r-\mathbf{q}_r\|^2 \approx&\|\mathbf{o}_r -\mathbf{c}\|^2 + \|\mathbf{q}_r -\mathbf{c}\|^2 +2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{2\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\end{align}
$$

The error bound is given by 

$$
\begin{align}
% \pm \ 2 \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \frac{c_{error}}{\sqrt{D}\cdot 2^{B-1}}\\
\pm \ 2 \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}
\end{align}
$$

Here $\epsilon_0$ is a tunable parameter which controls the confidence level of the error bound. By default, $\epsilon_0=1.9$ guarantees nearly perfect confidence. 

We store the following variables such that the estimator can be computed easily.

| Name (Type) of Variable | Description |
| ----------------- | ----------- |
| `F_add (float)`| $\|\mathbf{o}_r-\mathbf{c}\|^2+2\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}$ |
| `F_rescale (float)`| $-2\frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}$|
| `F_error (float)`| $\ 2 \|\mathbf{o}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}$|
| `G_add (float)`| $\|\mathbf{q}_r-\mathbf{c}\|^2$ |
| `G_kBxSumq (float)`| $c_BS_q$|
| `G_error (float)`| $\| \mathbf{q}_r-\mathbf{c}\|$|

<!-- | `F_error (float)`| $2\|\mathbf{o}_r-\mathbf{c}\| \cdot \frac{c_{error}}{\sqrt{D}\cdot 2^{B-1}}$| -->


### Estimator of Inner Product
When inner product is used as the metric of vector search, it targets the data
vector with the *maximum* inner product with the query. The implementation
uses $1-\langle\mathbf{o},\mathbf{q}\rangle$ so that smaller distances rank
higher-inner-product vectors first. The constant 1 does not affect ordering.

$$
\begin{align}
&1-\left< \mathbf{o}_r,\mathbf{q}_r\right>
\\=&1 -\left< \mathbf{o}_r-\mathbf{c} + \mathbf{c},\mathbf{q}_r-\mathbf{c} + \mathbf{c}\right>
\\=&1 -\left< \mathbf{q}_r,\mathbf{c}\right> -\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right> -  \left< \mathbf{o}_r-\mathbf{c},\mathbf{q}_r-\mathbf{c} \right>
\\ \approx &1 -\left< \mathbf{q}_r,\mathbf{c}\right> -\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right> +\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\\ 1-\left< \mathbf{o}_r,\mathbf{q}_r\right>\approx &1 -\left< \mathbf{q}_r,\mathbf{c}\right> -\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right> +\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>} - \frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}\cdot \left[ \left< \mathbf{x}_u, \mathbf{q}_r' \right>+c_B S_q\right]
\end{align}
$$

The error bound is given by 

$$
\begin{align}
% \pm  \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \frac{c_{error}}{\sqrt{D}\cdot 2^{B-1}}
\pm \|\mathbf{o}_r-\mathbf{c}\|\cdot \| \mathbf{q}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}
\end{align}
$$

<!-- Here $c_{error}$ is a tunable parameter which controls the failure probability of the error bound.  -->

Here $\epsilon_0$ is a tunable parameter which controls the confidence level of the error bound. By default, $\epsilon_0=1.9$ guarantees nearly perfect confidence. 

We store the following variables such that the estimator can be computed easily. The variables `F_` indicates the factors for the data vector. The variables `G_` indicates the factors for the query vector.

| Name (Type) of Variable | Description |
| ----------------- | ----------- |
| `F_add (float)`| $1-\left< \mathbf{o}_r-\mathbf{c},\mathbf{c}\right>+\| \mathbf{o}_r-\mathbf{c}\| \frac{\left< \mathbf{\bar o}, \mathbf{c}\right>}{\left<\mathbf{\bar o},\mathbf{o} \right>}$ |
| `F_rescale (float)`| $-\frac{\Delta_x}{\left<\mathbf{\bar o},\mathbf{o} \right>}$|
| `F_error (float)`| $\|\mathbf{o}_r-\mathbf{c}\| \cdot \sqrt{\frac{1 - \left< \mathbf{\bar o},\mathbf{o}\right>^2}{\left< \mathbf{\bar o},\mathbf{o}\right>^2}} \frac{\epsilon_{0}}{\sqrt{D-1}}$|
| `G_add (float)`| $-\left< \mathbf{q}_r,\mathbf{c}\right>$ |
| `G_kBxSumq (float)`| $c_BS_q$|
| `G_k1xSumq (float)`| $c_1S_q$|
| `G_error (float)`| $\| \mathbf{q}_r-\mathbf{c}\|$|
<!-- | `F_error (float)`| $\|\mathbf{o}_r-\mathbf{c}\| \cdot \frac{c_{error}}{\sqrt{D}\cdot 2^{B-1}}$| -->

## Implementation
### Distance Estimation
Based on the above derivation, we can implement the estimator of Euclidean distance and inner product using exactly the same implementation. A pseudo code is given below. Let `ip` be the inner product between the binary code and the randomly rotated query vector.

```cpp

// Compute the estimated distance
// Note that G_add is dependent on the center vector.
float est_dist = F_add + G_add + F_rescale * (ip + G_kBxSumq)

// Compute the error bound 
// Note that G_error is dependent on the center vector.
float error_bound = F_error * G_error

// Compute the lower and upper bounds of the estimated distance
float lb_dist = est_dist - error_bound
float ub_dist = est_dist + error_bound
```


### Incremental Distance Estimation

RaBitQ supports incremental distance estimation. We split the code into two parts: the binary code (the most significant bits) and the extended code (the remaining $B-1$ bits). 
Incremental distance estimation supports to first estimate a coarse distance based on the binary code. If the accuracy is insufficient, we then access the extended code to boost the accuracy. 
For this, we prepare factors for both parts. The binary layout stores `F_add`,
`F_rescale`, and `F_error`; the extended layout stores `F_add_ex` and
`F_rescale_ex`. The implementation obtains the refined error bound by scaling
the binary error factor by $2^{B-1}$ rather than storing a separate
`F_error_ex`.
The factors for the query includes `G_add`, `G_error`, `G_kBxSumq` and `G_k1xSumq`. 
Let `ip_bin` be the inner product between the binary code and the randomly rotated query vector and `ip_ex` be the inner product between the ex-code and the randomly rotated query vector. 


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
