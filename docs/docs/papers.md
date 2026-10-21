# Papers and citations

Research behind RaBitQ-Library's quantization and graph-search implementations.
If you use this work, please cite the papers relevant to the method you use.

## Original RaBitQ

The one-bit method and its theoretical error bound.

> Jianyang Gao and Cheng Long. “RaBitQ: Quantizing High-Dimensional Vectors
> with a Theoretical Error Bound for Approximate Nearest Neighbor Search.”
> *Proceedings of the ACM on Management of Data* 2, 3, Article 167 (May 2024),
> 27 pages. [https://doi.org/10.1145/3654970](https://doi.org/10.1145/3654970).

## Multi-bit RaBitQ

Practical multi-bit quantization and asymptotically optimal error guarantees.

> Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond
> Chi-Wing Wong. “Practical and Asymptotically Optimal Quantization of
> High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
> Search.” *Proceedings of the ACM on Management of Data* 3, 3, Article 202
> (June 2025), 26 pages. [https://doi.org/10.1145/3725413](https://doi.org/10.1145/3725413).

## SymphonyQG

Integration of quantization and graph search. The paper uses raw vectors for
candidate scoring; the library's optional 4/8-bit storage and PiPNN initialization
are implementation extensions. See the [SymphonyQG guide](index/qg.md).

> Yutong Gou, Jianyang Gao, Yuexuan Xu, and Cheng Long. “SymphonyQG: Towards
> Symphonious Integration of Quantization and Graph for Approximate Nearest
> Neighbor Search.” *Proceedings of the ACM on Management of Data* 3, 1,
> Article 80 (February 2025), 26 pages.
> [https://doi.org/10.1145/3709730](https://doi.org/10.1145/3709730).
