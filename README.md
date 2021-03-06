# tensor-RPCA
An implementation of robust principal component analysis for tensors.


This code decomposes multi-dimensional datasets into the sum of a low-rank tensor and a sparse tensor, performing principal component analysis that is robust to sparsely distributed gross errors. See the [paper for details](https://arxiv.org/abs/1901.10991).

![tensor-rpca-recovery](https://user-images.githubusercontent.com/10841845/75897128-f5e32780-5e2f-11ea-9257-c5d3e5d6bcdf.png)


Citation
---------
If you use this code, please consider citing this work using the following bibtex entry (for tensor-RPCA):

```
@misc{tensor-rpca2019,
    author       = "Driggs, D. and Becker, S. and Boyd-Graber, J.",
    title        = "Tensor Robust Principal Component Analysis:  Better recovery with atomic norm regularization",
    year         = "2019",
    Eprint       = {arXiv:1901.10991}
}
```

Code and installation
----

The code performing tensor RPCA runs on MATLAB, and the topic modeling demo uses Python and C++. The code requires tensor_toolbox to be installed and added to the path, and the topic modeling demo requires additional dependencies. Running `setup_fastRPCA` sets the correct paths and prompts the user before downloading and compiling all dependencies.
