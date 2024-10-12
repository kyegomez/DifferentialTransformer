
# Differential Transformer 
An open source community implementation of the model from "DIFFERENTIAL TRANSFORMER" paper by Microsoft. [Paper Link](https://arxiv.org/abs/2410.05258)



[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## Install

```bash
$ pip3 install differential-transformers
```

## Usage Transformer

```python

import torch
from differential_transformer.main import DifferentialTransformer
from loguru import logger

# Example usage:
# Example dimensions
batch_size = 32
seq_len = 128
embedding_dim = 64
h = 8
λ = 0.1
λinit = 0.05

# Create random input tensor
x = torch.randint(0, 256, (1, 1024))

# Instantiate and run the multi-head attention
multi_head = DifferentialTransformer(heads=h, dim=embedding_dim, λinit=λinit)
output = multi_head(x, λ=λ)

logger.info(f"Output shape: {output.shape}")


```

# License
MIT


## Citation


```bibtex
@misc{ye2024differentialtransformer,
    title={Differential Transformer}, 
    author={Tianzhu Ye and Li Dong and Yuqing Xia and Yutao Sun and Yi Zhu and Gao Huang and Furu Wei},
    year={2024},
    eprint={2410.05258},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.05258}, 
}

```