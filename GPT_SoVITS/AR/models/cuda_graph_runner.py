from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn

_N_WARMUP_ITERS = 5


class DecodingCUDAGraphRunner(nn.Module):
    """
    A CUDA graph runner for decoding with AR model.
    """

    def __init__(self, model: nn.Module, batch_size: int = 1, max_len: int = 1500):
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.hidden_dim = model.model_dim
        self.num_head = model.num_head
        self.head_dim = self.hidden_dim // self.num_head
        self.num_layers = model.num_layers
        dtype = model.bert_proj.weight.dtype
        device = model.bert_proj.weight.device

        self.x = torch.empty(
            (batch_size, 1, self.hidden_dim), dtype=dtype, device=device
        )

        self.out = torch.empty(
            (batch_size, 1, self.hidden_dim), dtype=dtype, device=device
        )

        self.k_caches = [
            torch.empty(
                (batch_size, max_len, self.num_head, self.head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.num_layers)
        ]

        self.v_caches = [
            torch.empty(
                (batch_size, max_len, self.num_head, self.head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.num_layers)
        ]

        self.kv_len = torch.tensor([0]).to(dtype=torch.int32).to(device)

        self._graph = None

    def decode_next_token(self, x: torch.Tensor, kv_len: torch.Tensor) -> torch.Tensor:
        """
        Decode the next token.
        """
        assert x.shape[0] == self.batch_size
        assert x.shape[1] == 1

        for i in range(self.num_layers):
            blk = self.model.h.layers[i]

            q, k, v = F.linear(
                x, blk.self_attn.in_proj_weight, blk.self_attn.in_proj_bias
            ).chunk(3, dim=-1)

            batch_size = self.batch_size

            self.k_caches[i][:, kv_len] = k[:, 0].view(
                batch_size, self.num_head, self.head_dim
            )
            self.v_caches[i][:, kv_len] = v[:, 0].view(
                batch_size, self.num_head, self.head_dim
            )

            q = q.view(batch_size, 1, self.num_head, -1)

            attn = flash_attn.flash_attn_with_kvcache(
                q, self.k_caches[i], self.v_caches[i], cache_seqlens=kv_len + 1
            )
            attn = attn.view(batch_size, 1, self.hidden_dim)
            attn = blk.self_attn.out_proj(attn)

            x = blk.norm1(x + attn)
            x = blk.norm2(x + blk.linear2(F.relu(blk.linear1(x))))
        return x

    def capture(self) -> None:
        """
        Capture the CUDA graph.
        """
        assert self._graph is None

        for _ in range(_N_WARMUP_ITERS):
            self.out = self.decode_next_token(self.x, self.kv_len)

        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.out = self.decode_next_token(self.x, self.kv_len)

        torch.cuda.synchronize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, replay the CUDA graph and **increment** the kv_len.
        """
        self.x.copy_(x)
        self._graph.replay()
        x_out = self.out.clone()
        self.kv_len += 1

        return x_out

    def assign_kvcache(
        self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]
    ):
        """
        Assign the key and value caches, and their lengths.
        """
        assert self._graph is not None

        kv_len = k_caches[0].shape[1]
        self.kv_len[0] = kv_len

        for i in range(self.num_layers):
            self.k_caches[i][:, :kv_len] = k_caches[i].view(
                self.batch_size, kv_len, self.num_head, self.head_dim
            )
            self.v_caches[i][:, :kv_len] = v_caches[i].view(
                self.batch_size, kv_len, self.num_head, self.head_dim
            )
