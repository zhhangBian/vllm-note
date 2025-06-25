# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: Optional[dict[str, int]] = None


# load：从第一个connector中获取可以加载的token数量，然后根据token数量从其他connector中获取可以加载的token数量
# save：将token保存到所有connector中
class MultiConnector(KVConnectorBase_V1):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # 设置多个connector，每个connector负责不同的kv缓存
        self._connectors: list[KVConnectorBase_V1] = []
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors")
        assert ktcs is not None
        for ktc in ktcs:
            temp_config = copy.copy(vllm_config)
            temp_config.kv_transfer_config = KVTransferConfig(**ktc)
            self._connectors.append(
                KVConnectorFactory.create_connector_v1(temp_config, role))

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request. Not needed for async loads since we only allow
        # a single connector to load.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_saves: dict[str, int] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    # We must override the base class method here because we need to bind
    # the metadata to each connector in the order of the connectors in the
    # MultiKVConnectorMetadata.
    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(
                connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)

    def clear_connector_metadata(self) -> None:
        for c in self._connectors:
            c.clear_connector_metadata()

    # ==============================
    # Worker-side methods
    # ==============================
    # 开始加载KVC，调用所有connector的start_load_kv
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    # 等待layer加载完成，调用所有connector的wait_for_layer_load
    def wait_for_layer_load(self, layer_name: str) -> None:
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    # 保存KVC，调用所有connector的save_kv_layer
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    # 等待保存完成，调用所有connector的wait_for_save
    def wait_for_save(self):
        for c in self._connectors:
            c.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            # Aggregate finished recving request ids.
            finished_recving.update(recving or ())
            # Aggregate finished sending request ids - only include
            # once we've drained the "extra" count (for cases where
            # more than one connector is async-saving the same request).
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1

        return finished_sending or None, finished_recving or None

    # ==============================
    # Scheduler-side methods
    # ==============================
    # TODO：这里可以进行一个简单的修改，改为匹配最大的token数量
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        to_return = (0, False)
        # 匹配第一个有token的connector，然后根据token数量从其他connector中获取可以加载的token数量
        for i, c in enumerate(self._connectors):
            # 获取匹配的token数量
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens)
            # The first connector that has new matched tokens will be assigned
            # to this request.
            if to_return[0] == 0 and toks > 0:
                self._requests_to_connector[request.request_id] = i
                to_return = (toks, load_async)
        return to_return

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(
            request.request_id, -1)
        empty_blocks = blocks.new_empty()
        # 只对选定的 connector 传递实际的 blocks 和 token 数量
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks,
                                           num_external_tokens)
            # 对其他 connector 传递空 blocks 和 0 token 数量
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    # 构建元数据，用于在调度器和工作者之间传递信息
    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput) -> MultiKVConnectorMetadata:
        metadata = MultiKVConnectorMetadata(metadata=tuple(
            c.build_connector_meta(scheduler_output)
            for c in self._connectors))
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata

    # 通知 Worker 端请求已完成，返回是否异步保存/发送，以及可选的 KVTransferParams
    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        async_saves = 0
        kv_txfer_params = None
        # 对所有的connector进行遍历，如果某个connector需要异步保存，则将async_saves加1
        # 如果某个connector需要异步发送，则将kv_txfer_params设置为该connector的KVTransferParams
        for c in self._connectors:
            async_save, txfer_params = c.request_finished(request, blocks)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    #TODO we can probably change this to merge the dicts here,
                    # checking for key clashes.
                    raise RuntimeError(
                        "Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        # 如果需要异步保存的connector数量大于1，则将async_saves减1，并存储到self._extra_async_saves中
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        # Clean up other state for this request.
        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
