import os
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    DecodeRequest,
    DecodeTransferQueue,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import NextBatchPlan, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import invariant_checker
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
    kv_private_swa_tokens,
)
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.session.streaming_session import SessionSlot

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCheckTreeCacheGate(CustomTestCase):
    @contextmanager
    def _without_explicit_sanity_check_setting(self):
        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.clear()
            yield

    def _make_checker(self, prefix_sharing=True, hybrid_swa=True, hybrid_ssm=False):
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = prefix_sharing
        tree_cache.supports_swa.return_value = hybrid_swa
        tree_cache.supports_mamba.return_value = hybrid_ssm
        return SchedulerInvariantChecker(
            is_hybrid_swa=hybrid_swa,
            is_hybrid_ssm=hybrid_ssm,
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=1,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=1024,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=MagicMock(),
            req_to_token_pool=MagicMock(),
            pool_stats_observer=MagicMock(),
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
            scheduler_stage_metrics=None,
        )

    def test_disabled_by_default(self):
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_not_called()

    def test_enabled_by_default_in_ci(self):
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            self._without_explicit_sanity_check_setting(),
        ):
            checker = self._make_checker()

            checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_called_once()

    def test_explicitly_disabled_in_ci(self):
        with envs.SGLANG_IS_IN_CI.override(True):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(False):
                checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_not_called()

    def test_runs_when_enabled(self):
        with envs.SGLANG_IS_IN_CI.override(False):
            checker = self._make_checker()

            with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
                checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_called_once()

    def test_skipped_without_prefix_sharing(self):
        with envs.SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK.override(True):
            checker = self._make_checker(
                prefix_sharing=False, hybrid_swa=False, hybrid_ssm=True
            )

            checker._check_tree_cache()

            checker.tree_cache.sanity_check.assert_not_called()


class TestPrivateSwaTokens(CustomTestCase):
    def test_floor_shielded_prefix_stays_private(self):
        # A prefill-aware SWA request with radix off: the window cursor jumps to
        # the floor (278) without freeing it, then frees [278, 300).
        kv = ReqKvInfo(
            req_pool_idx=0,
            kv_allocated_len=280,
            swa_evict_floor=278,
            component_evicted_seqlens={ComponentType.SWA: 278},
        )
        self.assertEqual(kv_private_swa_tokens(kv, page_size=1), 280)
        kv.kv_allocated_len = 450
        kv.set_evicted_seqlen(ComponentType.SWA, 300)
        self.assertEqual(kv_private_swa_tokens(kv, page_size=1), 428)


class TestShardedFullPoolInvariant(CustomTestCase):
    PAGE_SIZE = 16
    SHARD_SIZE = 4

    def setUp(self):
        super().setUp()
        parallel = patch.object(
            invariant_checker,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def _make_checker(self):
        self.allocator = PageInterleavePoolAllocator(
            size=4 * self.PAGE_SIZE,
            physical_page_size=self.PAGE_SIZE,
            shard_size=self.SHARD_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        req_pool = ReqToTokenPool(
            size=4,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=self.PAGE_SIZE,
                    tree_components=(ComponentType.FULL,),
                )
            )
        self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
        self.running_batch = self.last_batch
        self.chunked_req = None
        observer = SchedulerPoolStatsObserver(
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=req_pool,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.allocator.size,
        )
        return SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            disaggregation_mode=DisaggregationMode.PREFILL,
            page_size=self.PAGE_SIZE,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.allocator.size,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=req_pool,
            pool_stats_observer=observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
            get_chunked_req=lambda: self.chunked_req,
            scheduler_stage_metrics=None,
        )

    def _allocate(self, tokens, *, prefix=0, last_loc=-1, rotation_base=None):
        prefix_lens = torch.tensor([prefix], dtype=torch.int64)
        seq_lens = torch.tensor([prefix + tokens], dtype=torch.int64)
        bases = [rotation_base]
        locs = self.allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            last_loc=torch.tensor([last_loc], dtype=torch.int64),
            extend_num_tokens=tokens,
            rotation_bases=bases,
        )
        self.assertIsNotNone(locs)
        return locs, bases[0]

    def _cache_prefix(self, tokens, *, protected=False):
        locs, base = self._allocate(tokens)
        key = RadixKey(array("q", range(tokens)))
        self.cache.insert(InsertParams(key=key, value=locs, rotation_base=base))
        if protected:
            matched = self.cache.match_prefix(MatchPrefixParams(key=key))
            self.cache.inc_lock_ref(matched.last_device_node)
        return locs, base

    def _active_partial_page(self, checker):
        prefix_locs, base = self._cache_prefix(self.PAGE_SIZE, protected=True)
        tail_locs, _ = self._allocate(
            3,
            prefix=self.PAGE_SIZE,
            last_loc=prefix_locs[-1].item(),
            rotation_base=base,
        )
        locs = torch.cat((prefix_locs, tail_locs))
        checker.req_to_token_pool.req_to_token[0, : len(locs)] = locs
        req = SimpleNamespace(
            kv=SimpleNamespace(
                holds_kv=True,
                req_pool_idx=0,
                kv_allocated_len=len(locs),
                cache_protected_len=self.PAGE_SIZE,
            ),
            beam_group=None,
        )
        self.last_batch = SimpleNamespace(reqs=[req], is_empty=lambda: False)
        self.running_batch = self.last_batch
        return req

    def test_balanced_and_skewed_cache_ownership_conserve_full_pool(self):
        for cached_pages in (1, self.SHARD_SIZE):
            for protected in (False, True):
                with self.subTest(cached_pages=cached_pages, protected=protected):
                    checker = self._make_checker()
                    cached_tokens = cached_pages * self.PAGE_SIZE
                    self._cache_prefix(cached_tokens, protected=protected)
                    self.assertEqual(
                        self.cache.protected_size() + self.cache.evictable_size(),
                        cached_tokens,
                    )
                    stats = checker.pool_stats_observer.get_pool_stats()
                    if cached_pages == 1:
                        self.assertLess(
                            stats.full_available_size,
                            self.allocator.aggregate_free_size(),
                        )
                    else:
                        self.assertEqual(
                            stats.full_available_size,
                            self.allocator.aggregate_free_size(),
                        )
                    leak, msg = checker._check_full_pool(stats)
                    self.assertFalse(leak, msg)
                    self.assertIn("class_free_pages=", msg)

    def test_idle_check_detects_one_unowned_page(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        self._allocate(self.PAGE_SIZE)  # Deliberately no request or cache owner.
        leak, messages = checker._check_all_pools(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, messages)

    def test_busy_check_detects_one_unowned_page(self):
        checker = self._make_checker()
        self._active_partial_page(checker)
        self._allocate(self.PAGE_SIZE)  # Leak in addition to the active tail.
        self.assertEqual(checker._get_total_uncached_sizes(), (self.PAGE_SIZE, 0))
        with self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak Detected"):
            checker.self_check_during_busy()

    def test_partial_active_page_is_accounted_without_slack_exemption(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # Count the same request only once, including when it is parked
        # between prefill chunks instead of appearing in either batch.
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.chunked_req = req
                if parked:
                    self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
                    self.running_batch = self.last_batch
                uncached, swa_uncached = checker._get_total_uncached_sizes()
                self.assertEqual((uncached, swa_uncached), (self.PAGE_SIZE, 0))
                stats = checker.pool_stats_observer.get_pool_stats()
                leak, msg = checker._check_full_pool(stats, uncached=uncached)
                self.assertFalse(leak, msg)
                self.assertNotIn("slack_allowed", msg)
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
                    checker.self_check_during_busy()

    def test_session_record_is_counted_once_by_its_owner(self):
        checker = self._make_checker()
        req = self._active_partial_page(checker)
        # The session owns the row: session-held whether the turn is in a batch
        # or parked between prefill chunks, never also uncached.
        self.cache.session.slots["s"] = SessionSlot(kv=req.kv)
        for parked in (False, True):
            with self.subTest(parked=parked):
                self.chunked_req = req
                if parked:
                    self.last_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
                    self.running_batch = self.last_batch
                self.assertEqual(checker._get_total_uncached_sizes(), (0, 0))
                self.assertEqual(
                    checker.pool_stats_observer.session_held_tokens(), self.PAGE_SIZE
                )
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
                    checker.self_check_during_busy()

    def test_misaligned_evictable_counter_is_not_rounded_up(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        self.cache.tree_core.component_evictable_size_[ComponentType.FULL] -= 1
        leak, msg = checker._check_full_pool(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, msg)
        self.assertIn(f"evictable={self.PAGE_SIZE - 1}", msg)

    def test_compensating_misaligned_counters_still_report_alignment_error(self):
        checker = self._make_checker()
        self._cache_prefix(self.PAGE_SIZE)
        core = self.cache.tree_core
        core.component_evictable_size_[ComponentType.FULL] -= 1
        core.component_protected_size_[ComponentType.FULL] += 1
        # Their sum still matches the allocated page; neither individual
        # counter can legitimately contain a fraction of a physical page.
        self.assertEqual(
            self.cache.evictable_size() + self.cache.protected_size(), self.PAGE_SIZE
        )
        leak, msg = checker._check_full_pool(
            checker.pool_stats_observer.get_pool_stats()
        )
        self.assertTrue(leak, msg)
        self.assertIn("align", msg.lower())


class _HeldReq:
    # Hashable like Req: the decode offload manager keys in-flight copies by request.
    def __init__(self, rid):
        self.rid = rid
        self.kv = ReqKvInfo()
        self.beam_group = None


class TestPdDecodeKvOwners(CustomTestCase):
    """A PD decode worker holds KV for requests outside its batches; the checker
    the scheduler builds must count them and see a second owner on their pages."""

    PAGE_SIZE = 4
    NUM_TOKENS = 2 * PAGE_SIZE
    DECODE_LOOPS = (
        Scheduler.event_loop_normal_disagg_decode,
        Scheduler.event_loop_overlap_disagg_decode,
    )

    def setUp(self):
        super().setUp()
        parallel = patch.object(
            invariant_checker,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def _make_scheduler(self):
        self.allocator = PagedTokenToKVPoolAllocator(
            size=8 * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.req_pool = DecodeReqToTokenPool(
            size=2,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
            pre_alloc_size=2,
        )
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = []
        transfer_queue._deferred_releases = []
        transfer_queue.deferred_kv_release_timeout = 30.0
        batch = SimpleNamespace(reqs=[], is_empty=lambda: not batch.reqs)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.__dict__.update(
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            disaggregation_mode=DisaggregationMode.DECODE,
            page_size=self.PAGE_SIZE,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.allocator.size,
            tree_cache=UnifiedRadixCache(
                CacheInitParams(
                    disable=True,
                    req_to_token_pool=self.req_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=self.PAGE_SIZE,
                    tree_components=(ComponentType.FULL,),
                )
            ),
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            session_controller=None,
            hisparse_coordinator=None,
            last_batch=batch,
            running_batch=batch,
            chunked_req=None,
            waiting_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
            ),
            disagg_decode_transfer_queue=transfer_queue,
            decode_offload_manager=None,
            scheduler_stage_metrics=None,
        )
        scheduler.init_pool_stats_observer()
        scheduler.init_invariant_checker()
        return scheduler

    def _req(self, rid):
        req = _HeldReq(rid)
        self.req_pool.alloc([req])
        locs = self.allocator.alloc(self.NUM_TOKENS)
        self.req_pool.write((req.kv.req_pool_idx, slice(0, self.NUM_TOKENS)), locs)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = self.NUM_TOKENS
        return req

    @staticmethod
    def _hold(scheduler, req, where):
        if where == "waiting_queue":
            scheduler.waiting_queue.append(req)
        elif where == "transfer_queue":
            scheduler.disagg_decode_transfer_queue.queue.append(
                DecodeRequest(req=req, kv_receiver=None)
            )
        elif where == "deferred_release":
            scheduler.disagg_decode_transfer_queue._defer_release(
                DecodeRequest(
                    req=req, kv_receiver=SimpleNamespace(bootstrap_infos=[{}])
                )
            )
        else:
            scheduler.decode_offload_manager = SimpleNamespace(
                offload_inflight={req: 1}
            )

    def test_requests_outside_batches_own_their_rows(self):
        for where in ("waiting_queue", "transfer_queue", "deferred_release", "offload"):
            with self.subTest(where=where):
                scheduler = self._make_scheduler()
                scheduler.running_batch.reqs = [self._req("running")]
                self._hold(scheduler, self._req("held"), where)
                with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True):
                    scheduler.invariant_checker.self_check_during_busy()

    def test_pages_freed_under_a_held_request_are_reported(self):
        """A flush that resets the pools while a decode request still holds KV
        (deferred release, or waiting for admission) hands its pages to the next
        request; the checks must report that, not balance it."""
        for where in ("deferred_release", "waiting_queue"):
            with self.subTest(where=where):
                scheduler = self._make_scheduler()
                self._hold(scheduler, self._req("held"), where)
                self.req_pool.clear()
                self.allocator.clear()
                scheduler.running_batch.reqs = [self._req("newcomer")]
                checker = scheduler.invariant_checker
                with self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak"):
                    checker.self_check_during_busy()
                with self.assertRaisesRegex(
                    ValueError,
                    "page 1 is in the own ranges of req newcomer and req held",
                ):
                    checker._check_kv_page_invariants()

    @staticmethod
    def _run_decode_step(scheduler, event_loop):
        # One iteration of the real loop; intake, transfer polling and the forward
        # are stubbed, and admission moves the waiting queue into the batch.
        def next_plan(running_batch):
            reqs = running_batch.reqs + scheduler.waiting_queue
            scheduler.waiting_queue = []
            batch = SimpleNamespace(reqs=reqs, copy=lambda: batch)
            return NextBatchPlan(batch_to_run=batch, running_batch=batch)

        def run_batch(batch):
            scheduler.gracefully_exit = True

        scheduler.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries = (
            lambda: None
        )
        scheduler.__dict__.update(
            gracefully_exit=False,
            _engine_paused=False,
            ingest_requests=lambda: None,
            process_decode_queue=lambda: None,
            get_next_disagg_decode_batch_to_run=next_plan,
            ngram_embedding_manager=SimpleNamespace(
                prepare_for_forward=lambda batch, chunked_req: batch
            ),
            run_batch=run_batch,
            process_batch_result=lambda batch, result: None,
            is_disable_overlap_for_batch=lambda batch, last_batch: False,
            _apply_war_barrier=lambda: None,
            launch_batch_sample_if_needed=lambda result, batch: None,
        )
        event_loop(scheduler)

    def test_decode_step_admitting_a_request_passes_the_busy_check(self):
        """The decode loops check after the step's batch is set: a request moved
        from the waiting queue into the batch is counted once, beside a held one."""
        for event_loop in self.DECODE_LOOPS:
            with self.subTest(loop=event_loop.__name__):
                scheduler = self._make_scheduler()
                self._hold(scheduler, self._req("held"), "deferred_release")
                scheduler.waiting_queue.append(self._req("admitted"))
                with (
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2),
                    self.assertLogs(invariant_checker.logger, "INFO") as logs,
                ):
                    self._run_decode_step(scheduler, event_loop)
                self.assertIn("uncached=16", logs.output[0])

    def test_decode_step_on_pages_of_a_deferred_release_is_reported(self):
        """A flush while a deferred release holds KV frees its pages and the next
        request gets them; the decode step that runs that request reports it."""
        for event_loop in self.DECODE_LOOPS:
            with self.subTest(loop=event_loop.__name__):
                scheduler = self._make_scheduler()
                self._hold(scheduler, self._req("held"), "deferred_release")
                self.req_pool.clear()
                self.allocator.clear()
                scheduler.waiting_queue.append(self._req("newcomer"))
                with (
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
                    self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak"),
                ):
                    self._run_decode_step(scheduler, event_loop)


if __name__ == "__main__":
    unittest.main()
