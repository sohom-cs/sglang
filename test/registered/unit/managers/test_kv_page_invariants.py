"""Unit tests for SGLANG_CHECK_KV_PAGE_INVARIANTS: watermark + double-free checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_PAGE_SIZE = 256


def _make_checker(page_size=_PAGE_SIZE, row_width=4096, num_reqs=8, free_pages=None):
    rtt = torch.zeros((num_reqs, row_width), dtype=torch.int32)
    rtp = SimpleNamespace(req_to_token=rtt)
    if free_pages is None:
        free_pages = torch.arange(num_reqs * 4, dtype=torch.int64)
    alloc = SimpleNamespace(
        page_size=page_size,
        free_pages=free_pages,
        get_all_free_pages=lambda: free_pages,
    )
    records = {}
    tc = SimpleNamespace(session_records=lambda: records)
    _ps, _rtp, _alloc, _tc = page_size, rtp, alloc, tc

    class _FakeChecker:
        page_size = _ps
        req_to_token_pool = _rtp
        token_to_kv_pool_allocator = _alloc
        tree_cache = _tc
        get_last_batch = lambda self: None
        get_running_batch = lambda self: None
        get_chunked_req = lambda self: None
        disaggregation_mode = DisaggregationMode.NULL
        get_disagg_decode_reqs = lambda self: []
        count_memory_leak_warnings = 0

        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker as _RIC,
        )

        _check_kv_page_invariants = _RIC._check_kv_page_invariants
        _requests_owning_rows = _RIC._requests_owning_rows

    return _FakeChecker(), rtt, tc, alloc


class _FakeOwner:
    # A req or a session slot; the checker reads only `kv` (and `rid` for reqs).
    def __init__(self, rpi, committed, allocated, rid=None, protected=0):
        self.rid = rid
        self.kv = ReqKvInfo(
            req_pool_idx=rpi,
            cache_protected_len=protected,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
        )


class TestKVPageInvariants(CustomTestCase):
    def test_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeOwner(0, 145, 144, rid="a")]
        )
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_slot_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: None
        tc.session_records()["s1"] = _FakeOwner(0, 145, 144).kv
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_owner_references_free_page_raises(self):
        # req 0 owns page 5, but page 5 is in the free pool -> use-after-free.
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([5, 6, 7]))
        rtt[0, :3] = torch.tensor(
            [5 * _PAGE_SIZE, 5 * _PAGE_SIZE + 1, 5 * _PAGE_SIZE + 2]
        )
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeOwner(0, 3, 3, rid="a")]
        )
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_requests_outside_last_batch_are_checked(self):
        # A request still owns its row from running_batch or parked between chunks.
        for where in ("running", "chunked"):
            with self.subTest(where=where):
                chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([5, 6, 7]))
                rtt[0, :3] = torch.tensor(
                    [5 * _PAGE_SIZE, 5 * _PAGE_SIZE + 1, 5 * _PAGE_SIZE + 2]
                )
                owner = _FakeOwner(0, 3, 3, rid="a")
                if where == "running":
                    chk.get_running_batch = lambda: SimpleNamespace(reqs=[owner])
                else:
                    chk.get_chunked_req = lambda: owner
                with self.assertRaises(ValueError):
                    chk._check_kv_page_invariants()

    def test_classed_allocator_without_flat_free_list_is_checked(self):
        chk, rtt, _tc, alloc = _make_checker(page_size=4, row_width=8)
        alloc.free_pages = None
        alloc.get_all_free_pages = MagicMock(
            return_value=torch.tensor([5, 6], dtype=torch.int64)
        )
        rtt[0, :1] = torch.tensor([5 * alloc.page_size])
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeOwner(0, 1, 1, rid="classed")]
        )

        with patch(
            "sglang.srt.managers.scheduler_components.invariant_checker.page_interleave_shard_size",
            return_value=2,
        ):
            with self.assertRaisesRegex(ValueError, "use-after-free"):
                chk._check_kv_page_invariants()

        alloc.get_all_free_pages.assert_called()

    def test_free_pool_duplicate_raises(self):
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([3, 3, 4]))
        rtt[0, :1] = torch.tensor([10 * _PAGE_SIZE])  # owner page 10, not in free
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeOwner(0, 1, 1, rid="a")]
        )
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_page_in_two_owners_own_ranges_raises(self):
        """A page handed to a second live request while its owner still holds it
        is reported, also when the owner is a PD decode request outside the batches."""
        for second in ("running_batch", "disagg_decode"):
            with self.subTest(second=second):
                chk, rtt, tc, alloc = _make_checker(
                    free_pages=torch.arange(100, 200, dtype=torch.int64)
                )
                rtt[0, :_PAGE_SIZE] = torch.arange(_PAGE_SIZE)
                rtt[1, :_PAGE_SIZE] = torch.arange(_PAGE_SIZE)  # page 0 again
                a = _FakeOwner(0, _PAGE_SIZE, _PAGE_SIZE, rid="a")
                b = _FakeOwner(1, _PAGE_SIZE, _PAGE_SIZE, rid="b")
                chk.get_last_batch = lambda: SimpleNamespace(reqs=[a])
                if second == "running_batch":
                    chk.get_running_batch = lambda: SimpleNamespace(reqs=[b])
                else:
                    chk.disaggregation_mode = DisaggregationMode.DECODE
                    chk.get_disagg_decode_reqs = lambda: [b]
                with self.assertRaisesRegex(
                    ValueError, "page 0 is in the own ranges of req a and req b"
                ):
                    chk._check_kv_page_invariants()

    def test_shared_prefix_and_session_record_are_single_owners(self):
        """Pages below cache_protected_len are the tree's and may be shared; a
        request running on a session's record is the same owner as the slot."""
        chk, rtt, tc, alloc = _make_checker(
            free_pages=torch.arange(100, 200, dtype=torch.int64)
        )
        rtt[0, : 2 * _PAGE_SIZE] = torch.arange(2 * _PAGE_SIZE)  # pages 0, 1
        rtt[1, :_PAGE_SIZE] = torch.arange(_PAGE_SIZE)  # shared prefix page 0
        rtt[1, _PAGE_SIZE : 2 * _PAGE_SIZE] = torch.arange(
            2 * _PAGE_SIZE, 3 * _PAGE_SIZE
        )  # page 2
        a = _FakeOwner(0, 2 * _PAGE_SIZE, 2 * _PAGE_SIZE, rid="a", protected=_PAGE_SIZE)
        b = _FakeOwner(1, 2 * _PAGE_SIZE, 2 * _PAGE_SIZE, rid="b", protected=_PAGE_SIZE)
        tc.session_records()["session-b"] = b.kv
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[a, b])
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)


if __name__ == "__main__":
    unittest.main()
