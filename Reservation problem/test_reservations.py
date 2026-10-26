import os
import sqlite3
import tempfile
import threading
import unittest

from reservations import Request, RequestQueue, ReservationService


class CrashOnCommit:
    """Connection wrapper that dies right before COMMIT, like a process crash."""

    def __init__(self, con):
        self._con = con

    def execute(self, sql, *args):
        if sql.strip().upper() == "COMMIT":
            raise RuntimeError("crash")
        return self._con.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._con, name)


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "shop.db")

    def tearDown(self):
        self.dir.cleanup()

    def service(self, stock):
        return ReservationService(self.path, stock)

    def stock(self):
        con = sqlite3.connect(self.path)
        try:
            return con.execute("SELECT stock FROM inventory").fetchone()[0]
        finally:
            con.close()

    def reserved_total(self):
        con = sqlite3.connect(self.path)
        try:
            return con.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM reservations WHERE status = 'RESERVED'"
            ).fetchone()[0]
        finally:
            con.close()

    def run_concurrently(self, service, requests):
        results = {}
        barrier = threading.Barrier(len(requests))

        def worker(rid, qty):
            barrier.wait()
            results[rid] = service.reserve(rid, qty)

        threads = [threading.Thread(target=worker, args=r) for r in requests]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def test_single_unit_single_winner(self):
        s = self.service(1)
        results = self.run_concurrently(s, [(f"c{i}", 1) for i in range(20)])
        self.assertEqual(list(results.values()).count("RESERVED"), 1)
        self.assertEqual(self.stock(), 0)

    def test_n_units_exactly_n_winners(self):
        s = self.service(5)
        results = self.run_concurrently(s, [(f"c{i}", 1) for i in range(30)])
        self.assertEqual(list(results.values()).count("RESERVED"), 5)
        self.assertEqual(self.stock(), 0)

    def test_invariant_with_mixed_quantities(self):
        s = self.service(10)
        results = self.run_concurrently(s, [(f"c{i}", 1 + i % 4) for i in range(25)])
        self.assertGreaterEqual(self.stock(), 0)
        self.assertEqual(self.stock() + self.reserved_total(), 10)
        self.assertIn("RESERVED", results.values())

    def test_quantity_larger_than_stock_is_rejected(self):
        s = self.service(1)
        self.assertEqual(s.reserve("big", 2), "OUT_OF_STOCK")
        self.assertEqual(self.stock(), 1)
        self.assertEqual(s.reserve("small", 1), "RESERVED")

    def test_retry_of_reserved_does_not_take_more_stock(self):
        s = self.service(3)
        self.assertEqual(s.reserve("a", 1), "RESERVED")
        for _ in range(5):
            self.assertEqual(s.reserve("a", 1), "RESERVED")
        self.assertEqual(self.stock(), 2)

    def test_retry_of_out_of_stock_stays_out_of_stock(self):
        s = self.service(1)
        s.reserve("winner", 1)
        self.assertEqual(s.reserve("loser", 1), "OUT_OF_STOCK")
        self.assertEqual(s.reserve("loser", 1), "OUT_OF_STOCK")

    def test_concurrent_retries_of_same_request(self):
        s = self.service(1)
        results = self.run_concurrently(s, [("same", 1)] * 10)
        self.assertEqual(set(results.values()), {"RESERVED"})
        self.assertEqual(self.stock(), 0)

    def test_crash_before_commit_leaves_no_trace(self):
        s = self.service(1)
        real_connect = s._connect
        s._connect = lambda: CrashOnCommit(real_connect())
        with self.assertRaises(RuntimeError):
            s.reserve("a", 1)
        s._connect = real_connect
        self.assertEqual(self.stock(), 1)
        self.assertEqual(self.reserved_total(), 0)
        self.assertEqual(s.reserve("a", 1), "RESERVED")  # retry succeeds

    def test_crash_does_not_block_other_customers(self):
        s = self.service(1)
        real_connect = s._connect
        s._connect = lambda: CrashOnCommit(real_connect())
        with self.assertRaises(RuntimeError):
            s.reserve("a", 1)
        s._connect = real_connect
        self.assertEqual(s.reserve("b", 1), "RESERVED")
        self.assertEqual(s.reserve("a", 1), "OUT_OF_STOCK")

    def test_restart_keeps_state(self):
        self.service(1).reserve("a", 1)
        restarted = self.service(1)  # must not reset stock
        self.assertEqual(restarted.reserve("a", 1), "RESERVED")
        self.assertEqual(restarted.reserve("b", 1), "OUT_OF_STOCK")


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.service = ReservationService(os.path.join(self.dir.name, "q.db"), 1)

    def tearDown(self):
        self.dir.cleanup()

    def test_earliest_request_wins(self):
        q = RequestQueue(self.service)
        q.add(Request("late", 1, 300))
        q.add(Request("early", 1, 100))
        q.add(Request("mid", 1, 200))
        self.assertEqual(
            q.process(),
            {"early": "RESERVED", "mid": "OUT_OF_STOCK", "late": "OUT_OF_STOCK"},
        )

    def test_failed_request_with_retry_stays_in_queue(self):
        calls = {"n": 0}
        real = self.service.reserve

        def flaky(rid, qty):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("boom")
            return real(rid, qty)

        self.service.reserve = flaky
        q = RequestQueue(self.service)
        q.add(Request("a", 1, 1))
        q.add(Request("b", 1, 2))
        results = q.process(will_retry=lambda r, e: True)
        self.assertEqual(results, {"a": "RESERVED", "b": "OUT_OF_STOCK"})
        self.assertEqual(calls["n"], 4)

    def test_failed_request_without_retry_is_dropped(self):
        real = self.service.reserve

        def fail_a(rid, qty):
            if rid == "a":
                raise RuntimeError("boom")
            return real(rid, qty)

        self.service.reserve = fail_a
        q = RequestQueue(self.service)
        q.add(Request("a", 1, 1))
        q.add(Request("b", 1, 2))
        self.assertEqual(q.process(), {"b": "RESERVED"})
        self.assertEqual(q.pending, [])


if __name__ == "__main__":
    unittest.main()
