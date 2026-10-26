"""Who gets the last item? Reservation service using a transactional DB (sqlite as stand-in).

Invariant: stock >= 0 and stock + sum(reserved quantities) == initial stock.
It holds because the stock decrement, the reservation row and the idempotency
record are written in ONE transaction, and the decrement is conditional
(UPDATE ... WHERE stock >= qty), so the DB serializes competing requests.
"""
import sqlite3
import threading
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    quantity: int
    time_ms: int  # arrival time; only used to order the waiting queue


class ReservationService:
    def __init__(self, path, initial_stock):
        self.path = path
        con = self._connect()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS inventory (id INTEGER PRIMARY KEY CHECK (id = 1),
                                                  stock INTEGER NOT NULL CHECK (stock >= 0));
            -- one row per requestId: the stored result is what retries get back
            CREATE TABLE IF NOT EXISTS reservations (request_id TEXT PRIMARY KEY,
                                                     quantity INTEGER NOT NULL,
                                                     status TEXT NOT NULL);
            """
        )
        con.execute("INSERT OR IGNORE INTO inventory VALUES (1, ?)", (initial_stock,))
        con.close()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    def reserve(self, request_id, quantity):
        """Idempotent: same request_id always returns the same status."""
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")  # write lock: the winner is decided here
            row = con.execute(
                "SELECT status FROM reservations WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row:  # retry (also covers OUT_OF_STOCK)
                con.execute("COMMIT")
                return row[0]

            cur = con.execute(
                "UPDATE inventory SET stock = stock - ? WHERE id = 1 AND stock >= ?",
                (quantity, quantity),
            )
            status = "RESERVED" if cur.rowcount == 1 else "OUT_OF_STOCK"
            con.execute(
                "INSERT INTO reservations VALUES (?, ?, ?)", (request_id, quantity, status)
            )
            con.execute("COMMIT")  # crash before this line => nothing happened, retry is safe
            return status
        except Exception:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            con.close()


class RequestQueue:
    """Process by arrival time, one at a time. A failed request that will retry
    stays at the head; otherwise it is removed."""

    def __init__(self, service):
        self.service = service
        self.pending = []

    def add(self, request):
        self.pending.append(request)
        self.pending.sort(key=lambda r: r.time_ms)

    def process(self, will_retry=lambda request, error: False):
        results = {}
        while self.pending:
            request = self.pending[0]
            try:
                results[request.request_id] = self.service.reserve(
                    request.request_id, request.quantity
                )
                self.pending.pop(0)  # reserved or out of stock
            except Exception as error:
                if not will_retry(request, error):
                    self.pending.pop(0)
        return results


if __name__ == "__main__":
    import os
    import tempfile

    service = ReservationService(os.path.join(tempfile.mkdtemp(), "shop.db"), 1)
    outcomes = {}

    def worker(i):
        outcomes[f"c{i}"] = service.reserve(f"c{i}", 1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(outcomes)
    assert list(outcomes.values()).count("RESERVED") == 1

    for rid, status in outcomes.items():  # retries give the same answer
        assert service.reserve(rid, 1) == status

    q = RequestQueue(ReservationService(os.path.join(tempfile.mkdtemp(), "q.db"), 1))
    q.add(Request("late", 1, 200))
    q.add(Request("early", 1, 100))
    print(q.process())  # {'early': 'RESERVED', 'late': 'OUT_OF_STOCK'}

