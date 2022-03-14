"""Unit tests for :mod:`tmlt.ektelo.utils.cleanup`."""

from pathlib import Path
from random import choice, randint
from string import ascii_letters, digits

from tmlt.core.utils.cleanup import cleanup
from tmlt.core.utils.configuration import Config
from tmlt.core.utils.testing import PySparkTest

# <placeholder: boilerplate>


class TestCleanup(PySparkTest):
    """TestCase for cleanup functions."""

    def _generate_fake_data(self):
        """Generate some arbitrary data (for tests)."""
        rows = []
        size = randint(1, 10)
        for _ in range(size):
            s = ""
            for _ in range(randint(1, 10)):
                s += choice(ascii_letters + digits)
            rows.append((s, randint(-100, 100)))
        return self.spark.createDataFrame(rows, schema=["name", "number"])

    def _get_warehouse_path(self) -> Path:
        """Get a Path object pointing to Spark's warehouse directory."""
        config_path = self.spark.conf.get("spark.sql.warehouse.dir")
        if config_path[0:5] == "file:":
            config_path = config_path[5:]
        return Path(config_path).resolve()

    def setUp(self):
        """Setup."""
        prev_db = self.spark.catalog.currentDatabase()
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS `{Config.temp_db_name()}`;")
        self.spark.catalog.setCurrentDatabase(Config.temp_db_name())
        for i in range(3):
            self._generate_fake_data().write.saveAsTable(f"table{i}")

        self.spark.catalog.setCurrentDatabase(prev_db)

    @staticmethod
    def _recursive_remove(p: Path):
        """Recursively remove a path (just like `rm -r`)"""
        if not p.is_dir():
            p.unlink()
        for f in p.iterdir():
            TestCleanup._recursive_remove(f)
        p.rmdir()

    def tearDown(self):
        self.spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE;")
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name()}*"):
            self._recursive_remove(d)

    def test_cleanup(self):
        """Cleanup cleans up all temporary tables."""
        cleanup()
        self.assertNotIn(Config.temp_db_name(), self.spark.catalog.listDatabases())
        # Check to ensure that the relevant directory is also gone or empty
        # The warehouse may or may not have a directory for the database ...
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name}*"):
            # ... but if it does, that directory should be empty
            self.assertEqual(len(list(d.iterdir())), 0)

        # Cleanup should also work when the database doesn't exist
        cleanup()
        self.assertNotIn("db_that_does_not_exist", self.spark.catalog.listDatabases())
        self.assertNotIn(Config.temp_db_name(), self.spark.catalog.listDatabases())
        # Check to ensure that the relevant directory is also gone or empty
        # The warehouse may or may not have a directory for the database ...
        for d in self._get_warehouse_path().glob(f"*{Config.temp_db_name}*"):
            # ... but if it does, that directory should be empty
            self.assertEqual(len(list(d.iterdir())), 0)
