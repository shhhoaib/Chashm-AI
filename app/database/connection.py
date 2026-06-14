from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from app.config import DATABASE_URL
from app.database.models import Base


class DatabaseManager:
    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._engine = create_engine(
            DATABASE_URL,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
        )
        if "sqlite" in DATABASE_URL:
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self._session_factory = sessionmaker(bind=self._engine)

    def create_tables(self):
        Base.metadata.create_all(self._engine)
        self._migrate_schema()

    def _migrate_schema(self):
        """Add new columns to existing tables if they don't exist."""
        import sqlalchemy as sa
        inspector = sa.inspect(self._engine)
        for table, columns in [
            ("scan_records", ["image_type", "recommendation", "disclaimer", "results_json"]),
        ]:
            existing = [c["name"] for c in inspector.get_columns(table)]
            for col in columns:
                if col not in existing:
                    try:
                        with self._engine.connect() as conn:
                            conn.execute(sa.text(
                                f"ALTER TABLE {table} ADD COLUMN {col} TEXT"
                            ))
                            conn.commit()
                    except Exception:
                        pass

    def get_session(self) -> Session:
        return self._session_factory()

    @property
    def engine(self):
        return self._engine


db = DatabaseManager()


def get_session() -> Session:
    return db.get_session()
