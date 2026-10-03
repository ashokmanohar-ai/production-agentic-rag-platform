from app.config import get_settings
from app.persistence.database import build_session_factory


if __name__ == "__main__":
    build_session_factory(get_settings().database_url)
