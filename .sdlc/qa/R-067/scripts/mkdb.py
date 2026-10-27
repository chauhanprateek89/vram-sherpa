import sys
from pathlib import Path
from vramsherpa import database
from vramsherpa.catalog import seed_catalog
from vramsherpa.config import Settings
from vramsherpa.main import create_app
from sqlalchemy import text
S=Path(sys.argv[1])
app=create_app(Settings(database_url=None,app_env="test",allowed_hosts=("*",)),database_url=f"sqlite+pysqlite:///{S/'nosrc.db'}")
eng=database.get_engine()
database.Base.metadata.create_all(bind=eng)
with database.SessionLocal() as s:
    seed_catalog(s,data_dir=Path('data'))
    s.commit()
with eng.begin() as c:
    c.execute(text("update models set sources=NULL")); c.execute(text("update variants set sources=NULL"))
    print(c.execute(text("select count(*) from models where sources is not null")).scalar())
