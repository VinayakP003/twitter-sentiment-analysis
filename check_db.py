from sqlalchemy import create_engine, text

engine = create_engine("postgresql://postgres:postgres@localhost:5432/tweetsdb")

with engine.connect() as conn:
    print("Count:", conn.execute(text("SELECT count(*) FROM tweets")).scalar_one())
    rows = conn.execute(text("SELECT id, text, sentiment FROM tweets LIMIT 5")).all()
    print(rows)
