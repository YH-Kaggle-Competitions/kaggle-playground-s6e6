import os
import psycopg2

conn = psycopg2.connect(
    os.environ["NEON_DATABASE_URL"]
)

cur = conn.cursor()

cur.execute(
    """
    SELECT
        exp_id,
        competition
    FROM experiments
    WHERE competition = 'playground-series-s6e6'
      AND submitted = FALSE
    ORDER BY created_at DESC
    LIMIT 1
    """
)

row = cur.fetchone()
exp_id = row[0]
csv_path = f"submissions/{exp_id}.csv"

print(row)
print("csv_path =", csv_path)
print("exists =", os.path.exists(csv_path))
