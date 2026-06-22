import os
import subprocess
import psycopg2

DATABASE_URL = os.environ["NEON_DATABASE_URL"]
COMPETITION = "playground-series-s6e6"

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT exp_id, competition
    FROM experiments
    WHERE competition = %s
      AND submitted = FALSE
    ORDER BY created_at DESC
    LIMIT 1
    """,
    (COMPETITION,)
)

row = cur.fetchone()

if row is None:
    print("No pending submission.")
    conn.close()
    raise SystemExit(0)

exp_id, competition = row
csv_path = f"submissions/{exp_id}.csv"

print("candidate:", row)
print("csv_path:", csv_path)
print("exists:", os.path.exists(csv_path))

if not os.path.exists(csv_path):
    raise FileNotFoundError(csv_path)

subprocess.run(
    [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        csv_path,
        "-m",
        exp_id,
    ],
    check=True,
)

cur.execute(
    """
    UPDATE experiments
    SET submitted = TRUE
    WHERE exp_id = %s
    """,
    (exp_id,)
)

conn.commit()
conn.close()

print("submitted and updated:", exp_id)
