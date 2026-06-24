import os
import subprocess
import psycopg2

COMPETITION = "playground-series-s6e6"
DATABASE_URL = os.environ["NEON_DATABASE_URL"]

result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", COMPETITION],
    capture_output=True,
    text=True,
    check=True,
)

lines = result.stdout.splitlines()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT exp_id
    FROM experiments
    WHERE competition = %s
      AND submitted = TRUE
      AND public_lb IS NULL
    """,
    (COMPETITION,)
)

pending = [row[0] for row in cur.fetchall()]

for exp_id in pending:
    # filename = f"{exp_id}.csv"
    filename = f"submission.csv"
    for line in lines:
        if filename in line and "COMPLETE" in line:
            parts = line.split()
            public_lb = float(parts[-1])

            cur.execute(
                """
                UPDATE experiments
                SET public_lb = %s
                WHERE exp_id = %s
                """,
                (public_lb, exp_id),
            )

            print(f"updated {exp_id}: {public_lb}")
            break

conn.commit()
conn.close()
