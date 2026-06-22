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
        competition,
        ensemble_cv_accuracy
    FROM experiments
    WHERE competition = 'playground-series-s6e6'
      AND submitted = FALSE
    ORDER BY ensemble_cv_accuracy DESC
    LIMIT 1
    """
)

row = cur.fetchone()

print("candidate:", row)

conn.close()
