import os
import psycopg2

DATABASE_URL = os.environ["NEON_DATABASE_URL"]

conn = psycopg2.connect(DATABASE_URL)

cur = conn.cursor()

cur.execute(
    """
    SELECT
        exp_id,
        ensemble_cv_accuracy
    FROM experiments
    WHERE submitted = FALSE
    ORDER BY created_at DESC
    LIMIT 1
    """
)

row = cur.fetchone()

print("latest experiment:")
print(row)

conn.close()
