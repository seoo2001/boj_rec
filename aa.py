import sqlite3

# 소스 DB (원본 problems 테이블)
source_conn = sqlite3.connect("_pre/data/raw/baekjoon.db")
source_cursor = source_conn.cursor()

# 타겟 DB (최종 저장할 곳)
target_conn = sqlite3.connect("baekjoon.db")
target_cursor = target_conn.cursor()

# 1. 모든 컬럼 포함해서 가져오기
source_cursor.execute("""
    SELECT id, problem_id, title, is_solvable, accepted_user_count, level, average_tries, tags FROM problems
""")
rows = source_cursor.fetchall()

# 2. target DB로 insert (id 포함하여 삽입)
insert_sql = """
INSERT OR REPLACE INTO problems 
(id, problem_id, title, is_solvable, accepted_user_count, level, average_tries, tags)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
target_cursor.executemany(insert_sql, rows)

# 3. 커밋 & 닫기
target_conn.commit()
source_conn.close()
target_conn.close()

print(f"{len(rows)}개의 문제 데이터를 id 포함하여 복사 완료했습니다.")