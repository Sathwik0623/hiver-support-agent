import csv, sqlite3, os, sys, time

CSV_PATH = os.environ.get('TWCS_CSV', '/mnt/data/twcs.csv')
DB_PATH = os.environ.get('TWCS_DB', '/mnt/data/hiver_support_agent/data/processed/twcs.sqlite')

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
if os.path.exists(DB_PATH):
    print(f'Removing existing DB: {DB_PATH}')
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute('PRAGMA journal_mode=WAL;')
conn.execute('PRAGMA synchronous=OFF;')
conn.execute('PRAGMA temp_store=MEMORY;')
conn.execute('''
CREATE TABLE tweets (
    tweet_id INTEGER PRIMARY KEY,
    author_id TEXT NOT NULL,
    inbound INTEGER NOT NULL,
    created_at TEXT,
    text TEXT,
    response_tweet_id TEXT,
    in_response_to_tweet_id INTEGER
)
''')

insert_sql = '''INSERT INTO tweets
(tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id)
VALUES (?, ?, ?, ?, ?, ?, ?)'''

start = time.time(); rows=[]; count=0
with open(CSV_PATH, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        parent = r['in_response_to_tweet_id'].strip() if r['in_response_to_tweet_id'] else ''
        rows.append((
            int(r['tweet_id']),
            r['author_id'],
            1 if r['inbound'].strip().lower() == 'true' else 0,
            r['created_at'],
            r['text'],
            r['response_tweet_id'],
            int(parent) if parent else None,
        ))
        if len(rows) >= 10000:
            conn.executemany(insert_sql, rows)
            conn.commit(); count += len(rows); rows.clear()
            if count % 200000 == 0:
                print(f'Inserted {count:,} rows in {time.time()-start:.1f}s')
if rows:
    conn.executemany(insert_sql, rows); conn.commit(); count += len(rows)

print(f'Inserted {count:,} rows in {time.time()-start:.1f}s')
conn.execute('CREATE INDEX idx_tweets_parent ON tweets(in_response_to_tweet_id)')
conn.execute('CREATE INDEX idx_tweets_author ON tweets(author_id)')
conn.execute('CREATE INDEX idx_tweets_inbound ON tweets(inbound)')
conn.commit()

apple_count = conn.execute("SELECT COUNT(*) FROM tweets WHERE author_id='AppleSupport'").fetchone()[0]
print(f'AppleSupport tweets: {apple_count:,}')
print(f'DB size: {os.path.getsize(DB_PATH)/1024/1024:.1f} MiB')
conn.close()

