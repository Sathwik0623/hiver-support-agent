import sqlite3, json, os, time
DB='/mnt/data/hiver_support_agent/data/processed/twcs.sqlite'
OUT='/mnt/data/hiver_support_agent/data/processed/apple_cases.jsonl'
conn=sqlite3.connect(DB); conn.row_factory=sqlite3.Row
conn.execute('PRAGMA temp_store=MEMORY')

print('Building support-interaction membership table...')
start=time.time()
conn.execute('DROP TABLE IF EXISTS apple_case_members')
conn.execute('''
CREATE TABLE apple_case_members AS
WITH RECURSIVE cases(anchor_tweet_id, customer_id, tweet_id, depth) AS (
    SELECT p.tweet_id, p.author_id, p.tweet_id, 0
    FROM tweets a JOIN tweets p ON p.tweet_id=a.in_response_to_tweet_id
    WHERE a.author_id='AppleSupport' AND a.inbound=0 AND p.inbound=1
    UNION ALL
    SELECT c.anchor_tweet_id, c.customer_id, t.tweet_id, c.depth+1
    FROM cases c JOIN tweets t ON t.in_response_to_tweet_id=c.tweet_id
    WHERE c.depth < 20
      AND (t.author_id='AppleSupport' OR t.author_id=c.customer_id)
)
SELECT DISTINCT anchor_tweet_id, customer_id, tweet_id
FROM cases;
''')
conn.execute('CREATE INDEX idx_case_members_anchor ON apple_case_members(anchor_tweet_id)')
conn.execute('CREATE INDEX idx_case_members_tweet ON apple_case_members(tweet_id)')
conn.commit()
print('Membership rows:',conn.execute('SELECT COUNT(*) FROM apple_case_members').fetchone()[0], 'time',time.time()-start)

# Export cases. Use GROUP_CONCAT only for IDs, then fetch messages per case in batches.
anchors=conn.execute('SELECT DISTINCT anchor_tweet_id,customer_id FROM apple_case_members ORDER BY anchor_tweet_id').fetchall()
print('Cases:',len(anchors))
start=time.time(); total=0
with open(OUT,'w',encoding='utf8') as f:
    for i,a in enumerate(anchors,1):
        rows=conn.execute('''
        SELECT t.* FROM tweets t JOIN apple_case_members m ON m.tweet_id=t.tweet_id
        WHERE m.anchor_tweet_id=?
        ORDER BY t.created_at, t.tweet_id
        ''',(a['anchor_tweet_id'],)).fetchall()
        root=conn.execute('''
          WITH RECURSIVE p(tweet_id,parent_id) AS (
            SELECT tweet_id,in_response_to_tweet_id FROM tweets WHERE tweet_id=?
            UNION ALL SELECT t.tweet_id,t.in_response_to_tweet_id FROM tweets t JOIN p ON t.tweet_id=p.parent_id
          ) SELECT tweet_id FROM p WHERE parent_id IS NULL LIMIT 1
        ''',(a['anchor_tweet_id'],)).fetchone()
        case={'case_id':f"APPLE_CASE_{a['anchor_tweet_id']}_{a['customer_id']}",
              'source':'TWCS','brand':'AppleSupport','root_tweet_id':root['tweet_id'] if root else a['anchor_tweet_id'],
              'anchor_tweet_id':a['anchor_tweet_id'],'customer_id':a['customer_id'],
              'message_count':len(rows),'messages':[dict(r) for r in rows]}
        f.write(json.dumps(case,ensure_ascii=False)+'\n'); total+=len(rows)
        if i%5000==0: print(f'exported {i:,}/{len(anchors):,} in {time.time()-start:.1f}s')
print('Wrote',len(anchors),'cases and',total,'messages; output',os.path.getsize(OUT)/1024/1024,'MiB')
conn.close()

