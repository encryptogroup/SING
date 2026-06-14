-- adds missing `base-assignment: silph` metadata

INSERT OR IGNORE INTO metadata
SELECT DISTINCT metadata.shahash, metadata.assignment_hash, 'base-assignment', 'silph'
FROM metadata
JOIN (SELECT shahash, assignment_hash FROM metadata WHERE key = 'mode' AND value = 'random') metadata2
ON metadata.shahash = metadata2.shahash AND metadata.assignment_hash = metadata2.assignment_hash;
