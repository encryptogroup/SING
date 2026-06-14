-- adds missing `base-assignment: silph` metadata

SELECT DISTINCT metadata.shahash
FROM metadata
LEFT JOIN (SELECT DISTINCT metadata.shahash
     FROM metadata
     JOIN (SELECT shahash, assignment_hash FROM metadata WHERE key = 'base-assignment' AND value = 'all-b') metadata3
     ON metadata.shahash = metadata3.shahash AND metadata.assignment_hash = metadata3.assignment_hash
     WHERE key = 'mode' AND value = 'perturb'
) metadata2
ON metadata.shahash = metadata2.shahash
WHERE metadata2.shahash IS NULL;
