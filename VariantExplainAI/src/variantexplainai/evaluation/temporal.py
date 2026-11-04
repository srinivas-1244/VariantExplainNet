"""Strict temporal identity filtering: no historical variants reused for prospectively evaluated set."""
def temporal_new_variants(development, later):
    original={r['id'] for r in development}
    new=[r for r in later if r['id'] not in original]
    if len({r['id'] for r in new})!=len(new):raise ValueError('Temporal holdout has duplicates')
    return new

def overlap_categories(train,eval_records):
    genes={r['gene'] for r in train if r['gene']}
    counts={'seen_gene':0,'unseen_gene':0,'unassigned':0}
    for r in eval_records:
        if not r.get('gene'):counts['unassigned']+=1
        elif r['gene'] in genes:counts['seen_gene']+=1
        else:counts['unseen_gene']+=1
    return counts
