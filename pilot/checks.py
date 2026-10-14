"""Imperfect execution invariance checks; no access to gold answers."""
import math
import time
try:
    from .environment import query
except ImportError:
    from environment import query


def equivalent(left, right):
    """Compare result multisets, preserving duplicates; ignore row ordering."""
    if len(left['columns']) != len(right['columns']):
        return False
    lrows = left['rows']
    rrows = list(right['rows'])
    if len(lrows) != len(rrows):
        return False
    def same_row(lrow,rrow):
        if len(lrow) != len(rrow):
            return False
        return all(math.isclose(l,r,rel_tol=1e-7,abs_tol=1e-7) if isinstance(l,(int,float)) and isinstance(r,(int,float)) else l==r for l,r in zip(lrow,rrow))
    for lrow in lrows:
        match = next((i for i,rrow in enumerate(rrows) if same_row(lrow,rrow)),None)
        if match is None:
            return False
        rrows.pop(match)
    return True


def verify(sql, original_output, seed, contract, database_factory):
    started = time.perf_counter()
    mutations = []
    if contract.get('excludes_cancelled'):
        mutations.append('add_cancelled')
    if contract.get('null_discount_as_zero'):
        mutations.append('zero_missing_discount')
    evidence = []
    for mutation in mutations:
        con, _ = database_factory(seed, mutation=mutation)
        try:
            transformed = query(con, sql)
            evidence.append({'check':mutation,'status':'consistent' if equivalent(original_output,transformed) else 'violation',
                             'transformed_result':transformed})
        except Exception as exc:
            # A broken probe is inconclusive, not proof of a semantic mistake.
            evidence.append({'check':mutation,'status':'inconclusive','detail':str(exc)})
        finally:
            con.close()
    return {'flagged':any(e['status']=='violation' for e in evidence),
            'evidence':evidence, 'seconds':time.perf_counter()-started,
            'interpretation':'Passing these checks does not establish correctness.'}
