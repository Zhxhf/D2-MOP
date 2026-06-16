
import re
from eco_common import extract_events, simple_similarity, tokenize

TEMPORAL_KEYWORDS = {
    'before': ['before','prior'],
    'after': ['after','then','next','following'],
    'repeat': ['again','repeat','twice','multiple','more than once'],
    'return': ['return','come back','comes back','back again'],
    'stay': ['stay','stays','remain','remains','still'],
    'enter': ['enter','comes into','go into'],
    'leave': ['leave','exit','go away']
}

def parse_logic(question):
    q = question.lower()
    for name, keys in TEMPORAL_KEYWORDS.items():
        if any(k in q for k in keys):
            return {'op': name, 'raw': question}
    return {'op': 'general', 'raw': question}


def execute_logic(program, events):
    op = program.get('op','general')
    if not events:
        return 'No visual event evidence is available.', []
    if op == 'before':
        # find an event before the strongest query-related event
        path = events[:2]
        ans = 'The relevant earlier event is: ' + (path[0]['caption'] if path else 'unknown')
    elif op == 'after':
        path = events[-2:]
        ans = 'The later event is: ' + (path[-1]['caption'] if path else 'unknown')
    elif op == 'repeat':
        counts = {}
        for e in events: counts[e['action']] = counts.get(e['action'],0)+1
        repeated = [a for a,c in counts.items() if c >= 2 and a != 'unknown']
        path = [e for e in events if e['action'] in repeated[:1]][:4]
        ans = 'Yes, a repeated action appears.' if repeated else 'No clear repeated action is observed.'
    elif op == 'return':
        leave_idx = [i for i,e in enumerate(events) if e['action']=='leave']
        enter_idx = [i for i,e in enumerate(events) if e['action']=='enter']
        ok = any(j > i for i in leave_idx for j in enter_idx)
        path = [events[i] for i in leave_idx[:1] + enter_idx[:1]]
        ans = 'Yes, the subject leaves and later returns.' if ok else 'No clear leave-and-return pattern is observed.'
    elif op == 'stay':
        path = [e for e in events if e['action']=='stay'][:4]
        ans = 'Yes, a staying/remain event is observed.' if path else 'No clear staying event is observed.'
    elif op in {'enter','leave'}:
        path = [e for e in events if e['action']==op][:4]
        ans = f"Yes, an {op} event is observed." if path else f"No clear {op} event is observed."
    else:
        path = events[:6]
        ans = 'The answer should be inferred from the listed video events.'
    return ans, path


def extract_claims(answer):
    # simple clause-level claim extraction
    parts = re.split(r'\band\b|,|;|\.\s+', str(answer))
    claims = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 2:
            claims.append({'text': p})
    return claims[:8]


def support_claim(claim, events):
    text = claim.get('text','')
    if not events: return 0.0, None
    scored = [(simple_similarity(text, e.get('caption','') + ' ' + e.get('action','') + ' ' + e.get('object','')), e) for e in events]
    scored.sort(key=lambda x:x[0], reverse=True)
    return scored[0]


def repair_answer(answer, events, threshold=0.10):
    claims = extract_claims(answer)
    kept = []
    unsupported = []
    for cl in claims:
        s, ev = support_claim(cl, events)
        if s >= threshold:
            kept.append(cl['text'])
        else:
            unsupported.append(cl['text'])
    if kept:
        repaired = '. '.join(kept)
    else:
        # conservative fallback: summarize strongest events
        repaired = 'Evidence shows: ' + '; '.join(e['caption'] for e in events[:3])
    if unsupported:
        repaired += '. No clear evidence supports: ' + '; '.join(unsupported[:3])
    return repaired, {'claims': claims, 'unsupported': unsupported, 'unsupported_rate': len(unsupported)/max(1,len(claims))}


def choice_score(answer, logic_answer, path, choice):
    path_text = ' '.join(e.get('caption','') for e in path)
    return 0.60*simple_similarity(answer, choice) + 0.25*simple_similarity(logic_answer, choice) + 0.15*simple_similarity(path_text, choice)
