
"""Common utilities for ecological VideoQA modifications built on LLoVi captions.
The functions are deliberately light-weight so that experiments can run on a single RTX 4090.
"""
import re
import math
from collections import defaultdict, Counter

ANIMAL_WORDS = [
    'animal','person','man','woman','monkey','deer','bird','dog','cat','bear','panda','fox','rabbit','horse','cow',
    'sheep','goat','fish','duck','chicken','insect','butterfly','tiger','lion','wolf','boar','squirrel'
]
OBJECT_WORDS = [
    'water','water source','stream','river','tree','forest','grass','nest','cave','road','area','monitoring area',
    'ground','camera','food','plant','flower','branch','rock','field','door','table','room','kitchen','vehicle'
]
ACTION_MAP = {
    'enter': ['enter','comes into','come into','walks in','goes into','appears','arrives'],
    'leave': ['leave','leaves','left','exits','walks away','goes away','disappears'],
    'approach': ['approach','approaches','moves toward','walks toward','goes toward','near','closer'],
    'stay': ['stay','stays','standing','stands','sitting','sits','waiting','remains','pauses','stops'],
    'move': ['move','moves','walk','walks','run','runs','fly','flies','swim','swims','cross','crosses'],
    'feed': ['eat','eats','feeding','feeds','graze','grazes'],
    'drink': ['drink','drinks','drinking'],
    'interact': ['interact','interacts','touch','touches','fight','fights','chase','chases','follow','follows','play','plays'],
    'return': ['return','returns','comes back','back again'],
    'look': ['look','looks','watch','watches','observe','observes']
}
STATE_AFTER = {
    'enter': ('outside', 'inside monitored area'),
    'leave': ('inside monitored area', 'outside'),
    'approach': ('far from object', 'near object'),
    'stay': ('moving', 'staying'),
    'move': ('previous location', 'new location'),
    'feed': ('not feeding', 'feeding'),
    'drink': ('not drinking', 'drinking'),
    'interact': ('separate', 'interacting'),
    'return': ('outside after leaving', 'inside again'),
    'look': ('not attending', 'attending'),
    'unknown': ('unknown', 'unknown')
}


def safe_lower(x):
    return str(x).lower() if x is not None else ''


def split_narration(narration):
    """Parse LLoVi narration lines like '12: caption'. Returns [{'index','caption'}]."""
    items = []
    if isinstance(narration, list):
        lines = [str(x) for x in narration]
    else:
        lines = str(narration).split('\n')
    for k, line in enumerate(lines):
        line = line.strip().strip('.')
        if not line:
            continue
        m = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
        if m:
            idx, cap = int(m.group(1)), m.group(2).strip()
        else:
            idx, cap = k, line
        if cap:
            items.append({'index': idx, 'caption': cap})
    return items


def find_first(text, vocab, default='unknown'):
    t = safe_lower(text)
    for w in vocab:
        if w in t:
            return w
    return default


def detect_action(text):
    t = safe_lower(text)
    best = 'unknown'
    for act, keys in ACTION_MAP.items():
        for key in keys:
            if key in t:
                return act
    return best


def extract_events(narration):
    events = []
    for item in split_narration(narration):
        cap = item['caption']
        action = detect_action(cap)
        subject = find_first(cap, ANIMAL_WORDS, default='animal')
        obj = find_first(cap, OBJECT_WORDS, default='scene')
        before, after = STATE_AFTER.get(action, ('unknown','unknown'))
        if obj != 'scene':
            after = after.replace('object', obj)
        conf = 0.55 + (0.15 if action != 'unknown' else 0.0) + (0.10 if subject != 'animal' else 0.0) + (0.10 if obj != 'scene' else 0.0)
        conf = min(conf, 0.95)
        events.append({
            'event_id': f"e{len(events)+1}",
            'time': item['index'],
            'start': item['index'],
            'end': item['index'] + 1,
            'caption': cap,
            'subject': subject,
            'action': action,
            'object': obj,
            'before_state': before,
            'after_state': after,
            'location': obj if obj != 'scene' else 'unknown',
            'confidence': round(conf, 3),
        })
    return events


def build_state_memory(events):
    memory = []
    for e in events:
        memory.append({
            'memory_id': 'm' + e['event_id'][1:],
            **e,
            'transition': f"{e['before_state']} -> {e['after_state']}",
        })
    return memory


def build_cards(events):
    role = defaultdict(lambda: {'times': [], 'actions': [], 'objects': [], 'events': []})
    obj = defaultdict(lambda: {'subjects': [], 'actions': [], 'times': [], 'events': []})
    interactions = []
    for e in events:
        r = e['subject']; o = e['object']
        role[r]['times'].append(e['time']); role[r]['actions'].append(e['action']); role[r]['objects'].append(o); role[r]['events'].append(e['event_id'])
        obj[o]['subjects'].append(r); obj[o]['actions'].append(e['action']); obj[o]['times'].append(e['time']); obj[o]['events'].append(e['event_id'])
        if e['action'] in {'approach','interact','feed','drink','stay','return'} or o not in {'scene','unknown'}:
            interactions.append({'subject': r, 'object': o, 'relation': e['action'], 'time': e['time'], 'event_id': e['event_id'], 'confidence': e['confidence']})
    role_cards = [{'subject': k, **v, 'duration': len(v['times'])} for k, v in role.items()]
    object_cards = [{'object': k, **v, 'frequency': len(v['times'])} for k, v in obj.items()]
    return {'role_cards': role_cards, 'object_cards': object_cards, 'interaction_cards': interactions}


def infer_question_type(question):
    q = safe_lower(question)
    if any(w in q for w in ['before','after','then','next','return','again','leave','enter','stay','start','finish']):
        return 'state'
    if any(w in q for w in ['interact','together','with','follow','fight','chase','near','approach']):
        return 'interaction'
    if any(w in q for w in ['what animal','which animal','species','who']):
        return 'species'
    if any(w in q for w in ['abnormal','danger','alert','unusual']):
        return 'abnormal'
    return 'action'


def tokenize(text):
    return set(re.findall(r'[a-zA-Z]+', safe_lower(text)))


def score_evidence(question, ev, qtype=None):
    qt = tokenize(question)
    et = tokenize(' '.join(str(v) for v in ev.values()))
    lexical = len(qt & et) / max(1, len(qt))
    conf = float(ev.get('confidence', 0.6))
    type_bonus = 0.0
    qtype = qtype or infer_question_type(question)
    if qtype == 'state' and ('transition' in ev or ev.get('action') in {'enter','leave','approach','stay','return'}): type_bonus += 0.25
    if qtype == 'interaction' and ev.get('action') in {'interact','approach','feed','drink'}: type_bonus += 0.25
    return lexical + 0.35 * conf + type_bonus


def select_evidence(question, events, memory=None, cards=None, top_k=8):
    qtype = infer_question_type(question)
    candidates = []
    if qtype == 'state' and memory:
        candidates.extend(memory)
    else:
        candidates.extend(events)
    if cards and qtype == 'interaction':
        for c in cards.get('interaction_cards', []):
            candidates.append({**c, 'caption': str(c), 'confidence': c.get('confidence', 0.6)})
    scored = sorted(((score_evidence(question, c, qtype), c) for c in candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def format_events(events, max_items=16):
    lines = []
    for e in events[:max_items]:
        if 'transition' in e:
            lines.append(f"[{e.get('event_id', e.get('memory_id','e?'))} | t={e.get('time')}] subject={e.get('subject')}; action={e.get('action')}; object={e.get('object')}; state={e.get('transition')}; evidence='{e.get('caption','')}'")
        else:
            lines.append(f"[{e.get('event_id','e?')} | t={e.get('time')}] subject={e.get('subject')}; action={e.get('action')}; object={e.get('object')}; evidence='{e.get('caption','')}'")
    return '\n'.join(lines) if lines else 'No reliable event evidence was extracted.'


def letter_to_int(x):
    if isinstance(x, int): return x
    s = str(x).strip()
    if s.isdigit(): return int(s)
    return {'A':0,'B':1,'C':2,'D':3,'E':4}.get(s[:1].upper(), -1)


def parse_letter(text):
    s = str(text).strip()
    m = re.search(r'\b([A-E])\b', s.upper())
    return letter_to_int(m.group(1)) if m else -1


def word_count(s):
    return len(str(s).split())


def simple_similarity(a, b):
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))
