import requests
import json
import re
import sys

def fetch_json(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        raise

def make_variations_custom(variations):
    out = set()
    for item in variations:
        if not isinstance(item, str):
            continue
        v = item.strip().upper()
        out.add(v)
        out.add(v.replace(' ', ''))
        out.add(v.replace('-', ''))
        out.add(v.replace(' ', '').replace('-', ''))
    return list(out)

# MITRE Groups https://attack.mitre.org/groups/
mitre_url = 'https://raw.githubusercontent.com/mitre-attack/attack-stix-data/refs/heads/master/mobile-attack/mobile-attack-18.1.json'
mitre_actors = fetch_json(mitre_url)

# Build a map of objects by id for faster lookups
objects = mitre_actors.get('objects', [])
object_map = {obj.get('id'): obj for obj in objects if 'id' in obj}

mitre_actor_list = []
for adversary in objects:
    if adversary.get('type') != 'intrusion-set':
        continue

    mitre_actor_dict = {}
    mitre_actor_dict['id'] = adversary.get('id')
    mitre_actor_dict['name'] = adversary.get('name')
    created = adversary.get('created', '')
    mitre_actor_dict['created'] = created.split('T')[0] if 'T' in created else created
    last_modified = adversary.get('modified', '')
    mitre_actor_dict['last_modified'] = last_modified.split('T')[0] if 'T' in last_modified else last_modified

    mitre_actor_dict['variations'] = []
    mitre_actor_dict['variations_custom'] = []

    # Extract a sensible URL if present
    for reference in adversary.get('external_references', []):
        if 'url' in reference:
            mitre_actor_dict.setdefault('url', reference['url'])

    # aliases fallback to name if missing
    aliases = adversary.get('aliases')
    if isinstance(aliases, list):
        mitre_actor_dict['variations'].extend(aliases)
    else:
        # fallback
        if adversary.get('name'):
            mitre_actor_dict['variations'].append(adversary['name'])

    mitre_actor_dict['variations_custom'] = make_variations_custom(mitre_actor_dict['variations'])
    mitre_actor_list.append(mitre_actor_dict)

# Populate MITRE Group TTPs
for mitre_actor in mitre_actor_list:
    technique_list = []
    actorID = mitre_actor['id']
    ttp_patternIDs = []

    for rel in objects:
        # relationships can be many types; ensure keys exist
        if rel.get('source_ref') == actorID:
            target = rel.get('target_ref', '')
            if 'attack-pattern' in target:
                ttp_patternIDs.append(target)

    for patternID in ttp_patternIDs:
        object_ttp = object_map.get(patternID)
        if not object_ttp:
            continue
        for external_reference in object_ttp.get('external_references', []):
            ext_id = external_reference.get('external_id')
            if not ext_id:
                continue
            if 'CAPEC' in ext_id:
                continue
            technique_list.append(ext_id)

    # remove duplicates while preserving order
    mitre_actor['TTPs'] = list(dict.fromkeys(technique_list))

# ETDA Actors https://apt.etda.or.th/cgi-bin/listgroups.cgi
etda_url = 'https://apt.etda.or.th/cgi-bin/getmisp.cgi?o=g'
etda_actors = fetch_json(etda_url)

etda_actor_list = []
for adversary in etda_actors.get('values', []):
    etda_actor_dict = {}
    etda_actor_dict['id'] = adversary.get('uuid')
    name = adversary.get('value', '')
    etda_actor_dict['name'] = name

    # Normalize name list (remove brackets)
    name_clean = name.replace('[', '').replace(']', '')
    name_list = re.split(', |,', name_clean)

    etda_actor_dict['variations'] = []
    metadata = adversary.get('meta', {})
    etda_actor_dict['url'] = 'https://apt.etda.or.th/cgi-bin/showcard.cgi?u=' + (adversary.get('uuid') or '')

    etda_actor_dict['created'] = metadata.get('date', 'None Provided')

    synonyms = metadata.get('synonyms')
    if isinstance(synonyms, list):
        for variation in synonyms:
            variation = variation.replace('[', '').replace(']', '')
            etda_actor_dict['variations'].append(variation)

    for name_variation in name_list:
        if name_variation and name_variation not in etda_actor_dict['variations']:
            etda_actor_dict['variations'].append(name_variation)

    if 'country' in metadata:
        etda_actor_dict['country'] = metadata['country']
    if 'motivation' in metadata:
        etda_actor_dict['motivation'] = metadata['motivation']
    if 'cfr-target-category' in metadata:
        etda_actor_dict['targeted_industries'] = metadata['cfr-target-category']
    if 'cfr-suspected-victims' in metadata:
        etda_actor_dict['targeted_countries'] = metadata['cfr-suspected-victims']

    etda_actor_dict['variations_custom'] = make_variations_custom(etda_actor_dict['variations'])
    etda_actor_list.append(etda_actor_dict)

# Comparison
merge_list = []
id_check = set()
for mitre_actor in mitre_actor_list:
    matched = False
    for mitre_variation in mitre_actor['variations_custom']:
        for etda_actor in etda_actor_list:
            if mitre_variation in etda_actor['variations_custom']:
                if mitre_actor['id'] in id_check:
                    matched = True
                    break
                id_check.add(mitre_actor['id'])
                merge_dict = {
                    'mitre_attack_id': mitre_actor['id'],
                    'mitre_attack_name': mitre_actor['name'],
                    'mitre_attack_aliases': mitre_actor['variations'],
                    'mitre_attack_created': mitre_actor['created'],
                    'mitre_attack_last_modified': mitre_actor['last_modified'],
                    'mitre_url': mitre_actor.get('url'),
                    'etda_id': etda_actor.get('id'),
                    'etda_name': etda_actor.get('name'),
                    'etda_aliases': etda_actor.get('variations'),
                    'etda_first_seen': etda_actor.get('created'),
                    'etda_url': etda_actor.get('url'),
                    'country': etda_actor.get('country', 'None Provided'),
                    'motivation': etda_actor.get('motivation', 'None Provided'),
                    'victim_industries': etda_actor.get('targeted_industries', 'None Provided'),
                    'victim_countries': etda_actor.get('targeted_countries', 'None Provided'),
                    'mitre_attack_ttps': mitre_actor.get('TTPs', []),
                }
                merge_list.append(merge_dict)
                matched = True
                break
        if matched:
            break

with open('Categorized_Adversary_TTPs.json', 'w', encoding='utf-8') as outfile:
    json.dump(merge_list, outfile, indent=2, ensure_ascii=False)


try:
    print(f"MITRE actors: {len(mitre_actor_list)}")
    print(f"ETDA actors: {len(etda_actor_list)}")
    print(f"Matches found: {len(merge_list)}")
    print("Wrote file: Categorized_Adversary_TTPs.json")
    if len(merge_list) > 0:
        print("Sample merged entry:")
        print(json.dumps(merge_list[0], indent=2, ensure_ascii=False))
except Exception as e:
    import traceback
    print("Error summarizing results:", e)
    traceback.print_exc()
