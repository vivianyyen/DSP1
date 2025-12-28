"""
Combine MITRE ATT&CK and ATLAS Data

This script merges data from:
1. MITRE ATT&CK (enterprise-attack.json) - Traditional cyber TTPs
2. MITRE ATLAS (atlas.json) - AI/ML-specific TTPs

Output: Combined_ATTACK_ATLAS_Data.json
"""

import json
from collections import defaultdict

print("="*80)
print("COMBINING MITRE ATT&CK AND ATLAS DATA")
print("="*80)

# ============================================================================
# STEP 1: LOAD ATT&CK DATA
# ============================================================================

def load_attack_data(filepath='enterprise-attack-18.1.json'):
    """Load MITRE ATT&CK STIX bundle."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n✓ Loaded ATT&CK data: {len(data.get('objects', []))} objects")
        return data.get('objects', [])
    except FileNotFoundError:
        print(f"\n✗ {filepath} not found")
        return []

# ============================================================================
# STEP 2: LOAD ATLAS DATA
# ============================================================================

def load_atlas_data(filepath='atlas-data.json'):
    """Load MITRE ATLAS data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded ATLAS data: {len(data.get('objects', []))} objects")
        return data.get('objects', [])
    except FileNotFoundError:
        print(f"⚠ {filepath} not found - ATLAS data will be skipped")
        print("  Download from: https://github.com/mitre-atlas/atlas-data")
        return []

# ============================================================================
# STEP 3: EXTRACT AND COMBINE TECHNIQUES
# ============================================================================

def extract_techniques(objects, prefix="T"):
    """Extract techniques from STIX objects."""
    techniques = {}
    
    for obj in objects:
        if obj.get('type') == 'attack-pattern':
            # Get external ID (T1234 or AML.T0001)
            ext_refs = obj.get('external_references', [])
            tech_id = None
            for ref in ext_refs:
                ext_id = ref.get('external_id', '')
                if ext_id.startswith(prefix) or 'AML' in ext_id:
                    tech_id = ext_id
                    break
            
            if not tech_id:
                continue
            
            tech_name = obj.get('name', 'Unknown')
            
            # Extract tactics
            tactics = []
            for phase in obj.get('kill_chain_phases', []):
                phase_name = phase.get('phase_name', '')
                if phase_name:
                    tactics.append(phase_name)
            
            techniques[obj['id']] = {
                'id': tech_id,
                'name': tech_name,
                'tactics': tactics,
                'description': obj.get('description', '')[:300],
                'source': 'ATLAS' if 'AML' in tech_id else 'ATT&CK'
            }
    
    return techniques

# ============================================================================
# STEP 4: EXTRACT GROUPS/ADVERSARIES
# ============================================================================

def extract_groups(objects, source='ATT&CK'):
    """Extract adversary groups."""
    groups = []
    
    for obj in objects:
        if obj.get('type') == 'intrusion-set':
            ext_refs = obj.get('external_references', [])
            group_id = ext_refs[0].get('external_id', 'Unknown') if ext_refs else 'Unknown'
            group_name = obj.get('name', 'Unknown')
            
            groups.append({
                'mitre_attack_id': group_id,
                'mitre_attack_name': group_name,
                'aliases': obj.get('aliases', []),
                'description': obj.get('description', 'No description')[:500],
                'stix_id': obj['id'],
                'mitre_attack_ttps': [],
                'atlas_ttps': [],
                'tactics': set(),
                'source': source
            })
    
    return groups

# ============================================================================
# STEP 5: MAP RELATIONSHIPS
# ============================================================================

def map_relationships(objects, groups, techniques):
    """Map groups to their techniques."""
    
    group_lookup = {g['stix_id']: g for g in groups}
    relationship_count = 0
    
    for obj in objects:
        if obj.get('type') == 'relationship' and obj.get('relationship_type') == 'uses':
            source_ref = obj.get('source_ref', '')
            target_ref = obj.get('target_ref', '')
            
            if source_ref in group_lookup and target_ref in techniques:
                group = group_lookup[source_ref]
                technique = techniques[target_ref]
                
                # Create TTP string
                ttp_entry = f"{technique['id']}: {technique['name']}"
                
                # Separate ATT&CK and ATLAS TTPs
                if technique['source'] == 'ATLAS':
                    group['atlas_ttps'].append(ttp_entry)
                else:
                    group['mitre_attack_ttps'].append(ttp_entry)
                
                # Add tactics
                group['tactics'].update(technique['tactics'])
                
                relationship_count += 1
    
    print(f"✓ Mapped {relationship_count} group-technique relationships")
    
    # Convert sets to lists
    for group in groups:
        group['tactics'] = sorted(list(group['tactics']))
        group['combined_ttps'] = group['mitre_attack_ttps'] + group['atlas_ttps']
        group['attack_ttp_count'] = len(group['mitre_attack_ttps'])
        group['atlas_ttp_count'] = len(group['atlas_ttps'])
        group['total_ttp_count'] = len(group['combined_ttps'])
    
    return groups

# ============================================================================
# STEP 6: ENRICH WITH ML/AI CONTEXT
# ============================================================================

def enrich_with_ml_context(groups):
    """Add ML/AI specific context to groups that use ATLAS techniques."""
    
    for group in groups:
        if group['atlas_ttp_count'] > 0:
            group['ml_ai_threat'] = True
            group['threat_categories'] = ['Traditional Cyber', 'AI/ML Systems']
        else:
            group['ml_ai_threat'] = False
            group['threat_categories'] = ['Traditional Cyber']
    
    return groups

# ============================================================================
# STEP 7: ADD SAMPLE MITIGATIONS
# ============================================================================

def add_sample_mitigations():
    """Create sample mitigations database for common techniques."""
    
    mitigations = {
        # ATT&CK Techniques
        "T1566": {
            "name": "Phishing",
            "mitigations": [
                "M1049: Antivirus/Antimalware - Use anti-malware with email scanning",
                "M1031: Network Intrusion Prevention - Deploy email gateway solutions",
                "M1017: User Training - Conduct regular phishing awareness training",
                "M1021: Restrict Web-Based Content - Block suspicious attachments"
            ]
        },
        "T1059": {
            "name": "Command and Scripting Interpreter",
            "mitigations": [
                "M1038: Execution Prevention - Implement application control",
                "M1049: Antivirus/Antimalware - Use behavior-based detection",
                "M1026: Privileged Account Management - Restrict scripting permissions",
                "M1042: Disable or Remove Feature or Program - Disable unnecessary interpreters"
            ]
        },
        "T1071": {
            "name": "Application Layer Protocol",
            "mitigations": [
                "M1031: Network Intrusion Prevention - Deploy network IDS/IPS",
                "M1037: Filter Network Traffic - Use web proxies with SSL inspection",
                "M1020: SSL/TLS Inspection - Monitor encrypted traffic"
            ]
        },
        "T1003": {
            "name": "OS Credential Dumping",
            "mitigations": [
                "M1028: Operating System Configuration - Enable Credential Guard (Windows)",
                "M1043: Credential Access Protection - Implement LSASS protection",
                "M1026: Privileged Account Management - Minimize admin account usage",
                "M1027: Password Policies - Enforce strong password requirements"
            ]
        },
        "T1082": {
            "name": "System Information Discovery",
            "mitigations": [
                "M1038: Execution Prevention - Limit process execution capabilities",
                "M1018: User Account Management - Apply least privilege principles"
            ]
        },
        "T1486": {
            "name": "Data Encrypted for Impact",
            "mitigations": [
                "M1053: Data Backup - Maintain offline, encrypted backups",
                "M1040: Behavior Prevention on Endpoint - Deploy anti-ransomware",
                "M1022: Restrict File and Directory Permissions - Limit write access"
            ]
        },
        "T1490": {
            "name": "Inhibit System Recovery",
            "mitigations": [
                "M1053: Data Backup - Protect backup systems from tampering",
                "M1028: Operating System Configuration - Restrict access to recovery tools",
                "M1018: User Account Management - Limit admin privileges"
            ]
        },
        "T1047": {
            "name": "Windows Management Instrumentation",
            "mitigations": [
                "M1026: Privileged Account Management - Restrict WMI permissions",
                "M1038: Execution Prevention - Monitor and control WMI execution",
                "M1018: User Account Management - Apply least privilege"
            ]
        },
        
        # ATLAS Techniques (AI/ML specific)
        "AML.T0000": {
            "name": "ML Model Access",
            "mitigations": [
                "M1: Model Access Control - Implement authentication for model APIs",
                "M2: API Rate Limiting - Prevent excessive queries",
                "M3: Output Filtering - Sanitize model responses"
            ]
        },
        "AML.T0001": {
            "name": "Model Inversion",
            "mitigations": [
                "M1: Differential Privacy - Add noise to model outputs",
                "M2: Query Limitation - Restrict number of inference requests",
                "M3: Output Perturbation - Add randomness to predictions"
            ]
        },
        "AML.T0002": {
            "name": "Adversarial Examples",
            "mitigations": [
                "M1: Adversarial Training - Train model with adversarial examples",
                "M2: Input Validation - Sanitize and validate inputs",
                "M3: Ensemble Methods - Use multiple models for consensus"
            ]
        },
        "AML.T0003": {
            "name": "Data Poisoning",
            "mitigations": [
                "M1: Data Validation - Implement anomaly detection on training data",
                "M2: Provenance Tracking - Verify data sources and integrity",
                "M3: Robust Training - Use poisoning-resistant algorithms"
            ]
        }
    }
    
    return mitigations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\nStep 1: Loading data sources...")
    
    # Load ATT&CK
    attack_objects = load_attack_data()
    
    # Load ATLAS (optional)
    atlas_objects = load_atlas_data()
    
    if not attack_objects:
        print("\n✗ ERROR: ATT&CK data is required but not found.")
        print("\nDownload instructions:")
        print("1. Visit: https://github.com/mitre/cti")
        print("2. Download: enterprise-attack/enterprise-attack.json")
        print("3. Rename to: enterprise-attack-18.1.json")
        return
    
    print("\nStep 2: Extracting techniques...")
    
    # Extract techniques from both sources
    attack_techniques = extract_techniques(attack_objects, prefix="T")
    print(f"  ATT&CK techniques: {len(attack_techniques)}")
    
    atlas_techniques = {}
    if atlas_objects:
        atlas_techniques = extract_techniques(atlas_objects, prefix="AML")
        print(f"  ATLAS techniques: {len(atlas_techniques)}")
    
    # Combine technique dictionaries
    all_techniques = {**attack_techniques, **atlas_techniques}
    print(f"  Total combined techniques: {len(all_techniques)}")
    
    print("\nStep 3: Extracting adversary groups...")
    
    # Extract groups
    attack_groups = extract_groups(attack_objects, source='ATT&CK')
    print(f"  ATT&CK groups: {len(attack_groups)}")
    
    atlas_groups = []
    if atlas_objects:
        atlas_groups = extract_groups(atlas_objects, source='ATLAS')
        print(f"  ATLAS groups: {len(atlas_groups)}")
    
    # Combine groups
    all_groups = attack_groups + atlas_groups
    print(f"  Total combined groups: {len(all_groups)}")
    
    print("\nStep 4: Mapping relationships...")
    
    # Map ATT&CK relationships
    all_groups = map_relationships(attack_objects, all_groups, all_techniques)
    
    # Map ATLAS relationships if available
    if atlas_objects:
        all_groups = map_relationships(atlas_objects, all_groups, all_techniques)
    
    print("\nStep 5: Enriching with ML/AI context...")
    all_groups = enrich_with_ml_context(all_groups)
    
    ml_threat_count = sum(1 for g in all_groups if g['ml_ai_threat'])
    print(f"  Groups with ML/AI threats: {ml_threat_count}")
    
    print("\nStep 6: Generating mitigations database...")
    mitigations = add_sample_mitigations()
    print(f"  Mitigations for {len(mitigations)} techniques")
    
    print("\nStep 7: Generating statistics...")
    
    # Statistics
    total_groups = len(all_groups)
    groups_with_ttps = sum(1 for g in all_groups if g['total_ttp_count'] > 0)
    total_attack_ttps = sum(g['attack_ttp_count'] for g in all_groups)
    total_atlas_ttps = sum(g['atlas_ttp_count'] for g in all_groups)
    avg_ttps = total_attack_ttps / total_groups if total_groups > 0 else 0
    
    print("\n" + "="*80)
    print("COMBINED DATASET STATISTICS")
    print("="*80)
    print(f"Total Groups:                {total_groups}")
    print(f"Groups with TTPs:            {groups_with_ttps}")
    print(f"Total ATT&CK TTP mappings:   {total_attack_ttps}")
    print(f"Total ATLAS TTP mappings:    {total_atlas_ttps}")
    print(f"Average TTPs per group:      {avg_ttps:.2f}")
    print(f"ML/AI-aware threats:         {ml_threat_count}")
    print("="*80)
    
    # Show sample groups
    print("\nSample Combined Groups:")
    for i, group in enumerate(all_groups[:3], 1):
        print(f"\n{i}. {group['mitre_attack_name']} ({group['mitre_attack_id']})")
        print(f"   ATT&CK TTPs: {group['attack_ttp_count']}")
        print(f"   ATLAS TTPs: {group['atlas_ttp_count']}")
        print(f"   ML/AI Threat: {'Yes' if group['ml_ai_threat'] else 'No'}")
    
    print("\nStep 8: Saving combined data...")
    
    # Save combined adversary data
    output_file = 'Combined_ATTACK_ATLAS_Data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_groups, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {output_file}")
    
    # Save mitigations separately
    mitigations_file = 'Mitigations_Database.json'
    with open(mitigations_file, 'w', encoding='utf-8') as f:
        json.dump(mitigations, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {mitigations_file}")
    
    print("\n" + "="*80)
    print("✓ DATA COMBINATION COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  1. {output_file} - Combined adversary data")
    print(f"  2. {mitigations_file} - Mitigations database")
    print("\nNext steps:")
    print("  1. Run: streamlit run practical_ti_app.py")
    print("  2. Search by alias or TTPs")
    print("  3. View defensive actions and Diamond Model")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()