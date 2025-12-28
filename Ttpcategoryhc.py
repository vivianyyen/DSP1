"""
ttpCategory.py - Extract and categorize TTPs from MITRE ATT&CK STIX bundle

This script processes the raw MITRE ATT&CK JSON (STIX format) and extracts:
1. Adversary/Group information
2. Their associated TTPs (Techniques)
3. Tactics for categorization
4. Creates a structured JSON for clustering analysis

Input: enterprise-attack-18.1.json (MITRE ATT&CK STIX bundle)
Output: Categorized_Adversary_TTPs.json (processed adversary data)
"""

import json
from collections import defaultdict

def load_attack_data(filepath):
    """Load MITRE ATT&CK STIX bundle."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'objects' not in data:
            raise ValueError("Invalid STIX bundle format: missing 'objects' key")
        
        print(f"✓ Loaded MITRE ATT&CK bundle: {len(data['objects'])} objects")
        return data['objects']
    
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        print("\nTo download MITRE ATT&CK data:")
        print("1. Visit: https://github.com/mitre/cti")
        print("2. Download: enterprise-attack/enterprise-attack.json")
        print("3. Rename to: enterprise-attack-18.1.json")
        print("4. Place in the same directory as this script")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        exit(1)

def extract_techniques(objects):
    """Extract techniques/sub-techniques with their tactics and metadata."""
    techniques = {}
    
    for obj in objects:
        if obj.get('type') == 'attack-pattern':
            tech_id = obj.get('external_references', [{}])[0].get('external_id', 'Unknown')
            tech_name = obj.get('name', 'Unknown')
            
            # Extract kill chain phases (tactics)
            tactics = []
            for phase in obj.get('kill_chain_phases', []):
                if phase.get('kill_chain_name') == 'mitre-attack':
                    tactics.append(phase.get('phase_name', ''))
            
            techniques[obj['id']] = {
                'id': tech_id,
                'name': tech_name,
                'tactics': tactics,
                'description': obj.get('description', '')[:200]  # Truncate for brevity
            }
    
    print(f"✓ Extracted {len(techniques)} techniques")
    return techniques

def extract_groups(objects, techniques):
    """Extract adversary groups and their TTPs."""
    groups = []
    
    for obj in objects:
        if obj.get('type') == 'intrusion-set':
            group_id = obj.get('external_references', [{}])[0].get('external_id', 'Unknown')
            group_name = obj.get('name', 'Unknown')
            
            # Get aliases
            aliases = obj.get('aliases', [])
            
            # Extract description
            description = obj.get('description', 'No description available')
            
            groups.append({
                'mitre_attack_id': group_id,
                'mitre_attack_name': group_name,
                'aliases': aliases,
                'description': description[:300],  # Truncate
                'stix_id': obj['id'],
                'mitre_attack_ttps': [],  # Will be populated from relationships
                'tactics': set()  # Will be populated from techniques
            })
    
    print(f"✓ Extracted {len(groups)} adversary groups")
    return groups

def extract_relationships(objects, groups, techniques):
    """Map groups to their techniques via STIX relationships."""
    
    # Create lookup dictionaries
    group_lookup = {g['stix_id']: g for g in groups}
    
    relationship_count = 0
    
    for obj in objects:
        if obj.get('type') == 'relationship' and obj.get('relationship_type') == 'uses':
            source_ref = obj.get('source_ref', '')
            target_ref = obj.get('target_ref', '')
            
            # Check if source is a group and target is a technique
            if source_ref in group_lookup and target_ref in techniques:
                group = group_lookup[source_ref]
                technique = techniques[target_ref]
                
                # Add technique ID to group's TTP list
                ttp_entry = f"{technique['id']}: {technique['name']}"
                group['mitre_attack_ttps'].append(ttp_entry)
                
                # Add tactics from this technique
                group['tactics'].update(technique['tactics'])
                
                relationship_count += 1
    
    print(f"✓ Mapped {relationship_count} group-technique relationships")
    
    # Convert tactics set to list for JSON serialization
    for group in groups:
        group['tactics'] = sorted(list(group['tactics']))
        group['ttp_count'] = len(group['mitre_attack_ttps'])
    
    return groups

def categorize_by_tactics(groups):
    """Categorize groups by their primary tactics."""
    
    tactic_categories = defaultdict(list)
    
    for group in groups:
        primary_tactics = group.get('tactics', [])[:3]  # Top 3 tactics
        
        if not primary_tactics:
            tactic_categories['Unknown'].append(group['mitre_attack_name'])
        else:
            for tactic in primary_tactics:
                tactic_categories[tactic].append(group['mitre_attack_name'])
    
    print("\n" + "="*60)
    print("TACTIC-BASED CATEGORIZATION")
    print("="*60)
    
    for tactic, group_list in sorted(tactic_categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{tactic.upper()} ({len(group_list)} groups):")
        print(f"  {', '.join(group_list[:5])}")
        if len(group_list) > 5:
            print(f"  ... and {len(group_list) - 5} more")
    
    return tactic_categories

def generate_statistics(groups):
    """Generate statistics about the extracted data."""
    
    total_groups = len(groups)
    groups_with_ttps = sum(1 for g in groups if g['ttp_count'] > 0)
    total_ttps = sum(g['ttp_count'] for g in groups)
    avg_ttps = total_ttps / total_groups if total_groups > 0 else 0
    
    print("\n" + "="*60)
    print("EXTRACTION STATISTICS")
    print("="*60)
    print(f"Total Groups:              {total_groups}")
    print(f"Groups with TTPs:          {groups_with_ttps}")
    print(f"Groups without TTPs:       {total_groups - groups_with_ttps}")
    print(f"Total TTP associations:    {total_ttps}")
    print(f"Average TTPs per group:    {avg_ttps:.2f}")
    print(f"Max TTPs (single group):   {max(g['ttp_count'] for g in groups) if groups else 0}")
    print(f"Min TTPs (single group):   {min(g['ttp_count'] for g in groups) if groups else 0}")
    print("="*60 + "\n")
    
    # Show top 5 groups by TTP count
    top_groups = sorted(groups, key=lambda x: x['ttp_count'], reverse=True)[:5]
    print("TOP 5 GROUPS BY TTP COUNT:")
    for i, group in enumerate(top_groups, 1):
        print(f"  {i}. {group['mitre_attack_name']} ({group['mitre_attack_id']}): {group['ttp_count']} TTPs")
    print()

def save_output(groups, output_file):
    """Save processed data to JSON."""
    
    # Filter out groups with no TTPs (optional - comment out to keep all)
    # groups_filtered = [g for g in groups if g['ttp_count'] > 0]
    groups_filtered = groups  # Keep all groups
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(groups_filtered, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(groups_filtered)} groups to: {output_file}")

def main():
    """Main execution flow."""
    
    print("\n" + "="*60)
    print("MITRE ATT&CK TTP EXTRACTION & CATEGORIZATION")
    print("="*60 + "\n")
    
    # Configuration
    input_file = 'enterprise-attack-18.1.json'
    output_file = 'Categorized_Adversary_TTPs_hc.json'
    
    # Step 1: Load data
    print("Step 1: Loading MITRE ATT&CK data...")
    objects = load_attack_data(input_file)
    
    # Step 2: Extract techniques
    print("\nStep 2: Extracting techniques...")
    techniques = extract_techniques(objects)
    
    # Step 3: Extract groups
    print("\nStep 3: Extracting adversary groups...")
    groups = extract_groups(objects, techniques)
    
    # Step 4: Map relationships
    print("\nStep 4: Mapping group-technique relationships...")
    groups = extract_relationships(objects, groups, techniques)
    
    # Step 5: Categorize by tactics
    print("\nStep 5: Categorizing by tactics...")
    categorize_by_tactics(groups)
    
    # Step 6: Generate statistics
    generate_statistics(groups)
    
    # Step 7: Save output
    print("Step 6: Saving output...")
    save_output(groups, output_file)
    
    print("\n" + "="*60)
    print("✓ EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review: {output_file}")
    print(f"2. Run clustering: python hierarchical_clustering.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()