"""
Practical Threat Intelligence System - Streamlit Dashboard

Features:
1. Input: Alias or TTPs → Identify Threat Actor
2. Output: Threat actor name/code
3. Defensive Actions: Recommended mitigations
4. Diamond Model: Visual representation
5. Combined ATT&CK + ATLAS data

Run with: streamlit run practical_ti_app.py
"""

import streamlit as st
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Threat Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .threat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .defense-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .warning-badge {
        background-color: #ffc107;
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .danger-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .diamond-node {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_combined_data():
    """Load combined ATT&CK + ATLAS adversary data."""
    try:
        # Try to load combined data
        with open('Combined_ATTACK_ATLAS_Data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Fallback to ATT&CK only
        try:
            with open('Hierarchical_Clustered_Adversaries.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return None

@st.cache_resource
def load_model_and_vectorizer():
    """Load trained model and vectorizer."""
    try:
        with open('trained_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('ttp_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_mitigations_database():
    """Load MITRE mitigations mapped to techniques."""
    # This would be loaded from MITRE ATT&CK mitigations
    # For now, returning a sample structure
    return {
        "T1566": {
            "name": "Phishing",
            "mitigations": [
                "M1049: Antivirus/Antimalware - Use anti-virus/anti-malware software",
                "M1031: Network Intrusion Prevention - Use email gateway solutions",
                "M1017: User Training - Train users to identify phishing attempts",
                "M1021: Restrict Web-Based Content - Block suspicious email attachments"
            ]
        },
        "T1059": {
            "name": "Command and Scripting Interpreter",
            "mitigations": [
                "M1038: Execution Prevention - Use application control solutions",
                "M1049: Antivirus/Antimalware - Use behavior-based detection",
                "M1026: Privileged Account Management - Limit PowerShell to admins only",
                "M1042: Disable or Remove Feature or Program - Disable unnecessary scripting engines"
            ]
        },
        "T1071": {
            "name": "Application Layer Protocol",
            "mitigations": [
                "M1031: Network Intrusion Prevention - Deploy network IDS/IPS",
                "M1037: Filter Network Traffic - Use web proxies and SSL inspection",
                "M1020: SSL/TLS Inspection - Inspect encrypted traffic"
            ]
        },
        "T1003": {
            "name": "OS Credential Dumping",
            "mitigations": [
                "M1028: Operating System Configuration - Enable Credential Guard",
                "M1043: Credential Access Protection - Use LSASS protection",
                "M1026: Privileged Account Management - Limit admin privileges",
                "M1027: Password Policies - Enforce strong password policies"
            ]
        },
        "T1082": {
            "name": "System Information Discovery",
            "mitigations": [
                "M1038: Execution Prevention - Limit process execution",
                "M1018: User Account Management - Restrict user permissions"
            ]
        },
        # Add more as needed
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_adversary_by_alias(alias_input, adversaries):
    """Find adversary by alias name."""
    alias_lower = alias_input.lower().strip()
    
    matches = []
    for adv in adversaries:
        # Check primary name
        if alias_lower in adv.get('mitre_attack_name', '').lower():
            matches.append(adv)
            continue
        
        # Check aliases
        aliases = adv.get('aliases', [])
        for alias in aliases:
            if alias_lower in alias.lower():
                matches.append(adv)
                break
    
    return matches

def find_adversary_by_ttps(ttp_input, adversaries, vectorizer, model, threshold=0.70):
    """Find adversaries with similar TTPs."""
    
    # Parse TTPs
    ttps = [line.strip() for line in ttp_input.split('\n') if line.strip()]
    ttp_string = ' '.join(ttps)
    
    # Vectorize input
    X_input = vectorizer.transform([ttp_string]).toarray()
    
    # Calculate similarities with all adversaries
    similarities = []
    for adv in adversaries:
        adv_ttps = ' '.join(adv.get('mitre_attack_ttps', []))
        if not adv_ttps.strip():
            continue
        
        X_adv = vectorizer.transform([adv_ttps]).toarray()
        similarity = 1 - cosine(X_input[0], X_adv[0])
        
        if similarity >= threshold:
            similarities.append({
                'adversary': adv,
                'similarity': similarity
            })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities

def extract_technique_ids(ttp_list):
    """Extract technique IDs (T1234.001) from TTP strings."""
    import re
    technique_ids = []
    
    for ttp in ttp_list:
        # Match T1234 or T1234.001 pattern
        matches = re.findall(r'T\d+(?:\.\d+)?', ttp)
        technique_ids.extend(matches)
    
    # Get parent techniques (T1234 from T1234.001)
    parent_techniques = []
    for tid in technique_ids:
        if '.' in tid:
            parent = tid.split('.')[0]
            parent_techniques.append(parent)
        else:
            parent_techniques.append(tid)
    
    return list(set(parent_techniques))

def get_defensive_actions(technique_ids, mitigations_db):
    """Get defensive actions for given techniques."""
    all_mitigations = []
    
    for tid in technique_ids:
        if tid in mitigations_db:
            tech_info = mitigations_db[tid]
            all_mitigations.append({
                'technique': f"{tid}: {tech_info['name']}",
                'mitigations': tech_info['mitigations']
            })
    
    return all_mitigations

def create_diamond_model(adversary_name, victim_sector, infrastructure, capability_summary):
    """Create Diamond Model visualization using Plotly."""
    
    fig = go.Figure()
    
    # Diamond coordinates (rotated 45 degrees)
    positions = {
        'Adversary': (0.5, 1.0),      # Top
        'Capability': (1.0, 0.5),     # Right
        'Victim': (0.5, 0.0),         # Bottom
        'Infrastructure': (0.0, 0.5)  # Left
    }
    
    # Draw diamond edges
    edge_x = [0.5, 1.0, 0.5, 0.0, 0.5]
    edge_y = [1.0, 0.5, 0.0, 0.5, 1.0]
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='#667eea', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add nodes
    labels = {
        'Adversary': f"<b>Adversary</b><br>{adversary_name}",
        'Capability': f"<b>Capability</b><br>{capability_summary}",
        'Victim': f"<b>Victim</b><br>{victim_sector}",
        'Infrastructure': f"<b>Infrastructure</b><br>{infrastructure}"
    }
    
    colors = {
        'Adversary': '#dc3545',      # Red
        'Capability': '#28a745',     # Green
        'Victim': '#ffc107',         # Yellow
        'Infrastructure': '#007bff'  # Blue
    }
    
    for node, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color=colors[node], line=dict(color='white', width=2)),
            text=labels[node],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=labels[node],
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Diamond Model of Intrusion Analysis",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<div class="main-header">🛡️ Threat Intelligence System</div>', unsafe_allow_html=True)
    
    # Load data
    adversaries = load_combined_data()
    model, vectorizer = load_model_and_vectorizer()
    mitigations_db = load_mitigations_database()
    
    if not adversaries:
        st.error("⚠️ **Data not loaded.** Please ensure you have run the data preparation scripts.")
        st.info("""
        **Required steps:**
        1. Run `ttpCategory.py` to extract TTPs
        2. Run `hierarchical_clustering.py` to cluster adversaries
        3. Run `supervised_classification.py` to train the model
        """)
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔍 Search Options")
    search_method = st.sidebar.radio(
        "Select Search Method",
        ["🏷️ Search by Alias/Name", "🔧 Search by TTPs"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success(f"✅ {len(adversaries)} adversaries loaded")
    if model and vectorizer:
        st.sidebar.success("✅ ML model loaded")
    else:
        st.sidebar.warning("⚠️ ML model not available")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Select search method
    2. Enter alias or TTPs
    3. View threat actor details
    4. Review defensive actions
    5. Analyze Diamond Model
    """)
    
    # Main content
    if search_method == "🏷️ Search by Alias/Name":
        st.markdown("## 🏷️ Search by Threat Actor Alias")
        
        st.info("💡 **Tip:** Enter any known alias or name (e.g., APT28, Fancy Bear, Sofacy)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            alias_input = st.text_input(
                "Enter Alias or Threat Actor Name",
                placeholder="e.g., APT28, Fancy Bear, Lazarus Group"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and alias_input:
            matches = find_adversary_by_alias(alias_input, adversaries)
            
            if not matches:
                st.warning(f"❌ No threat actors found matching: **{alias_input}**")
                st.info("Try a different alias or use TTP search instead.")
            else:
                st.success(f"✅ Found {len(matches)} matching threat actor(s)")
                
                for i, adv in enumerate(matches, 1):
                    display_threat_actor_details(adv, mitigations_db, i)
    
    else:  # Search by TTPs
        st.markdown("## 🔧 Search by Tactics, Techniques & Procedures (TTPs)")
        
        st.info("💡 **Tip:** Enter TTPs in MITRE format (one per line). Example: T1566.001: Phishing")
        
        # Example TTPs
        with st.expander("📝 Click to see example TTPs"):
            st.markdown("""
            **APT28-style TTPs:**
            ```
            T1566.001: Phishing: Spearphishing Attachment
            T1059.001: Command and Scripting Interpreter: PowerShell
            T1071.001: Application Layer Protocol: Web Protocols
            T1003.001: OS Credential Dumping: LSASS Memory
            T1082: System Information Discovery
            ```
            
            **Ransomware-style TTPs:**
            ```
            T1486: Data Encrypted for Impact
            T1490: Inhibit System Recovery
            T1489: Service Stop
            T1047: Windows Management Instrumentation
            T1021.001: Remote Services: Remote Desktop Protocol
            ```
            """)
        
        ttp_input = st.text_area(
            "Enter TTPs (one per line)",
            height=200,
            placeholder="T1566.001: Phishing: Spearphishing Attachment\nT1059.001: Command and Scripting Interpreter: PowerShell\nT1071.001: Application Layer Protocol: Web Protocols"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            search_button = st.button("🔍 Identify Threat Actor", type="primary", use_container_width=True)
        
        if search_button and ttp_input:
            if model and vectorizer:
                with st.spinner("🔍 Analyzing TTPs and matching threat actors..."):
                    matches = find_adversary_by_ttps(ttp_input, adversaries, vectorizer, model, threshold=0.65)
                
                if not matches:
                    st.warning("❌ No similar threat actors found with these TTPs.")
                    st.info("Try adding more TTPs or adjusting your input.")
                else:
                    st.success(f"✅ Found {len(matches)} similar threat actor(s)")
                    
                    # Display top 3 matches
                    for i, match_data in enumerate(matches[:3], 1):
                        adv = match_data['adversary']
                        similarity = match_data['similarity']
                        display_threat_actor_details(adv, mitigations_db, i, similarity=similarity)
            else:
                st.error("⚠️ ML model not available. Please train the model first.")

def display_threat_actor_details(adv, mitigations_db, rank=1, similarity=None):
    """Display comprehensive threat actor information."""
    
    st.markdown("---")
    
    # Header with rank and similarity
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🎯 Match #{rank}: {adv.get('mitre_attack_name', 'Unknown')}")
    with col2:
        if similarity:
            st.metric("Similarity Score", f"{similarity:.1%}")
    
    # Basic Information
    with st.container():
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🆔 Threat Actor ID**")
            st.markdown(f"<h3>{adv.get('mitre_attack_id', 'Unknown')}</h3>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🏷️ Primary Name**")
            st.markdown(f"<h3>{adv.get('mitre_attack_name', 'Unknown')}</h3>", unsafe_allow_html=True)
        
        with col3:
            cluster = adv.get('cluster', 'Unknown')
            st.markdown("**📊 Behavioral Cluster**")
            st.markdown(f"<h3>Cluster {cluster}</h3>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Aliases
    aliases = adv.get('aliases', [])
    if aliases:
        st.markdown("**🔤 Known Aliases:**")
        alias_badges = " ".join([f'<span class="success-badge">{alias}</span>' for alias in aliases[:10]])
        st.markdown(alias_badges, unsafe_allow_html=True)
        if len(aliases) > 10:
            st.caption(f"... and {len(aliases) - 10} more aliases")
    
    # Description
    description = adv.get('description', 'No description available.')
    with st.expander("📖 Threat Actor Description"):
        st.write(description)
    
    # TTPs
    ttps = adv.get('mitre_attack_ttps', [])
    st.markdown(f"**🔧 Known TTPs:** {len(ttps)} techniques")
    
    if ttps:
        with st.expander(f"🔍 View all {len(ttps)} TTPs"):
            # Display in columns
            cols = st.columns(2)
            for i, ttp in enumerate(ttps):
                cols[i % 2].markdown(f"• {ttp}")
    
    # Tactics
    tactics = adv.get('tactics', [])
    if tactics:
        st.markdown("**⚔️ Attack Tactics:**")
        tactic_badges = " ".join([f'<span class="warning-badge">{tactic}</span>' for tactic in tactics[:10]])
        st.markdown(tactic_badges, unsafe_allow_html=True)
    
    # Defensive Actions
    st.markdown("---")
    st.markdown("## 🛡️ Recommended Defensive Actions")
    
    technique_ids = extract_technique_ids(ttps)
    defensive_actions = get_defensive_actions(technique_ids, mitigations_db)
    
    if defensive_actions:
        for action_data in defensive_actions[:5]:  # Show top 5
            with st.expander(f"🎯 {action_data['technique']}", expanded=True):
                for mitigation in action_data['mitigations']:
                    st.markdown(f"✅ {mitigation}")
        
        if len(defensive_actions) > 5:
            st.info(f"💡 {len(defensive_actions) - 5} more techniques have available mitigations")
    else:
        st.warning("⚠️ No specific mitigations found for these TTPs. Apply general security best practices.")
    
    # General recommendations
    with st.expander("📋 General Security Recommendations"):
        st.markdown("""
        ### 🔒 Essential Security Controls
        
        1. **Network Security**
           - Deploy next-generation firewalls with IPS
           - Implement network segmentation
           - Enable SSL/TLS inspection
           - Use DNS filtering and monitoring
        
        2. **Endpoint Protection**
           - Deploy EDR (Endpoint Detection and Response)
           - Enable application whitelisting
           - Keep systems patched and updated
           - Use anti-malware with behavioral detection
        
        3. **Identity & Access Management**
           - Implement MFA (Multi-Factor Authentication)
           - Follow principle of least privilege
           - Use privileged access management (PAM)
           - Monitor authentication logs
        
        4. **Detection & Monitoring**
           - Deploy SIEM (Security Information and Event Management)
           - Enable comprehensive logging
           - Implement threat hunting program
           - Use threat intelligence feeds
        
        5. **Incident Response**
           - Maintain incident response plan
           - Conduct regular tabletop exercises
           - Establish communication protocols
           - Maintain offline backups
        """)
    
    # Diamond Model
    st.markdown("---")
    st.markdown("## 💎 Diamond Model Analysis")
    
    st.info("""
    **Diamond Model** represents the relationships between four core features: 
    Adversary, Capability, Infrastructure, and Victim.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate Diamond Model
        adversary_name = adv.get('mitre_attack_name', 'Unknown')
        
        # Extract capability summary from top TTPs
        top_ttps = ttps[:3] if len(ttps) >= 3 else ttps
        capability_summary = f"{len(ttps)} TTPs including " + ", ".join([t.split(':')[0] for t in top_ttps])
        
        # Infer likely victims and infrastructure
        tactics_list = tactics if tactics else []
        
        # Determine victim sector based on tactics
        if any(t in ['collection', 'exfiltration'] for t in tactics_list):
            victim_sector = "High-Value Targets<br>(Intellectual Property)"
        elif any(t in ['impact'] for t in tactics_list):
            victim_sector = "Critical Infrastructure"
        elif any(t in ['credential-access'] for t in tactics_list):
            victim_sector = "Enterprise Networks"
        else:
            victim_sector = "Various Sectors"
        
        # Determine infrastructure based on TTPs
        infrastructure = "C2 Servers, Domains,<br>Compromised Infrastructure"
        
        fig = create_diamond_model(adversary_name, victim_sector, infrastructure, capability_summary)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Diamond Model Components")
        
        st.markdown('<div class="diamond-node">👤 Adversary</div>', unsafe_allow_html=True)
        st.caption(f"{adversary_name}")
        
        st.markdown('<div class="diamond-node">⚙️ Capability</div>', unsafe_allow_html=True)
        st.caption(f"{len(ttps)} documented TTPs")
        
        st.markdown('<div class="diamond-node">🎯 Victim</div>', unsafe_allow_html=True)
        st.caption("Typical targets of this actor")
        
        st.markdown('<div class="diamond-node">🌐 Infrastructure</div>', unsafe_allow_html=True)
        st.caption("Command & Control systems")
        
        st.markdown("---")
        
        st.markdown("### 📊 Threat Level")
        threat_level = len(ttps)
        if threat_level > 40:
            st.markdown('<span class="danger-badge">🔴 CRITICAL</span>', unsafe_allow_html=True)
            st.caption("Highly sophisticated adversary")
        elif threat_level > 20:
            st.markdown('<span class="warning-badge">🟡 HIGH</span>', unsafe_allow_html=True)
            st.caption("Advanced persistent threat")
        else:
            st.markdown('<span class="success-badge">🟢 MODERATE</span>', unsafe_allow_html=True)
            st.caption("Standard threat actor")
    
    # Additional Intelligence
    with st.expander("🔬 Additional Threat Intelligence"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Data Freshness**")
            st.info("Based on MITRE ATT&CK v14 (Latest)")
            
            st.markdown("**🔗 External References**")
            st.markdown(f"- [MITRE ATT&CK Profile](https://attack.mitre.org/groups/{adv.get('mitre_attack_id', '')})")
            st.markdown("- [VirusTotal Intelligence](https://www.virustotal.com/)")
            st.markdown("- [AlienVault OTX](https://otx.alienvault.com/)")
        
        with col2:
            st.markdown("**🎯 Targeting Information**")
            st.write(f"Tactics: {len(tactics)} | Techniques: {len(ttps)}")
            
            st.markdown("**⚠️ Risk Assessment**")
            if len(ttps) > 40:
                st.error("Extreme risk - immediate action required")
            elif len(ttps) > 20:
                st.warning("High risk - enhanced monitoring recommended")
            else:
                st.info("Moderate risk - standard defenses applicable")

if __name__ == "__main__":
    main()