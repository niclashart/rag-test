"""Product filter page for filtering and matching products."""
import streamlit as st
import requests
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def get_headers():
    """Get headers (no authentication needed)."""
    return {}

def show_product_filter():
    """Show product filter page."""
    st.title("🔍 Produkt-Filter")
    st.markdown("Finden Sie Laptops nach Ihren technischen Anforderungen")
    
    # Tabs für verschiedene Funktionen
    tab1, tab2, tab3 = st.tabs(["📋 Direkte Filterung", "📄 Anforderungen", "📊 Produktübersicht"])
    
    with tab1:
        show_direct_filter()
    
    with tab2:
        show_requirements()
    
    with tab3:
        show_product_overview()

def show_direct_filter():
    """Show direct filtering interface."""
    st.subheader("Produkte nach Kriterien filtern")
    st.markdown("Geben Sie Ihre Anforderungen ein und finden Sie passende Laptops")
    
    # Filter-Eingabefelder in Spalten
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Display")
        min_brightness = st.number_input(
            "Min. Helligkeit (nits)",
            min_value=0,
            value=300,
            step=50,
            help="Mindest-Display-Helligkeit in nits"
        )
        min_ratio = st.number_input(
            "Min. Screen-to-Body Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.5,
            help="Mindest Screen-to-Body Ratio in Prozent"
        )
        display_size_min = st.number_input(
            "Min. Display-Größe (Zoll)",
            min_value=0.0,
            value=14.0,
            step=0.1,
            help="Mindest-Display-Größe in Zoll"
        )
        display_size_max = st.number_input(
            "Max. Display-Größe (Zoll)",
            min_value=0.0,
            value=16.0,
            step=0.1,
            help="Maximal-Display-Größe in Zoll"
        )
    
    with col2:
        st.markdown("### Gewicht & Abmessungen")
        max_weight = st.number_input(
            "Max. Gewicht (kg)",
            min_value=0.0,
            value=1.43,
            step=0.01,
            help="Maximales Gewicht in Kilogramm"
        )
    
    with col3:
        st.markdown("### Performance")
        min_ram = st.number_input(
            "Min. RAM (GB)",
            min_value=0,
            value=16,
            step=4,
            help="Mindest-RAM in Gigabyte"
        )
        min_storage = st.number_input(
            "Min. Speicher (TB)",
            min_value=0.0,
            value=0.5,
            step=0.1,
            help="Mindest-Speicher in Terabyte"
        )
        min_battery = st.number_input(
            "Min. Akkukapazität (Wh)",
            min_value=0.0,
            value=50.0,
            step=5.0,
            help="Mindest-Akkukapazität in Wattstunden"
        )
    
    # Such-Button
    if st.button("🔍 Produkte suchen", type="primary", use_container_width=True):
        filter_products(
            min_brightness_nits=min_brightness if min_brightness > 0 else None,
            min_screen_to_body_ratio=min_ratio if min_ratio > 0 else None,
            max_weight_kg=max_weight if max_weight > 0 else None,
            min_ram_gb=min_ram if min_ram > 0 else None,
            min_storage_tb=min_storage if min_storage > 0 else None,
            min_battery_wh=min_battery if min_battery > 0 else None,
            display_size_inch_min=display_size_min if display_size_min > 0 else None,
            display_size_inch_max=display_size_max if display_size_max > 0 else None
        )

def filter_products(
    min_brightness_nits=None,
    min_screen_to_body_ratio=None,
    max_weight_kg=None,
    min_ram_gb=None,
    min_storage_tb=None,
    min_battery_wh=None,
    display_size_inch_min=None,
    display_size_inch_max=None
):
    """Filter products and display results."""
    try:
        # Build query parameters
        params = {}
        if min_brightness_nits:
            params["min_brightness_nits"] = min_brightness_nits
        if min_screen_to_body_ratio:
            params["min_screen_to_body_ratio"] = min_screen_to_body_ratio
        if max_weight_kg:
            params["max_weight_kg"] = max_weight_kg
        if min_ram_gb:
            params["min_ram_gb"] = min_ram_gb
        if min_storage_tb:
            params["min_storage_tb"] = min_storage_tb
        if min_battery_wh:
            params["min_battery_wh"] = min_battery_wh
        if display_size_inch_min:
            params["display_size_inch_min"] = display_size_inch_min
        if display_size_inch_max:
            params["display_size_inch_max"] = display_size_inch_max
        
        if not params:
            st.warning("Bitte geben Sie mindestens ein Kriterium ein.")
            return
        
        with st.spinner("Suche passende Produkte..."):
            response = requests.get(
                f"{API_BASE_URL}/api/products/filter",
                params=params,
                headers=get_headers(),
                timeout=30
            )
        
        if response.status_code == 200:
            products = response.json()
            
            if len(products) == 0:
                st.info("Keine Produkte gefunden, die alle Kriterien erfüllen.")
            else:
                st.success(f"✅ {len(products)} Produkt(e) gefunden!")
                
                # Display products in expandable cards
                for idx, product in enumerate(products, 1):
                    with st.expander(f"💻 {product.get('product_name', 'Unbekanntes Produkt')}", expanded=(idx == 1)):
                        display_product_details(product)
        else:
            error_msg = response.json().get('detail', 'Unbekannter Fehler')
            st.error(f"Fehler: {error_msg}")
    except requests.exceptions.ConnectionError:
        st.error("Verbindungsfehler: Backend nicht erreichbar. Bitte starten Sie das Backend.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def display_product_details(product):
    """Display detailed product information."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Display")
        if product.get('display_brightness_nits'):
            st.write(f"**Helligkeit:** {product['display_brightness_nits']} nits")
        if product.get('screen_to_body_ratio'):
            st.write(f"**Screen-to-Body Ratio:** {product['screen_to_body_ratio']}%")
        if product.get('display_size_inch'):
            st.write(f"**Größe:** {product['display_size_inch']}\"")
        if product.get('display_resolution'):
            st.write(f"**Auflösung:** {product['display_resolution']}")
        
        st.markdown("#### Gewicht")
        if product.get('weight_kg'):
            st.write(f"**Gewicht:** {product['weight_kg']} kg")
    
    with col2:
        st.markdown("#### Performance")
        if product.get('ram_max_gb'):
            st.write(f"**RAM:** {product['ram_max_gb']} GB")
        if product.get('storage_max_tb'):
            st.write(f"**Speicher:** {product['storage_max_tb']} TB")
        if product.get('battery_wh'):
            st.write(f"**Akku:** {product['battery_wh']} Wh")
        
        st.markdown("#### Informationen")
        if product.get('brand'):
            st.write(f"**Marke:** {product['brand']}")
        st.write(f"**Dokument ID:** {product['document_id']}")

def show_requirements():
    """Show requirements management."""
    st.subheader("Ausschreibungsanforderungen")
    st.markdown("Verwalten Sie Ausschreibungsanforderungen und finden Sie passende Produkte")
    
    try:
        # Load requirements
        response = requests.get(
            f"{API_BASE_URL}/api/products/requirements",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            requirements = response.json()
            
            if len(requirements) == 0:
                st.info("Noch keine Anforderungen vorhanden. Laden Sie ein Ausschreibungsdokument hoch und indizieren Sie es.")
            else:
                st.success(f"✅ {len(requirements)} Anforderung(en) gefunden")
                
                # Display requirements
                for req in requirements:
                    # Zähle wie viele strukturierte Anforderungen vorhanden sind
                    structured_count = sum([
                        1 if req.get('min_display_brightness_nits') else 0,
                        1 if req.get('min_screen_to_body_ratio') else 0,
                        1 if req.get('display_size_inch_min') else 0,
                        1 if req.get('display_size_inch_max') else 0,
                        1 if req.get('max_weight_kg') else 0,
                        1 if req.get('min_ram_gb') else 0,
                        1 if req.get('min_storage_tb') else 0,
                        1 if req.get('min_battery_wh') else 0,
                    ])
                    
                    additional_count = len(req.get('additional_requirements', {})) if req.get('additional_requirements') else 0
                    total_count = structured_count + additional_count
                    
                    with st.expander(f"📄 Anforderung #{req['id']} (Dokument ID: {req['document_id']}) - {total_count} Kriterien", expanded=False):
                        # Debug-Info
                        if total_count == 0:
                            st.error("⚠️ **WARNUNG:** Keine Anforderungen wurden aus diesem Dokument extrahiert!")
                            st.info("💡 **Mögliche Ursachen:**\n"
                                   "- Das Dokument wurde nicht korrekt indiziert\n"
                                   "- Das Dokument wurde nicht als 'Anforderungsdokument' erkannt\n"
                                   "- Die Extraktion ist fehlgeschlagen\n\n"
                                   "**Lösung:** Dokument neu indizieren oder prüfen ob es ein Anforderungsdokument ist.")
                        else:
                            st.success(f"✅ {structured_count} strukturierte + {additional_count} zusätzliche Anforderungen extrahiert")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Display-Anforderungen")
                            has_display = False
                            if req.get('min_display_brightness_nits'):
                                st.write(f"**Min. Helligkeit:** {req['min_display_brightness_nits']} nits")
                                has_display = True
                            if req.get('min_screen_to_body_ratio'):
                                st.write(f"**Min. Screen-to-Body Ratio:** {req['min_screen_to_body_ratio']}%")
                                has_display = True
                            if req.get('display_size_inch_min'):
                                st.write(f"**Min. Display-Größe:** {req['display_size_inch_min']}\"")
                                has_display = True
                            if req.get('display_size_inch_max'):
                                st.write(f"**Max. Display-Größe:** {req['display_size_inch_max']}\"")
                                has_display = True
                            if not has_display:
                                st.caption("*Keine Display-Anforderungen*")
                            
                            st.markdown("#### Gewicht")
                            if req.get('max_weight_kg'):
                                st.write(f"**Max. Gewicht:** {req['max_weight_kg']} kg")
                            else:
                                st.caption("*Keine Gewichts-Anforderung*")
                        
                        with col2:
                            st.markdown("#### Performance-Anforderungen")
                            has_perf = False
                            if req.get('min_ram_gb'):
                                st.write(f"**Min. RAM:** {req['min_ram_gb']} GB")
                                has_perf = True
                            if req.get('min_storage_tb'):
                                st.write(f"**Min. Speicher:** {req['min_storage_tb']} TB")
                                has_perf = True
                            if req.get('min_battery_wh'):
                                st.write(f"**Min. Akku:** {req['min_battery_wh']} Wh")
                                has_perf = True
                            if req.get('min_battery_runtime_hours'):
                                st.write(f"**Min. Akku-Laufzeit:** {req['min_battery_runtime_hours']} Stunden")
                                has_perf = True
                            if not has_perf:
                                st.caption("*Keine Performance-Anforderungen*")
                        
                        # Zeige alle zusätzlichen Anforderungen in erweiterter Ansicht
                        if req.get('additional_requirements'):
                            st.markdown("#### 📋 Alle Anforderungen im Detail")
                            
                            # Debug: Zeige rohe Daten
                            with st.expander("🔍 Debug: Rohe additional_requirements Daten", expanded=False):
                                st.json(req.get('additional_requirements'))
                            
                            # Dynamische Gruppierung basierend auf den tatsächlich extrahierten Kategorien
                            # Erkenne Hauptkategorien aus den Schlüsselnamen
                            categories = {}
                            
                            def get_category(key):
                                """Ermittelt die Kategorie basierend auf dem Schlüsselnamen."""
                                key_lower = key.lower()
                                # Prüfe auf verschachtelte Strukturen (z.B. "display.details")
                                if '.' in key:
                                    return key.split('.')[0]
                                
                                # Gruppiere nach Schlüsselworten
                                if any(x in key_lower for x in ['display', 'bildschirm', 'screen']):
                                    return "🖥️ Display"
                                elif any(x in key_lower for x in ['processor', 'prozessor', 'cpu']):
                                    return "⚙️ Prozessor"
                                elif any(x in key_lower for x in ['ram', 'memory', 'speicher', 'arbeitsspeicher']):
                                    return "💾 RAM/Speicher"
                                elif any(x in key_lower for x in ['storage', 'ssd', 'hdd', 'nvme']):
                                    return "💿 Storage/SSD"
                                elif any(x in key_lower for x in ['graphics', 'gpu', 'grafik']):
                                    return "🎮 Grafik"
                                elif any(x in key_lower for x in ['port', 'hdmi', 'thunderbolt', 'usb', 'ethernet', 'audio', 'anschluss']):
                                    return "🔌 Anschlüsse/Ports"
                                elif any(x in key_lower for x in ['connectivity', 'wifi', 'bluetooth', 'wake', 'pxe', 'miracast', 'konnektivität']):
                                    return "📡 Konnektivität"
                                elif any(x in key_lower for x in ['input', 'touchpad', 'keyboard', 'tastatur', 'microphone', 'mikrofon', 'eingabe']):
                                    return "⌨️ Eingabegeräte"
                                elif any(x in key_lower for x in ['camera', 'kamera', 'webcam']):
                                    return "📷 Kamera"
                                elif any(x in key_lower for x in ['sensor', 'sensor']):
                                    return "🔍 Sensoren"
                                elif any(x in key_lower for x in ['security', 'sicherheit', 'kensington', 'bios']):
                                    return "🔒 Sicherheit"
                                elif any(x in key_lower for x in ['battery', 'akku', 'batterie']):
                                    return "🔋 Akku"
                                elif any(x in key_lower for x in ['operating', 'os', 'betriebssystem', 'windows']):
                                    return "💻 Betriebssystem"
                                elif any(x in key_lower for x in ['power', 'netzteil', 'adapter']):
                                    return "⚡ Netzteil"
                                elif any(x in key_lower for x in ['warranty', 'garantie']):
                                    return "🛡️ Garantie"
                                else:
                                    return "📝 Weitere Anforderungen"
                            
                            # Gruppiere alle Anforderungen nach Kategorien
                            for key, value in req['additional_requirements'].items():
                                category = get_category(key)
                                if category not in categories:
                                    categories[category] = {}
                                categories[category][key] = value
                            
                            # Zeige alle Kategorien dynamisch
                            for category_name, items in sorted(categories.items()):
                                with st.expander(category_name, expanded=False):
                                    # Wenn es verschachtelte Objekte sind, zeige sie strukturiert
                                    for key, value in items.items():
                                        if isinstance(value, dict):
                                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                            for sub_key, sub_value in value.items():
                                                st.write(f"  - {sub_key.replace('_', ' ').title()}: {sub_value}")
                                        elif isinstance(value, list):
                                            st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(str(v) for v in value)}")
                                        else:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Match button
                        if st.button(f"🔍 Passende Produkte finden", key=f"match_{req['id']}", use_container_width=True):
                            match_requirement_to_products(req['id'])
        else:
            error_msg = response.json().get('detail', 'Unbekannter Fehler')
            st.error(f"Fehler: {error_msg}")
    except requests.exceptions.ConnectionError:
        st.error("Verbindungsfehler: Backend nicht erreichbar.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def match_requirement_to_products(requirement_id):
    """Match a requirement with products."""
    try:
        with st.spinner("Suche passende Produkte..."):
            response = requests.post(
                f"{API_BASE_URL}/api/products/match-requirement/{requirement_id}",
                headers=get_headers(),
                timeout=30
            )
        
        if response.status_code == 200:
            matches = response.json()
            
            if len(matches) == 0:
                st.info("Keine Produkte gefunden, die diese Anforderung erfüllen.")
            else:
                st.success(f"✅ {len(matches)} passende Produkt(e) gefunden!")
                
                # Sort by match score
                matches_sorted = sorted(matches, key=lambda x: x.get('match_score', 0), reverse=True)
                
                # Filter: Nur Produkte mit Score > 0 anzeigen
                valid_matches = [m for m in matches_sorted if m.get('match_score', 0) > 0]
                
                if len(valid_matches) == 0:
                    st.warning("⚠️ Keine Produkte mit Match-Score > 0 gefunden. Alle Produkte haben Score 0.00.")
                    st.info("💡 **Mögliche Ursachen:**\n"
                           "- Keine Anforderungen wurden aus dem Dokument extrahiert\n"
                           "- Die extrahierten Anforderungen stimmen nicht mit den Produktspezifikationen überein\n"
                           "- Bitte überprüfen Sie die extrahierten Anforderungen oben")
                    
                    # Zeige Debug-Info: Warum Score 0?
                    with st.expander("🔍 Debug-Informationen", expanded=False):
                        st.write(f"**Anzahl Matches:** {len(matches_sorted)}")
                        st.write(f"**Anzahl mit Score > 0:** {len(valid_matches)}")
                        if len(matches_sorted) > 0:
                            st.write(f"**Durchschnittlicher Score:** {sum(m.get('match_score', 0) for m in matches_sorted) / len(matches_sorted):.2f}")
                            st.write(f"**Min Score:** {min(m.get('match_score', 0) for m in matches_sorted):.2f}")
                            st.write(f"**Max Score:** {max(m.get('match_score', 0) for m in matches_sorted):.2f}")
                else:
                    st.info(f"📊 Zeige {len(valid_matches)} von {len(matches_sorted)} Produkten (Score > 0)")
                
                # Display matches (nur die mit Score > 0)
                for idx, match in enumerate(valid_matches, 1):
                    match_score = match.get('match_score', 0)
                    score_color = "🟢" if match_score >= 1.0 else "🟡" if match_score >= 0.8 else "🔴"
                    
                    with st.expander(
                        f"{score_color} {match.get('product_name', 'Unbekannt')} (Match: {match_score:.2f})",
                        expanded=(idx == 1)
                    ):
                        display_product_details(match)
        else:
            error_msg = response.json().get('detail', 'Unbekannter Fehler')
            st.error(f"Fehler: {error_msg}")
    except requests.exceptions.ConnectionError:
        st.error("Verbindungsfehler: Backend nicht erreichbar.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def show_product_overview():
    """Show all products overview."""
    st.subheader("Alle Produkte")
    st.markdown("Übersicht aller indizierten Produkte")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/products/list",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            products = response.json()
            
            if len(products) == 0:
                st.info("Noch keine Produkte vorhanden. Laden Sie Produktdatenblätter hoch und indizieren Sie sie.")
            else:
                st.success(f"✅ {len(products)} Produkt(e) gefunden")
                
                # Create DataFrame for better display
                df_data = []
                for product in products:
                    df_data.append({
                        "Produktname": product.get('product_name', 'N/A'),
                        "Marke": product.get('brand', 'N/A'),
                        "Helligkeit (nits)": product.get('display_brightness_nits', 'N/A'),
                        "Screen-to-Body (%)": product.get('screen_to_body_ratio', 'N/A'),
                        "Gewicht (kg)": product.get('weight_kg', 'N/A'),
                        "RAM (GB)": product.get('ram_max_gb', 'N/A'),
                        "Speicher (TB)": product.get('storage_max_tb', 'N/A'),
                        "Akku (Wh)": product.get('battery_wh', 'N/A'),
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Also show expandable cards
                st.divider()
                st.markdown("### Detaillierte Ansicht")
                for product in products:
                    with st.expander(f"💻 {product.get('product_name', 'Unbekanntes Produkt')}", expanded=False):
                        display_product_details(product)
        else:
            error_msg = response.json().get('detail', 'Unbekannter Fehler')
            st.error(f"Fehler: {error_msg}")
    except requests.exceptions.ConnectionError:
        st.error("Verbindungsfehler: Backend nicht erreichbar.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")
