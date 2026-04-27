"""Specification extractor for product specs and requirements."""
from typing import Dict, Optional
import json
from dotenv import load_dotenv
from logging_config.logger import get_logger
from src.llm import create_chat_llm

load_dotenv()

logger = get_logger(__name__)


class SpecExtractor:
    """Extrahiert strukturierte Spezifikationen aus Dokumenten."""
    
    def __init__(self):
        self.llm = create_chat_llm(
            temperature=0,
            max_retries=3,
            request_timeout=180  # Erhöht für umfangreichere Extraktion
        )
    
    def detect_document_type(self, text: str, filename: str) -> str:
        """Erkennt ob es sich um ein Produktdatenblatt oder eine Ausschreibung handelt."""
        text_lower = text.lower()[:5000]  # Erste 5000 Zeichen analysieren
        
        # Indikatoren für Ausschreibung
        requirement_keywords = [
            "mindestanforderung", "mindestausstattung", "ausschreibung",
            "anforderung", "anforderungen", "ausschlußkriterium",
            "mindestens", "mind.", "min.", "maximal", "max.", "höchstens",
            "basisgerät", "notebook", "basis-", "mindest-"
        ]
        
        # Indikatoren für Produktdatenblatt
        product_keywords = [
            "produktspezifikation", "technische daten", "specification",
            "thinkpad", "zbook", "ideapad", "modell", "series",
            "product specification", "technical specifications"
        ]
        
        requirement_score = sum(1 for kw in requirement_keywords if kw in text_lower)
        product_score = sum(1 for kw in product_keywords if kw in text_lower)
        
        # Wenn Dateiname "spec" enthält, ist es wahrscheinlich ein Produktdatenblatt
        if "spec" in filename.lower() and "requirement" not in filename.lower():
            product_score += 2
        
        if requirement_score > product_score:
            logger.info(f"Document type detected: requirement (score: {requirement_score} vs {product_score})")
            return "requirement"
        else:
            logger.info(f"Document type detected: product (score: {product_score} vs {requirement_score})")
            return "product"
    
    def extract_product_specs(self, text: str, filename: str) -> Dict:
        """Extrahiert Produktspezifikationen aus einem Produktdatenblatt."""
        
        # Erhöhe Text-Limit für vollständige Analyse (erste 50000 Zeichen)
        relevant_text = text[:50000]
        
        prompt = f"""Analysiere das folgende Produktdatenblatt GRÜNDLICH und extrahiere ALLE technischen Spezifikationen.

Dateiname: {filename}

Dokument:
{relevant_text}

KRITISCH WICHTIG - VOLLSTÄNDIGE EXTRAKTION:
1. Analysiere die DOKUMENTSTRUKTUR und identifiziere ALLE Spezifikationskategorien
2. Extrahiere ALLE technischen Details - NICHTS übersehen!
3. Erstelle eine FLEXIBLE JSON-Struktur basierend auf den tatsächlich vorhandenen Kategorien im Dokument
4. Verwende die TERMINOLOGIE aus dem Dokument

ANALYSESCHRITTE:

SCHRITT 1: Dokumentstruktur analysieren
- Identifiziere alle Abschnitte, Überschriften, Tabellen und Listen
- Erkenne die verwendete Terminologie und Struktur des Dokuments
- Beachte spezifische Begriffe, die im Dokument verwendet werden

SCHRITT 2: Alle Spezifikationen extrahieren
- Numerische Werte: Extrahiere als Zahlen (z.B. 300, 1.43, 16)
- Text-Werte: Extrahiere vollständig mit allen Details aus dem Dokument
- Einheiten konvertieren wenn nötig:
  * lbs zu kg: 1 lb = 0.453592 kg
  * cm zu Zoll: 1 Zoll = 2.54 cm
  * GB zu TB: 1 TB = 1000 GB

SCHRITT 3: JSON-Struktur erstellen
- Erstelle eine hierarchische oder flache Struktur, die der Dokumentstruktur entspricht
- Verwende aussagekräftige Schlüsselnamen basierend auf dem Dokument
- Verwende verschachtelte Objekte wenn sinnvoll, sonst flache Struktur

WICHTIGE REGELN:
- Verwende NUR Informationen, die explizit im Dokument stehen
- Wenn ein Wert nicht gefunden wird, setze null
- Für screen_to_body_ratio: Suche nach Prozentangaben (z.B. "85%", "87.5%")
- Für display_brightness_nits: Suche nach "nits" oder "cd/m²" Angaben
- Für weight_kg: Konvertiere lbs zu kg wenn nötig (1 lb = 0.453592 kg)
- Extrahiere ALLE Details: Prozessor, RAM, Storage, Display, Ports, Konnektivität, Kamera, Sensoren, etc.

STRUKTURIERTE FELDER (für schnelle Filterung):
Für häufig vorkommende Kriterien verwende diese Schlüssel (wenn im Dokument vorhanden):
- product_name: Vollständiger Produktname (z.B. "ThinkPad E14 Gen 6 AMD")
- brand: Hersteller (Lenovo, HP, etc.)
- display_brightness_nits: Display-Helligkeit in nits (nur Zahl, z.B. 300)
- screen_to_body_ratio: Screen-to-Body Ratio in Prozent (nur Zahl, z.B. 87.5)
- weight_kg: Gewicht in kg (nur Zahl, z.B. 1.43)
- ram_max_gb: Maximaler RAM in GB (nur Zahl)
- storage_max_tb: Maximaler Speicher in TB (nur Zahl)
- battery_wh: Akkukapazität in Wh (nur Zahl)
- display_size_inch: Display-Größe in Zoll (nur Zahl, z.B. 14.0)
- display_resolution: Display-Auflösung (z.B. "1920x1200")

ABER: Erstelle auch eigene Kategorien für ALLE anderen Spezifikationen, die im Dokument stehen!

BEISPIEL-STRUKTUR (anpassen an tatsächliches Dokument):
Das Dokument könnte so strukturiert sein:
{{
  "product_name": "ThinkPad E14 Gen 6 AMD",
  "brand": "Lenovo",
  "display_brightness_nits": 300,
  "screen_to_body_ratio": 87.5,
  "weight_kg": 1.43,
  "ram_max_gb": 64,
  "storage_max_tb": 2.0,
  "battery_wh": 57.0,
  "display_size_inch": 14.0,
  "display_resolution": "1920x1200",
  "prozessor": {{
    "modell": "AMD Ryzen 7 8840HS",
    "kerne": 8,
    "taktfrequenz": "3.3 GHz"
  }},
  "display": {{
    "details": "14\" WUXGA (1920x1200), IPS, 300 nits, Anti-Glare",
    "typ": "IPS",
    "touch": false
  }},
  "ram": {{
    "typ": "DDR5",
    "slots": 2,
    "aufrüstbar": true
  }},
  "speicher": {{
    "typ": "NVMe SSD",
    "interface": "PCIe Gen4x4",
    "aufrüstbar": true
  }},
  "grafik": {{
    "typ": "Integriert",
    "modell": "AMD Radeon Graphics"
  }},
  "anschlüsse": {{
    "usb_c": "2x USB-C 3.2 Gen 2",
    "usb_a": "2x USB-A 3.2 Gen 1",
    "hdmi": "1x HDMI 2.1",
    "ethernet": "RJ45",
    "audio": "3.5mm Kopfhörer/Mikrofon"
  }},
  "konnektivität": {{
    "wifi": "Wi-Fi 6E",
    "bluetooth": "Bluetooth 5.1",
    "wwan": "Optional"
  }},
  "kamera": {{
    "auflösung": "1080p",
    "abdeckung": true
  }},
  "betriebssystem": {{
    "typ": "Windows 11 Pro",
    "architektur": "64-bit"
  }}
}}

WICHTIG: 
- Passe die Struktur AN DAS TATSÄCHLICHE DOKUMENT an
- Wenn das Dokument andere Kategorien verwendet, verwende diese
- Wenn das Dokument andere Begriffe verwendet, verwende diese
- Erstelle eine Struktur, die ALLE Spezifikationen aus dem Dokument erfasst
- Verwende verschachtelte Objekte oder flache Struktur - je nachdem was besser passt

Antworte NUR mit einem validen JSON-Objekt. Verwende null für fehlende Werte. Verwende die Terminologie aus dem Dokument."""

        if self.llm is None:
            logger.warning("Cannot extract product specs: no LLM API key configured")
            return {}

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Entferne mögliche Markdown-Code-Blöcke
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            specs = json.loads(content)
            
            # Definiere welche Felder für schnelle Filterung separat gespeichert werden
            structured_fields = {
                "product_name", "brand", "display_brightness_nits", "screen_to_body_ratio",
                "weight_kg", "ram_max_gb", "storage_max_tb", "battery_wh",
                "display_size_inch", "display_resolution"
            }
            
            # Alle extrahierten Daten bleiben in specs erhalten (werden als raw_specs gespeichert)
            # Die strukturierten Felder werden zusätzlich separat gespeichert für schnelle Filterung
            
            logger.info(f"Extracted product specs: {specs.get('product_name', 'Unknown')}, "
                        f"total_fields={len(specs)}")
            return specs
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error extracting product specs: {e}\nResponse (first 500 chars): {content[:500]}")
            # Versuche JSON aus dem Text zu extrahieren (manchmal gibt LLM Text vor/nach JSON aus)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    specs = json.loads(json_match.group())
                    logger.info("JSON erfolgreich aus Response-Text extrahiert (Fallback)")
                    return specs
                except json.JSONDecodeError:
                    pass
            logger.error("JSON-Extraktion fehlgeschlagen - leeres Dict wird zurückgegeben")
            return {}
        except Exception as e:
            logger.error(f"Error extracting product specs: {e}", exc_info=True)
            return {}
    
    def extract_requirements(self, text: str, filename: str) -> Dict:
        """Extrahiert Anforderungen aus einem Ausschreibungsdokument."""
        
        # Erhöhe Text-Limit für vollständige Analyse (erste 50000 Zeichen)
        relevant_text = text[:50000]
        
        prompt = f"""Analysiere das folgende Ausschreibungsdokument GRÜNDLICH und extrahiere ausschließlich datenblattfähig prüfbare Laptop/PC-Konfigurationsmerkmale.

Dateiname: {filename}

Dokument:
{relevant_text}

KRITISCH WICHTIG - FOKUS AUF GERÄTEKONFIGURATION:
1. Analysiere die DOKUMENTSTRUKTUR und identifiziere Notebook-/Laptop-/PC-Gruppen und deren Mindestausstattung
2. Extrahiere nur technische Geräteanforderungen, die später gegen Produktdatenblätter abgeglichen werden können
3. Erstelle eine FLEXIBLE JSON-Struktur basierend auf den tatsächlich vorhandenen Kategorien im Dokument
4. Verwende die TERMINOLOGIE aus dem Dokument (nicht vordefinierte Begriffe)

NICHT EXTRAHIEREN:
- Mengen, Preispositions-Mengen, Preisberechnung, Los-/Positionsnummern ohne technisches Merkmal
- Vergabe-/Formularhinweise, Ansprechpartner, Liefer-/Vertragsbedingungen, Bewertungslogik, Zuschlagskriterien
- allgemeine Hinweise wie "die angegebenen Mengen dienen ausschließlich der Berechnung des Preises"
- Service-/Support-/Reklamationskonzepte, außer wenn direkt als produktbezogene Eigenschaft formuliert (z.B. Garantiedauer oder Einbehaltung defekter Datenträger)

ANALYSESCHRITTE:

SCHRITT 1: Dokumentstruktur analysieren
- Identifiziere alle Abschnitte, Überschriften, Nummerierungen, Tabellen und Listen
- Erkenne die verwendete Terminologie und Struktur des Dokuments
- Beachte spezifische Begriffe, die im Dokument verwendet werden

SCHRITT 2: Kategorien identifizieren
- Erstelle Kategorien nur für datenblattfähige Geräte-Konfigurationen (z.B. Prozessor, RAM, SSD, Display, Ports, Netzwerk, Kamera, Sensoren, BIOS/Sicherheit, Akku, Netzteil, Betriebssystem)
- Erstelle Kategorien basierend auf dem Dokument (z.B. wenn das Dokument "Bildschirm" sagt, verwende "Bildschirm", nicht "Display")
- Erkenne auch ungewöhnliche oder spezifische Kategorien, die nur in diesem Dokument vorkommen
- Gruppiere verwandte Anforderungen logisch

SCHRITT 3: Alle Werte extrahieren
- Numerische Werte: Extrahiere als Zahlen (z.B. 300, 1.43, 16)
- Text-Werte: Extrahiere vollständig mit allen Details aus dem Dokument
- Einheiten konvertieren wenn nötig:
  * lbs zu kg: 1 lb = 0.453592 kg
  * cm zu Zoll: 1 Zoll = 2.54 cm
  * GB zu TB: 1 TB = 1000 GB

SCHRITT 4: JSON-Struktur erstellen
- Erstelle eine hierarchische oder flache Struktur, die der Dokumentstruktur entspricht
- Verwende aussagekräftige Schlüsselnamen basierend auf dem Dokument
- Verwende verschachtelte Objekte wenn sinnvoll, sonst flache Struktur

WICHTIGE REGELN:
- Suche nach Begriffen wie "mindestens", "min.", "mind.", "maximal", "max.", "höchstens", "Ausschlußkriterium"
- Wenn "mindestens 300 nits" steht, extrahiere 300
- Wenn "maximal 1.43 kg" steht, extrahiere 1.43
- Extrahiere technische Bullet-Points auch dann, wenn sie nicht explizit als "mindestens" formuliert sind
- Wenn eine Anforderung als "Ausschlußkriterium" markiert ist, ist sie besonders wichtig
- Speichere alle Informationen so vollständig wie möglich
- Verwende die Terminologie aus dem Dokument (z.B. "Bildschirm" statt "Display", "Prozessor" statt "CPU")

OPTIONALE STRUKTURIERUNG (nur wenn im Dokument vorhanden):
Für häufig vorkommende Kriterien kannst du diese Schlüssel verwenden (wenn sie im Dokument stehen):
- min_display_brightness_nits (Zahl, wenn Helligkeit in nits erwähnt)
- min_screen_to_body_ratio (Zahl in Prozent, wenn Screen-to-Body Ratio erwähnt)
- max_weight_kg (Zahl, wenn Gewicht erwähnt)
- min_ram_gb (Zahl, wenn RAM erwähnt)
- min_storage_tb (Zahl, wenn Speicher erwähnt)
- min_battery_wh (Zahl, wenn Akkukapazität erwähnt)
- min_battery_runtime_hours (Zahl, wenn Akkulaufzeit erwähnt)
- display_size_inch_min (Zahl, wenn Display-Größe erwähnt)
- warranty_years (Zahl, wenn Garantie erwähnt)

ABER: Erstelle auch eigene Kategorien für ALLE anderen Anforderungen, die im Dokument stehen!

BEISPIEL-STRUKTUR (anpassen an tatsächliches Dokument):
Das Dokument könnte so strukturiert sein:
{{
  "min_display_brightness_nits": 800,
  "max_weight_kg": 1.43,
  "min_ram_gb": 16,
  "display": {{
    "details": "35,6 cm (14\") diagonal, 16:10 WUXGA (1920x1200), LCD, UWVA, Anti-Glarte, WLED + Low Blue Light, 800nits, Low Power, sRGB 100%, kein touch",
    "resolution": "1920x1200",
    "type": "LCD, UWVA",
    "features": ["Anti-Glarte", "Low Blue Light", "sRGB 100%"],
    "touch": false
  }},
  "prozessor": {{
    "anforderung": "Intel Core Ultra5 235U",
    "optionen": "auch optionale Prozessoren (Intel vPro Zertifizierung)"
  }},
  "ram": {{
    "details": "16GB DDR5, 2 SODIMM Slots verfügbar, aufrüstbar/austauschbar durch den Kunden ohne Garantieverlust",
    "typ": "DDR5",
    "aufrüstbar": "durch den Kunden ohne Garantieverlust"
  }},
  "speicher": {{
    "anforderung": "NVMe mind. PCIe Gen4x4 500 GB",
    "aufrüstbar": "durch den Kunden ohne Garantieverlust"
  }},
  "grafik": {{
    "anforderung": "Integriert (Intel Graphics) oder entsprechend"
  }},
  "anschlüsse": {{
    "hdmi": "1x Monitorausgang mind. HDMI 2.1",
    "thunderbolt": "2x Thunderbolt 4",
    "usb": "1x USB-A 5Gbps",
    "ethernet": "nativer RJ45 Ethernet Port",
    "audio": "Kopfhörer/Mikrofon Kombo-Anschluss (3.5mm)"
  }},
  "konnektivität": {{
    "wifi": "Wi-Fi 7",
    "bluetooth": "Bluetooth 5.4",
    "wake_on_lan": "Wake on LAN/WLAN",
    "pxe": "PXE über LAN/WLAN",
    "miracast": "Nativer Miracast Support"
  }},
  "eingabegeräte": {{
    "touchpad": "Multitouch gesture support, Clickpad",
    "tastatur": "spritzwassergeschützt, hintergrundbeleuchtet, deutsches Tastaturlayout",
    "mikrofon": "Integriertes Mikrofon"
  }},
  "kamera": {{
    "anforderung": "Integrierte Webcam mit 5MP und Kameraabdeckung",
    "auflösung": "5MP",
    "abdeckung": true
  }},
  "sensoren": {{
    "thermal": true,
    "ambient_light": true,
    "hall_effekt": true
  }},
  "sicherheit": {{
    "kensington_lock": true,
    "bios_features": "MAC-Passthrough, BIOS-Update über Netzwerk"
  }},
  "akku": {{
    "min_kapazität_wh": 50.0,
    "min_laufzeit_stunden": 6,
    "austauschbar": "durch den Kunden ohne Garantieverlust"
  }},
  "betriebssystem": {{
    "anforderung": "Windows® 11 Professional 64 Bit",
    "version": "Windows 11 Professional",
    "architektur": "64 Bit"
  }},
  "netzteil": {{
    "typ": "Standard USB Type-C",
    "original": true
  }},
  "garantie": {{
    "dauer_jahre": 3,
    "details": "Dauer 3 Jahre (Mindestanforderung)"
  }}
}}

WICHTIG: 
- Passe die Struktur AN DAS TATSÄCHLICHE DOKUMENT an
- Wenn das Dokument andere Kategorien verwendet, verwende diese
- Wenn das Dokument andere Begriffe verwendet (z.B. "Bildschirm" statt "Display"), verwende diese
- Erstelle eine Struktur, die ALLE Anforderungen aus dem Dokument erfasst
- Verwende verschachtelte Objekte oder flache Struktur - je nachdem was besser passt
- Wenn das Dokument z.B. "Mindestausstattung Basisgerät Notebook 14" als Überschrift hat, erstelle eine entsprechende Kategorie

Antworte NUR mit einem validen JSON-Objekt. Verwende null für fehlende Werte. Verwende die Terminologie aus dem Dokument."""

        if self.llm is None:
            logger.warning("Cannot extract requirements: no LLM API key configured")
            return {}

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Entferne mögliche Markdown-Code-Blöcke
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            requirements = json.loads(content)
            
            # Definiere welche Felder für schnelle Filterung separat gespeichert werden
            # (nur häufig verwendete numerische Werte)
            structured_fields = {
                "min_display_brightness_nits", "min_screen_to_body_ratio", "max_weight_kg",
                "min_ram_gb", "min_storage_tb", "min_battery_wh", "min_battery_runtime_hours",
                "display_size_inch_min", "display_size_inch_max", "warranty_years"
            }
            
            # Trenne strukturierte Felder (für schnelle Filterung) von flexiblen Daten
            structured_data = {}
            flexible_data = {}
            
            for key, value in requirements.items():
                # Überspringe Metadaten
                if key == "document_type":
                    continue
                
                # Speichere strukturierte numerische Felder separat
                if key in structured_fields and value is not None:
                    structured_data[key] = value
                # Alle anderen Felder gehen in flexible Struktur
                elif value is not None:
                    flexible_data[key] = value
            
            # Kombiniere strukturierte und flexible Daten
            result = {}
            result.update(structured_data)  # Strukturierte Felder zuerst
            
            # Alle flexiblen Daten in additional_requirements speichern
            # Dies enthält ALLE Kategorien, die dynamisch aus dem Dokument extrahiert wurden
            if flexible_data:
                result["additional_requirements"] = flexible_data
            
            logger.info(f"Extracted requirements: structured_fields={len(structured_data)}, "
                       f"flexible_categories={len(flexible_data)}, "
                       f"total_categories={len(structured_data) + len(flexible_data)}")
            
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error extracting requirements: {e}\nResponse (first 500 chars): {content[:500]}")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    requirements = json.loads(json_match.group())
                    logger.info("JSON erfolgreich aus Response-Text extrahiert (Fallback)")
                    # Weiterverarbeitung wie im try-Block
                    structured_fields = {
                        "min_display_brightness_nits", "min_screen_to_body_ratio", "max_weight_kg",
                        "min_ram_gb", "min_storage_tb", "min_battery_wh", "min_battery_runtime_hours",
                        "display_size_inch_min", "display_size_inch_max", "warranty_years"
                    }
                    structured_data = {}
                    flexible_data = {}
                    for key, value in requirements.items():
                        if key == "document_type":
                            continue
                        if key in structured_fields and value is not None:
                            structured_data[key] = value
                        elif value is not None:
                            flexible_data[key] = value
                    result = {}
                    result.update(structured_data)
                    if flexible_data:
                        result["additional_requirements"] = flexible_data
                    return result
                except json.JSONDecodeError:
                    pass
            logger.error("JSON-Extraktion fehlgeschlagen - leeres Dict wird zurückgegeben")
            return {}
        except Exception as e:
            logger.error(f"Error extracting requirements: {e}", exc_info=True)
            return {}
    
    def extract_requirements_from_pages(self, text: str, filename: str, group_name: str = "") -> Dict:
        """Extrahiert Anforderungen aus spezifischen Seiten eines Leistungsverzeichnisses.

        Args:
            text: Extracted text from the page range
            filename: Original filename
            group_name: Optional group name like "Gruppe 1: Notebook 14\""
        """
        prompt = f"""Analysiere das folgende Leistungsverzeichnis und extrahiere ausschließlich datenblattfähig prüfbare Laptop/PC-Konfigurationsmerkmale für {group_name if group_name else "das Basisgerät"}.

Dateiname: {filename}

Dokumentinhalt:
{text}

AUFGABE:
Extrahiere nur die technische Mindestausstattung, Ausschlusskriterien und optionale Geräte-Konfigurationen, die später gegen Produktdatenblätter abgeglichen werden können.
Typische relevante Bereiche sind z.B. Prozessor, RAM, SSD/Speicher, Grafik, Display, Anschlüsse, Funk/Netzwerk, Eingabegeräte, Kamera, Sensoren, Sicherheit/BIOS, Akku, Netzteil, Betriebssystem und produktbezogene Garantie/Datenträger-Regeln.

NICHT EXTRAHIEREN:
- Mengen, Preispositions-Mengen, Los-/Positionsnummern und reine Vergabe-/Formularhinweise
- Beschaffungstexte, Vertragsbedingungen, Lieferfristen, Zuschlagslogik, Bewertungspunkte, Ansprechpartner, Seitenkopf/-fuß
- allgemeine Hinweise wie "die angegebenen Mengen dienen ausschließlich der Berechnung des Preises"
- Service-/Support-/Reklamationskonzepte, außer wenn direkt als gerätebezogene Eigenschaft formuliert (z.B. konkrete Garantiedauer oder Einbehaltung defekter Datenträger)

Wenn der Text eine Gruppe wie "Gruppe 1 Notebook 14\"" und darunter "Mindestausstattung Basisgerät Notebook 14\" - Ausschlußkriterium" enthält, nutze genau diese Gruppe als Kontext und extrahiere die darunter stehenden Bullet-Points als technische Gerätemerkmale.
Strukturiere die Anforderungen als JSON.

WICHTIGE REGELN:
- Numerische Werte als Zahlen extrahieren (z.B. 300, 16, 1.43)
- Einheiten konvertieren: lbs→kg (1 lb = 0.453592 kg), GB→TB (1 TB = 1000 GB)
- "mindestens"/"mind." → Mindestwert
- "maximal"/"max." → Maximalwert
- Ausschlusskriterien besonders markieren
- Terminologie aus Dokument beibehalten
- Bewahre die vollständige Originalanforderung je Kategorie als Textfeld, damit der Abgleich nachvollziehbar bleibt
- Kurze Bullet-Points ohne "mind." sind trotzdem relevant, wenn sie konkrete Hardware-/Software-Konfigurationen nennen (z.B. "Wi-Fi 7", "2x Thunderbolt 4", "TPM 2.0 IPS 140-2 kompatibel")

STRUKTURIERTE FELDER (wenn im Dokument vorhanden):
- min_display_brightness_nits: Display-Helligkeit in nits (Zahl)
- min_screen_to_body_ratio: Screen-to-Body Ratio in % (Zahl)
- max_weight_kg: Max. Gewicht in kg (Zahl)
- min_ram_gb: Min. RAM in GB (Zahl)
- min_storage_tb: Min. Speicher in TB (Zahl)
- min_battery_wh: Min. Akkukapazität in Wh (Zahl)
- display_size_inch_min: Min. Display-Größe in Zoll (Zahl)
- display_size_inch_max: Max. Display-Größe in Zoll (Zahl)

ZUSÄTZLICHE KATEGORIEN (alle Anforderungen die im Dokument stehen):
- prozessor: Anforderung an Prozessor/CPU
- display: Display-Details (Auflösung, Typ, Features)
- ram: RAM-Details (Typ, Slots, Aufrüstbarkeit)
- speicher: Storage-Details (Typ, Interface, Größe)
- grafik: Grafik-Anforderungen
- anschlüsse: Ports (USB-C, USB-A, HDMI, Thunderbolt, Ethernet, Audio)
- konnektivität: WiFi, Bluetooth, Wake-on-LAN, PXE, Miracast
- eingabegeräte: Tastatur, Touchpad, Mikrofon
- kamera: Webcam, Auflösung, Abdeckung
- sensoren: Thermal, Ambient Light, Hall-Effekt
- sicherheit: Kensington Lock, BIOS-Features
- akku: Kapazität, Laufzeit, Austauschbarkeit
- betriebssystem: Windows-Version
- netzteil: Netzteil-Typ
- garantie: Dauer, Details

Erkenne und extrahiere auch Anforderungen, die nicht in dieser Liste stehen.

Antworte NUR mit einem validen JSON-Objekt. Verwende null für fehlende Werte."""

        if self.llm is None:
            logger.warning("Cannot extract LV requirements: no LLM API key configured")
            return {}

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            requirements = json.loads(content)

            structured_fields = {
                "min_display_brightness_nits", "min_screen_to_body_ratio", "max_weight_kg",
                "min_ram_gb", "min_storage_tb", "min_battery_wh", "min_battery_runtime_hours",
                "display_size_inch_min", "display_size_inch_max", "warranty_years"
            }

            structured_data = {}
            flexible_data = {}

            for key, value in requirements.items():
                if key == "document_type":
                    continue
                if key in structured_fields and value is not None:
                    structured_data[key] = value
                elif value is not None:
                    flexible_data[key] = value

            result = {"document_type": "requirement"}
            result.update(structured_data)
            if flexible_data:
                result["additional_requirements"] = flexible_data

            logger.info(f"Extracted LV requirements: structured={len(structured_data)}, "
                       f"additional={len(flexible_data)}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in LV extraction: {e}")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}
        except Exception as e:
            logger.error(f"Error in LV extraction: {e}", exc_info=True)
            return {}

    def extract_from_document(self, text: str, filename: str) -> Dict:
        """Hauptmethode: Erkennt Dokumenttyp und extrahiert entsprechend."""
        doc_type = self.detect_document_type(text, filename)

        if doc_type == "requirement":
            specs = self.extract_requirements(text, filename)
            specs["document_type"] = "requirement"
        else:
            specs = self.extract_product_specs(text, filename)
            specs["document_type"] = "product"

        return specs
