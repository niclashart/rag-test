"""Login and registration page."""
import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def check_backend_connection():
    """Check if backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_login_page():
    """Show login/registration page."""
    st.markdown('<h1 class="main-header">RAG Pipeline</h1>', unsafe_allow_html=True)
    
    # Check backend connection
    if not check_backend_connection():
        st.error("⚠️ Backend nicht erreichbar! Bitte starten Sie das Backend mit:")
        st.code("./run_backend.sh", language="bash")
        st.info("Das Backend sollte auf http://localhost:8000 laufen.")
        return
    
    tab1, tab2 = st.tabs(["Anmelden", "Registrieren"])
    
    with tab1:
        st.header("Anmelden")
        with st.form("login_form"):
            email = st.text_input("E-Mail", key="login_email")
            password = st.text_input("Passwort", type="password", key="login_password")
            submit = st.form_submit_button("Anmelden", use_container_width=True)
            
            if submit:
                try:
                    # Login request
                    form_data = {
                        "username": email,
                        "password": password
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/api/auth/login",
                        data=form_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            st.session_state.access_token = data["access_token"]
                            st.session_state.authenticated = True
                            
                            # Get user info
                            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                            user_response = requests.get(
                                f"{API_BASE_URL}/api/auth/me",
                                headers=headers,
                                timeout=10
                            )
                            if user_response.status_code == 200:
                                st.session_state.user = user_response.json()
                            
                            st.success("Erfolgreich angemeldet!")
                            st.rerun()
                        except (ValueError, requests.exceptions.JSONDecodeError):
                            st.error("Ungültige Antwort vom Backend. Bitte überprüfen Sie die Backend-Logs.")
                    else:
                        try:
                            error_detail = response.json().get('detail', 'Unbekannter Fehler')
                        except (ValueError, requests.exceptions.JSONDecodeError):
                            error_detail = f"HTTP {response.status_code}: {response.text[:100]}"
                        st.error(f"Anmeldung fehlgeschlagen: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Verbindungsfehler: Backend nicht erreichbar. Bitte starten Sie das Backend mit './run_backend.sh'")
                except requests.exceptions.Timeout:
                    st.error("Zeitüberschreitung: Das Backend antwortet nicht.")
                except Exception as e:
                    st.error(f"Fehler: {str(e)}")
    
    with tab2:
        st.header("Registrieren")
        with st.form("register_form"):
            email = st.text_input("E-Mail", key="register_email")
            password = st.text_input("Passwort", type="password", key="register_password")
            confirm_password = st.text_input("Passwort bestätigen", type="password", key="register_confirm")
            submit = st.form_submit_button("Registrieren", use_container_width=True)
            
            if submit:
                if password != confirm_password:
                    st.error("Passwörter stimmen nicht überein")
                elif len(password) < 6:
                    st.error("Passwort muss mindestens 6 Zeichen lang sein")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/auth/register",
                            json={"email": email, "password": password},
                            timeout=10
                        )
                        
                        if response.status_code == 201:
                            st.success("Registrierung erfolgreich! Bitte melden Sie sich an.")
                        else:
                            try:
                                error_data = response.json()
                                error_detail = error_data.get('detail', 'Unbekannter Fehler')
                                # Show full error in expander for debugging
                                with st.expander("Fehlerdetails anzeigen"):
                                    st.json(error_data)
                            except (ValueError, requests.exceptions.JSONDecodeError):
                                error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                                with st.expander("Fehlerdetails anzeigen"):
                                    st.text(response.text)
                            st.error(f"Registrierung fehlgeschlagen: {error_detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("Verbindungsfehler: Backend nicht erreichbar. Bitte starten Sie das Backend mit './run_backend.sh'")
                    except requests.exceptions.Timeout:
                        st.error("Zeitüberschreitung: Das Backend antwortet nicht.")
                    except Exception as e:
                        st.error(f"Fehler: {str(e)}")

