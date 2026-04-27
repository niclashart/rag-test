"""Tender upload, review, and matching page."""
from __future__ import annotations

import re
import time

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

_REQUIREMENT_TYPE_LABELS = {
    "must": "Muss",
    "should": "Soll",
    "optional": "Optional",
    "scored": "Bewertet",
    "service": "Service",
    "contract": "Vertrag",
    "unknown": "Unbekannt",
}

_STATUS_ICONS = {
    "pending": "⏳",
    "approved": "✅",
    "rejected": "❌",
    "edited": "✏️",
}


def show_leistungsverzeichnis():
    st.title("Ausschreibungs-Matching")
    upload_tab, review_tab, matches_tab = st.tabs(["Upload", "Review", "Matches"])
    with upload_tab:
        _show_upload()
    with review_tab:
        _show_review()
    with matches_tab:
        _show_matches()


def _show_upload():
    file = st.file_uploader("Ausschreibungs-PDF", type=["pdf"])
    if st.button("PDF analysieren", type="primary", use_container_width=True, disabled=file is None):
        with st.spinner("PDF wird hochgeladen..."):
            try:
                response = requests.post(f"{API_BASE_URL}/api/tenders/upload", files={"file": file}, timeout=30)
                response.raise_for_status()
                tender = response.json()
                st.session_state["tender_id"] = tender["id"]
                st.success("Upload abgeschlossen. Analyse läuft im Backend.")
            except requests.exceptions.RequestException as exc:
                _show_error(exc)
                return

        _wait_for_tender_processing(st.session_state["tender_id"])

    tender_id = st.session_state.get("tender_id")
    if tender_id:
        tender = _get_json(f"/api/tenders/{tender_id}")
        if tender:
            meta = tender.get("metadata") or {}
            st.caption(f"Status: {tender['status']}")
            st.progress(_status_progress(tender["status"], meta))
            st.write(
                {
                    "Phase": meta.get("phase", tender["status"]),
                    "PDF gelesen": bool(meta.get("page_count")),
                    "Sections erkannt": meta.get("section_count", 0),
                    "Chunks verarbeitet": f"{meta.get('chunks_processed', 0)} / {meta.get('chunks_total', meta.get('chunk_count', 0))}",
                    "DeepSeek Calls": f"{meta.get('llm_calls', 0)} / {meta.get('llm_max_calls', '-')}",
                    "DeepSeek übersprungen": meta.get("llm_skipped", 0),
                    "Requirements extrahiert": meta.get("requirement_count", 0),
                    "Review offen": meta.get("needs_review_count", 0),
                }
            )
            if meta.get("message"):
                st.info(meta["message"])
            if tender["status"] == "error" and meta.get("error"):
                st.error(meta["error"])
            if tender["status"] == "processing" and st.button("Status aktualisieren", use_container_width=True):
                st.rerun()


def _wait_for_tender_processing(tender_id, max_wait_seconds=600):
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    started_at = time.monotonic()

    while time.monotonic() - started_at < max_wait_seconds:
        tender = _get_json(f"/api/tenders/{tender_id}", show_errors=False)
        if not tender:
            time.sleep(2)
            continue

        meta = tender.get("metadata") or {}
        status = tender["status"]
        phase = meta.get("phase", status)
        processed = meta.get("chunks_processed", 0)
        total = meta.get("chunks_total", meta.get("chunk_count", 0))
        status_placeholder.caption(f"Status: {status} | Phase: {phase} | Chunks: {processed}/{total}")
        progress_placeholder.progress(_status_progress(status, meta))

        if status == "review_ready":
            st.success(f"Review bereit: {meta.get('requirement_count', 0)} Anforderungen")
            st.rerun()
        if status == "error":
            st.error(meta.get("error", "Extraktion fehlgeschlagen."))
            return
        if status == "matching_completed":
            st.rerun()

        time.sleep(3)

    st.info("Die Analyse läuft weiter im Backend. Aktualisiere den Status später erneut.")


def _status_progress(status, meta=None):
    meta = meta or {}
    if status == "processing":
        total = meta.get("chunks_total") or meta.get("chunk_count") or 0
        processed = meta.get("chunks_processed") or 0
        if total:
            return min(0.95, 0.25 + 0.65 * (processed / total))
        phase_progress = {
            "pdf_extracting": 0.15,
            "sections_detecting": 0.25,
            "llm_extracting": 0.35,
        }
        return phase_progress.get(meta.get("phase"), 0.2)
    return {
        "uploaded": 0.1,
        "review_ready": 0.75,
        "matching_completed": 1.0,
        "error": 1.0,
    }.get(status, 0.2)


def _category_from_attribute(attr: str) -> str:
    if not attr:
        return "Sonstige"
    prefix = attr.split(".")[0].lower()
    return {
        "cpu": "Prozessor",
        "memory": "Arbeitsspeicher",
        "storage": "Speicher",
        "gpu": "Grafik",
        "display": "Display",
        "ports": "Anschlüsse",
        "network": "Netzwerk",
        "keyboard": "Tastatur",
        "webcam": "Kamera",
        "security": "Sicherheit",
        "bios": "BIOS",
        "battery": "Akku",
        "warranty": "Garantie",
        "os": "Betriebssystem",
        "service": "Service",
        "certifications": "Zertifizierungen",
        "sustainability": "Nachhaltigkeit",
        "manufacturer": "Hersteller",
        "unknown": "Unbekannt",
    }.get(prefix, "Sonstige")


def _show_review():
    tender_id = st.session_state.get("tender_id")
    tender_id = st.text_input("Tender ID", value=tender_id or "")
    if tender_id:
        st.session_state["tender_id"] = tender_id
    else:
        st.info("Erst PDF hochladen oder Tender ID eintragen.")
        return

    requirements = _get_json(f"/api/tenders/{tender_id}/requirements") or []
    if not requirements:
        st.info("Keine Anforderungen gefunden.")
        return

    # --- Summary metrics ---
    total = len(requirements)
    pending = sum(1 for r in requirements if r["status"] == "pending")
    approved = sum(1 for r in requirements if r["status"] == "approved")
    rejected = sum(1 for r in requirements if r["status"] == "rejected")
    edited = sum(1 for r in requirements if r["status"] == "edited")
    needs_review = sum(1 for r in requirements if r["needs_review"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Gesamt", total)
    m2.metric("Offen", pending)
    m3.metric("Freigegeben", approved)
    m4.metric("Bearbeitet", edited)
    m5.metric("Review nötig", needs_review, delta=None if needs_review == 0 else f"von {total}")

    # --- Batch actions ---
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("High Confidence freigeben", use_container_width=True):
            result = _post_json(f"/api/tenders/{tender_id}/requirements/approve-high-confidence")
            if result:
                st.success(f"{result['approved']} Anforderungen freigegeben")
                st.rerun()
    with col_batch2:
        if st.button("Alle unknown ablehnen", use_container_width=True):
            rejected_count = 0
            for req in requirements:
                if req["attribute"].startswith("unknown.") and req["status"] == "pending":
                    if _post_json(f"/api/tenders/requirements/{req['id']}/reject"):
                        rejected_count += 1
            if rejected_count:
                st.success(f"{rejected_count} unknown-Anforderungen abgelehnt")
                st.rerun()

    st.divider()

    # --- Filters ---
    col_f1, col_f2, col_f3 = st.columns(3)
    status_filter = col_f1.selectbox("Status", ["alle", "pending", "approved", "rejected", "edited"])
    type_filter = col_f2.selectbox("Typ", ["alle"] + sorted({_REQUIREMENT_TYPE_LABELS.get(r["requirement_type"], r["requirement_type"]) for r in requirements}))
    review_filter = col_f3.selectbox("Review", ["alle", "review nötig", "ok"])

    filtered = _apply_filters(requirements, status_filter, type_filter, review_filter)

    if not filtered:
        st.info("Keine Anforderungen passen zu den Filtern.")
        return

    # --- Group by category ---
    categories: dict[str, list] = {}
    for req in filtered:
        cat = _category_from_attribute(req["attribute"])
        categories.setdefault(cat, []).append(req)

    category_order = [
        "Prozessor", "Arbeitsspeicher", "Speicher", "Grafik", "Display",
        "Anschlüsse", "Netzwerk", "Tastatur", "Kamera", "Sicherheit",
        "BIOS", "Akku", "Betriebssystem", "Garantie", "Service",
        "Zertifizierungen", "Unbekannt", "Sonstige",
    ]

    for cat in category_order:
        items = categories.get(cat)
        if not items:
            continue

        cat_ok = sum(1 for r in items if r["status"] == "approved")
        cat_total = len(items)

        with st.expander(f"{cat} ({cat_ok}/{cat_total} freigegeben)", expanded=cat_ok < cat_total):
            for req in items:
                _render_requirement_card(req)


def _render_requirement_card(req):
    status_icon = _STATUS_ICONS.get(req["status"], "?")
    type_label = _REQUIREMENT_TYPE_LABELS.get(req["requirement_type"], req["requirement_type"])
    is_unknown = req["attribute"].startswith("unknown.")
    attr_display = req["attribute"] if not is_unknown else f"⚠️ {req['attribute']}"

    col_header, col_actions = st.columns([4, 1])
    with col_header:
        st.markdown(
            f"**{status_icon} {type_label}** | {attr_display} "
            f"`{req['operator']}` **{_fmt_value(req)}** "
            f"| Confidence: {req['confidence']:.0%}"
            f"{' | ⚠️ Review' if req['needs_review'] else ''}"
        )
    with col_actions:
        if req["status"] == "pending":
            col_a, col_b = st.columns(2)
            if col_a.button("✅", key=f"approve_{req['id']}", help="Freigeben"):
                if _post_json(f"/api/tenders/requirements/{req['id']}/approve"):
                    st.rerun()
            if col_b.button("❌", key=f"reject_{req['id']}", help="Ablehnen"):
                if _post_json(f"/api/tenders/requirements/{req['id']}/reject"):
                    st.rerun()
        elif req["status"] == "approved":
            if st.button("↩️", key=f"unapprove_{req['id']}", help="Zurückziehen"):
                if _patch_json(f"/api/tenders/requirements/{req['id']}", {"status": "pending"}):
                    st.rerun()
        elif req["status"] == "rejected":
            if st.button("↩️", key=f"unreject_{req['id']}", help="Wiederherstellen"):
                if _patch_json(f"/api/tenders/requirements/{req['id']}", {"status": "pending"}):
                    st.rerun()

    with st.expander("Details & Bearbeiten", key=f"detail_{req['id']}"):
        st.caption(f"S. {req.get('source_page', '?')} | {req.get('rationale', '')}")
        st.text(req["original_text"])

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        attribute = c1.text_input("Attribut", value=req["attribute"], key=f"attr_{req['id']}")
        operator = c2.selectbox("Operator", [">=", "<=", "=", "contains", "exists", "one_of", "compatible_with"],
                                index=[">=", "<=", "=", "contains", "exists", "one_of", "compatible_with"].index(req["operator"]),
                                key=f"op_{req['id']}")
        value = c3.text_input("Wert", value=str(req["value"]), key=f"val_{req['id']}")
        unit = c4.text_input("Einheit", value=req.get("unit") or "", key=f"unit_{req['id']}")
        if st.button("Speichern", key=f"save_{req['id']}"):
            edited = {"attribute": attribute, "operator": operator, "value": _coerce_value(value), "unit": unit or None, "status": "edited"}
            if _patch_json(f"/api/tenders/requirements/{req['id']}", edited):
                st.rerun()


def _fmt_value(req):
    v = req["value"]
    u = req.get("unit") or ""
    if isinstance(v, bool):
        return "Ja" if v else "Nein"
    return f"{v} {u}".strip()


def _apply_filters(requirements, status_filter, type_filter, review_filter):
    filtered = []
    for req in requirements:
        if status_filter != "alle" and req["status"] != status_filter:
            continue
        if type_filter != "alle":
            req_label = _REQUIREMENT_TYPE_LABELS.get(req["requirement_type"], req["requirement_type"])
            if req_label != type_filter:
                continue
        if review_filter == "review nötig" and not req["needs_review"]:
            continue
        if review_filter == "ok" and req["needs_review"]:
            continue
        filtered.append(req)
    return filtered


def _show_matches():
    tender_id = st.session_state.get("tender_id")
    if not tender_id:
        st.info("Kein Tender gewählt.")
        return

    if st.button("Matching starten", type="primary", use_container_width=True):
        with st.spinner("Deterministisches Matching läuft..."):
            result = _post_json(f"/api/tenders/{tender_id}/match", timeout=180)
            if result is not None:
                st.success(f"{len(result)} Produkte bewertet")

    matches = _get_json(f"/api/tenders/{tender_id}/matches") or []
    if not matches:
        st.info("Noch keine Matches.")
        return

    eligibility = st.selectbox("Eligibility", ["alle", "eligible", "unknown", "not_eligible"])
    for match in sorted(matches, key=lambda item: (item["eligibility"] != "eligible", -item["score"])):
        if eligibility != "alle" and match["eligibility"] != eligibility:
            continue
        with st.expander(f"{match['eligibility']} | {match['model']} | {match['score']}/{match['max_score']}"):
            st.write(f"Muss erfüllt: {match['must_passed']} | Muss fehlgeschlagen: {match['must_failed']} | Unbekannt: {match['unknown_count']}")
            detail = _get_json(f"/api/tenders/{tender_id}/matches/{match['product_id']}")
            if detail:
                st.dataframe(detail.get("requirement_results", []), use_container_width=True)


def _get_json(path, show_errors=True):
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        if show_errors:
            _show_error(exc)
        return None


def _post_json(path, timeout=60):
    try:
        response = requests.post(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        _show_error(exc)
        return None


def _patch_json(path, payload):
    try:
        response = requests.patch(f"{API_BASE_URL}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        _show_error(exc)
        return None


def _coerce_value(value):
    try:
        number = float(value.replace(",", "."))
        return int(number) if number.is_integer() else number
    except ValueError:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return value


def _show_error(exc):
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            st.error(response.json().get("detail", str(exc)))
            return
        except ValueError:
            pass
    st.error(str(exc))
