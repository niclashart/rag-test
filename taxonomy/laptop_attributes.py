"""Canonical laptop attribute taxonomy used by tender matching."""

from __future__ import annotations

from dataclasses import dataclass
import re


LAPTOP_ATTRIBUTES = {
    "cpu.model",
    "cpu.vendor",
    "cpu.vpro",
    "cpu.benchmark.passmark",
    "memory.capacity",
    "memory.type",
    "memory.slots",
    "memory.upgradeable",
    "storage.capacity",
    "storage.interface",
    "storage.upgradeable",
    "gpu.type",
    "gpu.dedicated",
    "display.size",
    "display.resolution",
    "display.aspect_ratio",
    "display.touch",
    "display.brightness",
    "display.color_coverage.srgb",
    "display.anti_glare",
    "ports.thunderbolt4",
    "ports.usb_a",
    "ports.usb_c",
    "ports.hdmi",
    "ports.rj45_native",
    "network.wifi",
    "network.bluetooth",
    "network.wake_on_lan",
    "network.pxe",
    "keyboard.backlit",
    "keyboard.layout",
    "webcam.resolution",
    "security.tpm",
    "security.lock_slot",
    "bios.mac_passthrough",
    "bios.network_update",
    "battery.capacity",
    "battery.runtime",
    "battery.replaceable",
    "warranty.years",
    "warranty.keep_your_drive",
    "certifications.energy_star",
    "certifications.ce",
    "certifications.tco",
    "certifications.rohs",
    "certifications.mil_std_810h",
    "service.onsite",
    "service.response_time",
    "service.repair_time",
    "service.coverage_window",
    "os.windows_compatible",
    "os.linux_compatible",
    "sustainability.recycled_plastic_percent",
    "sustainability.recycled_metal_percent",
    "sustainability.recyclable_percent",
    "manufacturer.iso_9001",
    "manufacturer.iso_14001",
    "manufacturer.iso_27001",
}


@dataclass(frozen=True)
class AttributeGuess:
    attribute: str
    needs_review: bool


_SYNONYMS = [
    (r"\b(cpu|prozessor|processor)\b", "cpu.model"),
    (r"\b(vpro|v-pro)\b", "cpu.vpro"),
    (r"\b(passmark|benchmark)\b", "cpu.benchmark.passmark"),
    (r"\b(ram|arbeitsspeicher|hauptspeicher|memory)\b", "memory.capacity"),
    (r"\b\d+(?:[,.]\d+)?\s*gb\b.*\b(ddr4|ddr5|lpddr|sodimm|so-dimm)\b", "memory.capacity"),
    (r"\b(ddr4|ddr5|lpddr)\b", "memory.type"),
    (r"\b(slot|sodimm|so-dimm)\b", "memory.slots"),
    (r"\b(ssd|massenspeicher|storage|speicher)\b", "storage.capacity"),
    (r"\b(nvme|pcie|sata)\b", "storage.interface"),
    (r"\b(grafik|graphics|gpu)\b", "gpu.type"),
    (r"\b(gewicht|weight|kg|gramm)\b", "unknown.weight"),
    (r"(\b(bildschirm|display|screen|monitor)\b.*\b(zoll|inch|\")\b|\b\d+(?:[,.]\d+)?\s*(zoll|inch|\")\b)", "display.size"),
    (r"\b(auflösung|resolution|wuxga|fhd|qhd|uhd|\d{3,4}\s*x\s*\d{3,4})\b", "display.resolution"),
    (r"\b(helligkeit|nits|cd/m)\b", "display.brightness"),
    (r"\b(srgb)\b", "display.color_coverage.srgb"),
    (r"\b(entspiegelt|anti-glare|anti glare|antiglare)\b", "display.anti_glare"),
    (r"\b(touch|touchscreen)\b", "display.touch"),
    (r"\b(thunderbolt\s*4|tb4)\b", "ports.thunderbolt4"),
    (r"\b(usb-a|usb a|type-a)\b", "ports.usb_a"),
    (r"\b(usb-c|usb c|type-c)\b", "ports.usb_c"),
    (r"\b(hdmi)\b", "ports.hdmi"),
    (r"\b(rj45|ethernet)\b", "ports.rj45_native"),
    (r"\b(wi-?fi|wlan)\b", "network.wifi"),
    (r"\b(bluetooth)\b", "network.bluetooth"),
    (r"\b(wake on lan|wol)\b", "network.wake_on_lan"),
    (r"\b(pxe)\b", "network.pxe"),
    (r"\b(tastatur|keyboard)\b.*\b(beleuchtet|backlit)\b", "keyboard.backlit"),
    (r"\b(tastatur|keyboard)\b.*\b(deutsch|german|layout)\b", "keyboard.layout"),
    (r"\b(webcam|kamera|camera)\b", "webcam.resolution"),
    (r"\b(tpm)\b", "security.tpm"),
    (r"\b(kensington|lock slot|sicherheitsschloss)\b", "security.lock_slot"),
    (r"\b(mac[- ]?passthrough)\b", "bios.mac_passthrough"),
    (r"\b(bios).*\b(update|netzwerk)\b", "bios.network_update"),
    (r"\b(akku|battery).*\b(wh|wattstunden)\b", "battery.capacity"),
    (r"\b(akkulaufzeit|battery runtime|laufzeit)\b", "battery.runtime"),
    (r"\b(garantie|warranty)\b", "warranty.years"),
    (r"\b(onsite|vor[- ]?ort)\b", "service.onsite"),
    (r"\b(windows)\b", "os.windows_compatible"),
    (r"\b(linux)\b", "os.linux_compatible"),
    (r"\b(energy star)\b", "certifications.energy_star"),
    (r"\b(tco)\b", "certifications.tco"),
    (r"\b(rohs)\b", "certifications.rohs"),
    (r"\b(mil[- ]?std[- ]?810h)\b", "certifications.mil_std_810h"),
    (r"\b(iso\s*9001)\b", "manufacturer.iso_9001"),
    (r"\b(iso\s*14001)\b", "manufacturer.iso_14001"),
    (r"\b(iso\s*27001)\b", "manufacturer.iso_27001"),
]


def normalize_attribute(text: str) -> AttributeGuess:
    """Map tender/product wording to canonical taxonomy."""
    lower = text.lower()
    for pattern, attribute in _SYNONYMS:
        if re.search(pattern, lower):
            return AttributeGuess(attribute=attribute, needs_review=attribute.startswith("unknown."))

    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")[:48] or "requirement"
    return AttributeGuess(attribute=f"unknown.{slug}", needs_review=True)


def is_known_attribute(attribute: str) -> bool:
    return attribute in LAPTOP_ATTRIBUTES
