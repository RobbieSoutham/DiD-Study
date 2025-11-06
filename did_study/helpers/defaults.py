"""Default mappings and constants for the DiD CCUS pipeline.

This module defines the default mapping used to allocate emission
sectors to CCUS sectors and to relate supply/demand categories to
emission sectors.  The mapping is defined as a nested dictionary
mirroring the structure in the original code.  Users may override
these mappings by passing a custom dictionary via the
``StudyConfig.mapping`` parameter.
"""

from __future__ import annotations

from typing import Dict, Any, Set

DEFAULT_MAPPING: Dict[str, Any] = {
    "emissions_to_ccus": {
        "Power and heat": {"Main Activity Electricity and Heat Production"},
        "Other fuel transformation": {
            "Petroleum Refining - Manufacture of Solid Fuels and Other Energy Industries",
            "Solid Fuels",
        },
        "Natural gas processing/LNG": {"Oil and Natural Gas"},
        "Cement": {
            "Cement production",
            "Lime production",
            "Other Process Uses of Carbonates",
            "Glass Production",
        },
        "Iron and steel": {"Metal Industry"},
        "Chemicals": {"Chemical Industry"},
        "Hydrogen or ammonia": set(),
        "Biofuels": set(),
        "DAC": set(),
        "Agriculture (no CCUS)": {
            "Enteric Fermentation",
            "Manure Management",
            "Indirect N2O Emissions from manure management",
            "Direct N2O Emissions from managed soils",
            "Indirect N2O Emissions from managed soils",
            "Indirect N2O emissions from the atmospheric deposition of nitrogen in NOx and NH3",
            "Rice cultivations",
            "Urea application",
            "Emissions from biomass burning",
            "Liming",
        },
        "Waste (no CCUS)": {
            "Solid Waste Disposal",
            "Wastewater Treatment and Discharge",
            "Biological Treatment of Solid Waste",
            "Incineration and Open Burning of Waste",
        },
        "Fugitives/Other (no CCUS)": {"Fossil fuel fires", "Non-Specified"},
    },
    "ccus_to_supdem": {
        "Power and heat": ["Total energy supply (PJ)"],
        "Other fuel transformation": ["Total energy supply (PJ)"],
        "Natural gas processing/LNG": ["Total energy supply (PJ)"],
        "Cement": ["Industry (PJ)"],
        "Iron and steel": ["Industry (PJ)"],
        "Chemicals": ["Industry (PJ)"],
        "Hydrogen or ammonia": ["Industry (PJ)"],
        "Biofuels": ["Industry (PJ)"],
        "DAC": ["Total energy supply (PJ)"],
        "Agriculture (no CCUS)": ["Other final consumption (PJ)"],
        "Waste (no CCUS)": ["Total energy supply (PJ)"],
        "Fugitives/Other (no CCUS)": ["Total energy supply (PJ)"],
    },
    "supdem_to_emissions": {
        "Total energy supply (PJ)": {
            "Main Activity Electricity and Heat Production",
            "Petroleum Refining - Manufacture of Solid Fuels and Other Energy Industries",
            "Solid Fuels",
            "Oil and Natural Gas",
        },
        "Industry (PJ)": {
            "Manufacturing Industries and Construction",
            "Cement production",
            "Lime production",
            "Glass Production",
            "Metal Industry",
            "Chemical Industry",
            "Other Process Uses of Carbonates",
            "Non-Energy Products from Fuels and Solvent Use",
            "Product Uses as Substitutes for Ozone Depleting Substances",
            "Other Product Manufacture and Use",
            "Electronics Industry",
        },
        "Transport (PJ)": {
            "Road Transportation no resuspension",
            "Civil Aviation",
            "Railways",
            "Water-borne Navigation",
            "Other Transportation",
        },
        "Residential (PJ)": {"Residential and other sectors"},
        "Commercial and public services (PJ)": {"Residential and other sectors"},
        "Other final consumption (PJ)": {"Residential and other sectors", "Non-Specified"},
    },
}

__all__ = ["DEFAULT_MAPPING"]