"""
Satellite Data Utilities for FlowSat.

References:
    - DiffusionSat data_util.py (caption generation, metadata normalization)

Provides:
    - Caption generation from FMoW metadata (used as fallback when no
      VLM caption is available on disk)
    - Metadata normalization/unnormalization
    - Image normalization helpers
"""

import random
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np
from torch import Tensor


# ---------------------------------------------------------------------------
# FMoW Category Definitions
# ---------------------------------------------------------------------------

FMOW_CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution",
    "electric_substation", "factory_or_powerplant", "fire_station",
    "flooded_road", "fountain", "gas_station", "golf_course",
    "ground_transportation_station", "helipad", "hospital",
    "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility",
    "park", "parking_lot_or_garage", "place_of_worship",
    "police_station", "port", "prison", "race_track",
    "railway_bridge", "recreational_facility", "road_bridge",
    "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm",
    "space_facility", "stadium", "storage_tank",
    "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal",
    "water_treatment_facility", "wind_farm", "zoo",
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(FMOW_CATEGORIES)}


# Country code mappings (subset for common codes)
COUNTRY_CODE_MAP = {
    "US": "United States", "GB": "United Kingdom", "FR": "France",
    "DE": "Germany", "CN": "China", "JP": "Japan", "IN": "India",
    "BR": "Brazil", "RU": "Russia", "AU": "Australia", "CA": "Canada",
    "IT": "Italy", "ES": "Spain", "KR": "South Korea", "MX": "Mexico",
    "ID": "Indonesia", "TR": "Turkey", "SA": "Saudi Arabia",
    "ZA": "South Africa", "AR": "Argentina", "EG": "Egypt",
    "NG": "Nigeria", "PK": "Pakistan", "TH": "Thailand",
    "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden",
    "PL": "Poland", "BE": "Belgium", "AT": "Austria",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland",
    "IE": "Ireland", "PT": "Portugal", "GR": "Greece",
    "CZ": "Czech Republic", "RO": "Romania", "HU": "Hungary",
    "NZ": "New Zealand", "SG": "Singapore", "MY": "Malaysia",
    "PH": "Philippines", "VN": "Vietnam", "CL": "Chile",
    "CO": "Colombia", "PE": "Peru", "UA": "Ukraine",
    "IL": "Israel", "AE": "United Arab Emirates",
}

CATEGORY_DESCRIPTORS = {
    "airport": "Large expanses of paved runways and taxiways forming long straight gray strips, often intersecting. Terminal buildings, parking aprons with aircraft, and service roads.",
    "airport_hangar": "Rectangular or semi-cylindrical large buildings near runways, often metallic or white. Adjacent to taxiways with wide entrances facing aircraft parking zones.",
    "airport_terminal": "Complex elongated structures with multiple gates extending outward. Adjacent to aircraft parking bays, connected to road networks.",
    "amusement_park": "Highly colorful irregular layout with circular rides, roller coaster tracks, clustered attractions. Bright colors, varied textures, surrounded by parking lots.",
    "aquaculture": "Grid-like arrangement of rectangular ponds filled with water, often greenish or brown. Separated by narrow embankments near coastal areas.",
    "archaeological_site": "Irregular patterns of ruins or excavated structures, sandy or earthy color. Circular or rectangular outlines, sparse vegetation.",
    "barn": "Simple rectangular structures with pitched roofs, often red or brown. Rural area surrounded by fields and dirt paths.",
    "border_checkpoint": "Structured layout along boundary line with multiple lanes, inspection booths, barriers. Roads, fencing, and sparse vegetation.",
    "burial_site": "Organized rows of small rectangular plots or tombstones. Regular grid-like pattern with pathways and sparse greenery.",
    "car_dealership": "Clusters of uniformly parked vehicles in neat rows adjacent to showroom building. High reflectivity from car roofs, large paved surfaces.",
    "construction_site": "Irregular terrain with exposed soil, machinery, partially built structures. Mixed textures with dirt, debris, construction materials.",
    "crop_field": "Large rectangular or circular plots with uniform texture, varying shades of green or brown. Grid patterns with irrigation lines.",
    "dam": "Linear barrier across water body, often concrete, separating water levels. Reservoir on one side, downstream flow on other.",
    "debris_or_rubble": "Chaotic arrangement of irregular shapes and textures, gray or brown. Scattered fragments and uneven terrain.",
    "educational_institution": "Cluster of medium-sized buildings arranged systematically with open grounds, sports fields, pathways. Mixed vegetation and structured layout.",
    "electric_substation": "Grid-like arrangement of transformers, wires, metallic structures. Enclosed area with geometric patterns, minimal vegetation.",
    "factory_or_powerplant": "Large industrial buildings with smokestacks, storage tanks, pipelines. Paved areas and transport links.",
    "fire_station": "Medium-sized building with garage bays for fire trucks, adjacent parking and road access in urban area.",
    "flooded_road": "Road partially submerged in water with visible linear structure under reflective water surface.",
    "fountain": "Circular or decorative structure with water at center in plazas or parks. Symmetrical design with pathways.",
    "gas_station": "Small structure with canopy covering fuel pumps, rectangular. Adjacent to roads with parked vehicles.",
    "golf_course": "Large green area with smooth grassy textures, sand bunkers, water hazards. Curved fairways and distinct holes.",
    "ground_transportation_station": "Large building with multiple bus or vehicle bays. Adjacent parking and road connectivity.",
    "helipad": "Circular or square pad with prominent H marking. Usually isolated or on rooftops, surrounded by open space.",
    "hospital": "Large building complex with multiple wings, emergency access areas and parking lots. Roads and green spaces.",
    "impoverished_settlement": "Dense clusters of small irregular structures with varied materials. Narrow pathways, minimal planning, uneven layout.",
    "interchange_or_intersection": "Complex road patterns with loops, ramps, multiple crossing levels. Smooth curved roads forming geometric patterns.",
    "lake_or_pond": "Irregular or circular water body with smooth dark or reflective surface. Surrounded by vegetation or land.",
    "lighthouse": "Tall narrow structure near coastline, often white. Minimal surrounding buildings and open terrain.",
    "military_facility": "Organized layout with secured boundaries, barracks, equipment areas. Large open grounds and restricted access roads.",
    "nuclear_powerplant": "Large complex with cooling towers, reactor buildings, water sources. Distinct dome-shaped structures and industrial layout.",
    "office_building": "Rectangular multi-story buildings with parking lots and road access in dense urban areas.",
    "oil_or_gas_facility": "Clusters of storage tanks, pipelines, processing units. Industrial layout with metallic structures.",
    "park": "Green open space with trees, pathways, sometimes water features. Irregular layout with natural textures.",
    "parking_lot_or_garage": "Large paved area with rows of parked vehicles. Regular grid pattern with marked spaces.",
    "place_of_worship": "Distinct architectural structure with domes, spires, or crosses. Surrounded by open space or smaller buildings.",
    "police_station": "Medium-sized building with parking and road access, centrally located in urban areas.",
    "port": "Coastal area with docks, ships, cranes, containers. Structured layout along water edge with blue water visible.",
    "prison": "Highly secured complex with perimeter walls, watchtowers, internal blocks in grid-like patterns.",
    "race_track": "Oval or irregular track with smooth surface, surrounded by stands or open land.",
    "railway_bridge": "Linear structure carrying tracks over water or land. Narrow and elongated with rail lines visible.",
    "recreational_facility": "Varied structures like courts or playgrounds with bright colors and organized layout.",
    "road_bridge": "Wide linear structure for vehicles crossing water or terrain. Connected to road networks.",
    "runway": "Long straight paved strip with markings, often isolated and aligned with wind direction.",
    "shipyard": "Industrial coastal area with ships under construction, cranes, dry docks along waterfront.",
    "shopping_mall": "Large building complex with extensive parking areas. Rectangular layout with multiple entrances.",
    "single-unit_residential": "Individual houses with surrounding yards in suburban layouts with roads and driveways.",
    "smokestack": "Tall cylindrical structure casting shadow, attached to industrial buildings.",
    "solar_farm": "Large arrays of dark rectangular panels arranged in regular grids on open cleared land.",
    "space_facility": "Launch pads, large buildings, clear zones. Distinct geometric layouts and restricted areas.",
    "stadium": "Large circular or oval structure with open center field. Surrounded by parking lots and roads.",
    "storage_tank": "Circular tanks with flat tops, often clustered together in industrial areas.",
    "surface_mine": "Large excavated area with terraced patterns and exposed earth. Irregular shapes and haul roads.",
    "swimming_pool": "Rectangular or irregular blue water body within residential or recreational areas.",
    "toll_booth": "Multiple lanes with small booths along road in structured linear arrangement.",
    "tower": "Tall narrow structure casting shadow, often isolated or part of infrastructure.",
    "tunnel_opening": "Dark opening in terrain or hillside connected to roads or rail tracks.",
    "waste_disposal": "Large area with irregular piles of waste, brown or gray. No uniform structure.",
    "water_treatment_facility": "Series of circular or rectangular tanks with water processing patterns. Organized industrial layout.",
    "wind_farm": "Multiple wind turbines spaced evenly across open land. Long shadows and circular bases.",
    "zoo": "Irregular enclosures with vegetation, pathways, structures. Mixed natural and artificial patterns.",
}


# ---------------------------------------------------------------------------
# Caption Generation
# ---------------------------------------------------------------------------

def generate_fmow_caption(
    category: str,
    metadata: Dict[str, Any],
    drop_pct: float = 0.03,
    return_text: bool = True,
) -> str:
    """Generate a text caption for an FMoW satellite image.

    Follows DiffusionSat caption format with random field dropping
    for classifier-free guidance during training.
    Enriched with per-category structural descriptors.

    NOTE: Used as fallback when no rich VLM caption is available on disk.

    Args:
        category: FMoW category string (e.g., "airport").
        metadata: JSON metadata dictionary.
        drop_pct: probability of dropping each optional field.
        return_text: always True for our pipeline.
    Returns:
        Caption string.
    """
    str_incl = lambda x: x if random.random() > drop_pct else ""

    # Clean category name
    cls_name = " ".join(category.split("_"))

    # Extract fields
    gsd = metadata.get("gsd", None)
    cloud_cover = metadata.get("cloud_cover", None)
    country_code = metadata.get("country_code", "")
    country = COUNTRY_CODE_MAP.get(country_code, country_code)

    # Get structural descriptor for this category
    descriptor = CATEGORY_DESCRIPTORS.get(category, "")

    # Build enriched caption
    caption = (
        f"a{str_incl(' fmow')} satellite image"
        f"{str_incl(f' of a {cls_name}')}"
        f"{str_incl(f' in {country}') if country else ''}"
        f"{str_incl(f'. {descriptor}') if descriptor else ''}"
    )

    return caption


def generate_fmow_caption_with_metadata(
    category: str,
    metadata: Dict[str, Any],
    normalized_metadata: Tensor,
    drop_pct: float = 0.1,
) -> str:
    """Generate caption with embedded metadata text (text_metadata mode).

    Used when metadata is provided as text rather than numeric conditioning.
    """
    caption = generate_fmow_caption(category, metadata, drop_pct)

    lon = normalized_metadata[0].item()
    lat = normalized_metadata[1].item()
    gsd = normalized_metadata[2].item()
    year = normalized_metadata[4].item()
    month = normalized_metadata[5].item()
    day = normalized_metadata[6].item()

    caption += f" at a resolution of {round(gsd, 5)}."
    caption += f" The longitude, latitude is {lon}, {lat}."
    if year != 0.0 and month != 0.0 and day != 0.0:
        caption += f" The date is {year}, {month}, {day}"

    return caption


# ---------------------------------------------------------------------------
# Metadata Extraction and Normalization
# ---------------------------------------------------------------------------

def extract_fmow_metadata(
    metadata: Dict[str, Any],
    img_h: int,
    img_w: int,
    target_resolution: int = 512,
    base_year: int = 1980,
    base_lon: float = 180.0,
    base_lat: float = 90.0,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
) -> Tensor:
    """Extract numerical metadata vector from FMoW JSON metadata.

    Args:
        metadata: parsed JSON metadata dict.
        img_h, img_w: original image dimensions.
        target_resolution: training resolution (for GSD scaling).
        base_year, base_lon, base_lat: offset bases.
        lon, lat: geographic coordinates (from bounding box or polygon).
    Returns:
        (7,) tensor: [lon+base, lat+base, gsd, cloud_cover, year, month, day]
    """
    # GSD scaling
    orig_res = min(img_h, img_w)
    scale = orig_res / target_resolution
    gsd = metadata.get("gsd", 1.0) * scale

    # Cloud cover
    cloud_cover = metadata.get("cloud_cover", 0.0)
    if cloud_cover is not None:
        cloud_cover = cloud_cover / 100.0
    else:
        cloud_cover = 0.0

    # Timestamp
    timestamp = metadata.get("timestamp", "2000-01-01T00:00:00")
    try:
        year = int(timestamp[:4]) - base_year
        month = int(timestamp[5:7])
        day = int(timestamp[8:10])
    except (ValueError, IndexError):
        year, month, day = 0, 0, 0

    # Coordinates
    if lon is None or lat is None:
        # Coordinates should have been extracted by the dataset loader
        # using CSV lookup or JSON parsing. If we get here with None,
        # it means no coordinates were available.
        lon = 0.0
        lat = 0.0

    return torch.tensor([
        lon + base_lon,
        lat + base_lat,
        gsd,
        cloud_cover,
        float(year),
        float(month),
        float(day),
    ])


def metadata_normalize(
    metadata: Tensor,
    base_lon: float = 180.0,
    base_lat: float = 90.0,
    base_year: int = 1980,
    max_gsd: float = 1.0,
    scale: float = 1000.0,
) -> Tensor:
    """Normalize metadata to roughly [0, scale] range.

    Compatible with DiffusionSat's normalization scheme.

    Args:
        metadata: (7,) raw metadata tensor.
        base_lon, base_lat, base_year: offset bases used during extraction.
        max_gsd: maximum expected GSD for normalization.
        scale: scaling factor.
    Returns:
        (7,) normalized metadata tensor.
    """
    lon, lat, gsd, cloud_cover, year, month, day = metadata

    lon_norm = lon / (180.0 + base_lon) * scale
    lat_norm = lat / (90.0 + base_lat) * scale
    gsd_norm = gsd / max_gsd * scale
    cc_norm = cloud_cover * scale
    year_norm = year / (2100.0 - base_year) * scale
    month_norm = month / 12.0 * scale
    day_norm = day / 31.0 * scale

    return torch.tensor([lon_norm, lat_norm, gsd_norm, cc_norm, year_norm, month_norm, day_norm])


def metadata_unnormalize(
    norm_metadata: Tensor,
    base_lon: float = 180.0,
    base_lat: float = 90.0,
    base_year: int = 1980,
    max_gsd: float = 1.0,
    scale: float = 1000.0,
    to_real_coords: bool = False,
) -> Tensor:
    """Unnormalize metadata back to original scale.

    Args:
        norm_metadata: (7,) or (B, 7) normalized metadata.
        to_real_coords: if True, subtract base offsets to get real lon/lat.
    Returns:
        unnormalized metadata tensor.
    """
    lon, lat, gsd, cloud_cover, year, month, day = (
        norm_metadata[..., i] for i in range(7)
    )

    lon = lon / scale * (180.0 + base_lon)
    lat = lat / scale * (90.0 + base_lat)
    gsd = gsd / scale * max_gsd
    cloud_cover = cloud_cover / scale
    year = year / scale * (2100.0 - base_year)
    month = month / scale * 12.0
    day = day / scale * 31.0

    if to_real_coords:
        lon = lon - base_lon
        lat = lat - base_lat
        year = year + base_year

    return torch.stack([lon, lat, gsd, cloud_cover, year, month, day], dim=-1)


# ---------------------------------------------------------------------------
# Image Normalization
# ---------------------------------------------------------------------------

def percentile_normalization(
    img: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    axis=None,
) -> np.ndarray:
    """Apply percentile normalization to an image.

    Rescales values so that values <= lower percentile become 0
    and values >= upper percentile become 1.

    Args:
        img: input image array.
        lower: lower percentile [0, 100].
        upper: upper percentile [0, 100].
        axis: axis along which to compute percentiles.
    Returns:
        normalized image in [0, 1].
    """
    assert lower < upper
    lower_p = np.percentile(img, lower, axis=axis)
    upper_p = np.percentile(img, upper, axis=axis)
    return np.clip(
        (img - lower_p) / (upper_p - lower_p + 1e-5), 0, 1
    ).astype(np.float32)


def normalize_to_minus1_plus1(img: np.ndarray) -> np.ndarray:
    """Normalize image from [0, 255] or [0, 1] to [-1, 1]."""
    if img.max() > 1.0:
        img = img / 255.0
    return (img * 2.0 - 1.0).astype(np.float32)