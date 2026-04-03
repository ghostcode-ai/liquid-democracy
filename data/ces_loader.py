"""CES (Cooperative Election Study) 2024 data loader.

Downloads the 2024 CES Common Content CSV from Harvard Dataverse
and parses it into agent-ready profiles by congressional district.

Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/X11EP6
License: Public, no authentication required.

The CSV is ~184MB. Downloaded once and cached to data/raw/.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from agents.voter_agent import Demographics, PartyID

# ---------------------------------------------------------------------------
# Harvard Dataverse download URL (file ID for the CSV)
# ---------------------------------------------------------------------------

CES_CSV_FILE_ID = "12050325"
CES_DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{CES_CSV_FILE_ID}"
CES_FILENAME = "CCES24_Common_OUTPUT_vv_topost_final.csv"

DATA_DIR = Path(__file__).parent / "raw"


@dataclass
class CESRespondent:
    """A single CES survey respondent, parsed into simulation-ready form."""

    case_id: str
    district: str  # e.g., "CA-27"
    state: str

    # Demographics
    age: int
    gender: str
    race: str
    education: int  # mapped to 0-5 scale
    income: float  # midpoint of bracket
    urban_rural: str  # "urban", "suburban", "rural"

    # Political
    party_id_7: int  # 1-7 scale (1=Strong D, 7=Strong R)
    ideology_5: int  # 1-5 (1=Very Liberal, 5=Very Conservative)
    voted: bool


# ---------------------------------------------------------------------------
# Column mappings (CES 2024 Common Content)
# ---------------------------------------------------------------------------

# pid7: 1=Strong Democrat, 2=Not very strong Democrat, 3=Lean Democrat,
#        4=Independent, 5=Lean Republican, 6=Not very strong Republican,
#        7=Strong Republican
PID7_TO_PARTY = {
    1: PartyID.STRONG_D,
    2: PartyID.LEAN_D,
    3: PartyID.LEAN_D,
    4: PartyID.INDEPENDENT,
    5: PartyID.LEAN_R,
    6: PartyID.LEAN_R,
    7: PartyID.STRONG_R,
}

# ideo5: 1=Very Liberal, 2=Liberal, 3=Moderate, 4=Conservative, 5=Very Conservative
IDEO5_TO_FLOAT = {
    1: -0.8,
    2: -0.4,
    3: 0.0,
    4: 0.4,
    5: 0.8,
}

# educ: map CES education codes to 0-5 scale
# CES: 1=No HS, 2=HS grad, 3=Some college, 4=2-year, 5=4-year, 6=Post-grad
EDUC_MAP = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 5}

# faminc: family income bracket midpoints (approximate)
INCOME_MIDPOINTS = {
    1: 5_000, 2: 15_000, 3: 25_000, 4: 35_000, 5: 45_000,
    6: 55_000, 7: 65_000, 8: 75_000, 9: 90_000, 10: 125_000,
    11: 175_000, 12: 250_000, 13: 350_000, 14: 500_000,
    15: 750_000, 16: 1_000_000,
    97: 60_000,  # prefer not to say -> median
}

# race: CES codes
RACE_MAP = {
    1: "white", 2: "black", 3: "hispanic", 4: "asian",
    5: "other", 6: "other", 7: "other", 8: "other",
}

# gender: CES codes
GENDER_MAP = {1: "male", 2: "female"}

# urbanicity: Pew classification
URBAN_MAP = {1: "urban", 2: "suburban", 3: "rural"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_ces(dest_dir: Path | None = None, force: bool = False) -> Path:
    """Download the CES 2024 CSV from Harvard Dataverse.

    Caches to data/raw/. Returns the path to the CSV file.
    Raises RuntimeError if download fails.
    """
    if dest_dir is None:
        dest_dir = DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / CES_FILENAME
    if csv_path.exists() and not force:
        size_mb = csv_path.stat().st_size / 1_048_576
        print(f"CES data already cached ({size_mb:.0f} MB): {csv_path}")
        return csv_path

    print(f"Downloading CES 2024 Common Content (~184 MB)...")
    print(f"Source: {CES_DOWNLOAD_URL}")

    import shutil
    import urllib.request

    try:
        # Dataverse requires a User-Agent header or returns 403
        req = urllib.request.Request(
            CES_DOWNLOAD_URL,
            headers={"User-Agent": "liquid-democracy-sim/0.1"},
        )
        with urllib.request.urlopen(req) as resp, open(csv_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        # Clean up partial download
        if csv_path.exists():
            csv_path.unlink()
        raise RuntimeError(
            f"Failed to download CES data: {e}\n"
            f"You can manually download from:\n"
            f"  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/X11EP6\n"
            f"Place the CSV in: {dest_dir}/"
        ) from e

    size_mb = csv_path.stat().st_size / 1_048_576
    print(f"Downloaded {size_mb:.0f} MB to {csv_path}")
    return csv_path


def is_ces_available() -> bool:
    """Check if the CES CSV is already downloaded."""
    return (DATA_DIR / CES_FILENAME).exists()


def list_available_districts() -> list[str]:
    """Return sorted list of district IDs available in the CES data.

    Caches the result to a small text file so subsequent calls are instant.
    Returns an empty list if the CES CSV hasn't been downloaded yet.
    """
    cache_file = DATA_DIR / "ces_districts.txt"
    if cache_file.exists():
        return cache_file.read_text().strip().split("\n")

    csv_path = DATA_DIR / CES_FILENAME
    if not csv_path.exists():
        return []

    districts: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state_fips = _safe_int(row.get("inputstate", ""))
            dist_num = _safe_int(row.get("cdid119", row.get("cdid118", "")))
            cd = _build_district_id(state_fips, dist_num)
            if cd:
                districts.add(cd)

    result = sorted(districts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("\n".join(result))
    return result


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}

# States with a single at-large district
AT_LARGE_STATES = {"AK", "DE", "MT", "ND", "SD", "VT", "WY"}


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _build_district_id(state_fips: int, district_num: int) -> str | None:
    """Convert FIPS state code + district number to 'XX-NN' format."""
    state = FIPS_TO_STATE.get(state_fips)
    if state is None:
        return None
    if state in AT_LARGE_STATES or district_num == 0:
        return f"{state}-AL"
    return f"{state}-{district_num:02d}"


def parse_ces_csv(csv_path: Path | None = None) -> list[CESRespondent]:
    """Parse the CES CSV into a list of CESRespondent objects.

    Uses inputstate (FIPS code) + cdid119 (119th Congress district number)
    to construct district IDs like 'PA-07', 'CA-27'.
    """
    if csv_path is None:
        csv_path = DATA_DIR / CES_FILENAME

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CES data not found at {csv_path}. "
            f"Run `from data.ces_loader import download_ces; download_ces()` first."
        )

    respondents = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Build district ID from FIPS state + district number
            state_fips = _safe_int(row.get("inputstate", ""))
            dist_num = _safe_int(row.get("cdid119", row.get("cdid118", "")))
            cd = _build_district_id(state_fips, dist_num)
            if cd is None:
                continue

            # Core political variables
            pid7 = _safe_int(row.get("pid7", ""))
            ideo5 = _safe_int(row.get("ideo5", ""))
            if pid7 < 1 or pid7 > 7:
                continue
            if ideo5 < 1 or ideo5 > 5:
                ideo5 = 3  # default to moderate if missing

            # Demographics
            birthyr = _safe_int(row.get("birthyr", ""))
            age = (2024 - birthyr) if 1920 < birthyr < 2010 else 45
            age = max(18, min(95, age))

            gender_code = _safe_int(row.get("gender4", "2"))
            gender = GENDER_MAP.get(min(gender_code, 2), "female")  # gender4 has 4 codes; map 1/2

            race_code = _safe_int(row.get("race", "1"))
            race = RACE_MAP.get(race_code, "white")

            educ_code = _safe_int(row.get("educ", "3"))
            education = EDUC_MAP.get(educ_code, 2)

            faminc_code = _safe_int(row.get("faminc_new", "7"))
            income = INCOME_MIDPOINTS.get(faminc_code, 60_000)

            urban_code = _safe_int(row.get("urbancity", "2"))
            urban_rural = URBAN_MAP.get(urban_code, "suburban")

            # Validated vote
            vv = row.get("vv_turnout_gvm", "")
            voted = vv in ("1", "Voted")

            state_abbr = FIPS_TO_STATE.get(state_fips, "")

            respondents.append(CESRespondent(
                case_id=row.get("caseid", ""),
                district=cd,
                state=state_abbr,
                age=age,
                gender=gender,
                race=race,
                education=education,
                income=float(income),
                urban_rural=urban_rural,
                party_id_7=pid7,
                ideology_5=ideo5,
                voted=voted,
            ))

    return respondents


def get_district_respondents(
    respondents: list[CESRespondent], district_id: str
) -> list[CESRespondent]:
    """Filter respondents to a specific congressional district."""
    # Normalize: CES might use "CA-27" or "CA27" or "6-27"
    target = district_id.upper().replace(" ", "")
    return [r for r in respondents if r.district.upper().replace(" ", "") == target]


def respondent_to_agent_params(
    resp: CESRespondent, rng: np.random.Generator
) -> tuple[Demographics, PartyID, np.ndarray]:
    """Convert a CES respondent into VoterAgent construction parameters.

    Returns (demographics, party_id, ideology_vector).
    """
    demographics = Demographics(
        age=resp.age,
        income=resp.income,
        education=resp.education,
        race=resp.race,
        gender=resp.gender,
        urban_rural=resp.urban_rural,
    )

    party_id = PID7_TO_PARTY.get(resp.party_id_7, PartyID.INDEPENDENT)

    # Build 10-D ideology from the 5-point scale + noise
    base = IDEO5_TO_FLOAT.get(resp.ideology_5, 0.0)
    ideology = np.full(10, base) + rng.normal(0, 0.15, size=10)
    ideology = np.clip(ideology, -1.0, 1.0)

    return demographics, party_id, ideology


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def seed_agents_from_ces(
    district_id: str,
    n_agents: int = 10_000,
    seed: Optional[int] = None,
    csv_path: Path | None = None,
) -> dict:
    """Seed agents from real CES survey respondents for a district.

    If the district has >= n_agents respondents, samples exactly n_agents.
    If fewer, resamples with replacement to reach n_agents.

    Returns {agent_id: VoterAgent}.
    """
    from agents.voter_agent import VoterAgent

    rng = np.random.default_rng(seed)

    # Auto-download CES data if not cached
    if csv_path is None and not is_ces_available():
        download_ces()

    respondents = parse_ces_csv(csv_path)
    district_resps = get_district_respondents(respondents, district_id)

    if not district_resps:
        raise ValueError(
            f"No CES respondents found for district '{district_id}'. "
            f"The CES has ~100 respondents per district; check the district ID format."
        )

    # Resample to n_agents
    indices = rng.choice(len(district_resps), size=n_agents, replace=True)

    agents = {}
    for i, idx in enumerate(indices):
        resp = district_resps[idx]
        demographics, party_id, ideology = respondent_to_agent_params(resp, rng)
        agent = VoterAgent.from_profile(
            agent_id=i,
            demographics=demographics,
            ideology=ideology,
            party_id=party_id,
        )
        agents[i] = agent

    return agents
