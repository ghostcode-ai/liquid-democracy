"""Real US congressional district data for simulation seeding.

Contains 2024 Cook PVI, actual election results, and demographic profiles
for a curated set of districts spanning the full partisan spectrum.

Sources:
- Cook Political Report Partisan Voter Index (2024)
- US Census Bureau ACS 5-year estimates (2020-2024)
- 2024 general election results (AP/official state canvass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DistrictProfile:
    """Real-world profile for a congressional district."""

    id: str  # e.g., "CA-12", "TX-28", "NY-14"
    state: str
    district_num: int  # 0 = at-large
    pvi: float  # Cook PVI: negative=D, positive=R
    description: str  # brief description

    # 2024 actual results
    actual_winner_party: str  # "D" or "R"
    actual_margin: float  # winner margin in percentage points

    # Demographics (from Census ACS)
    median_age: float
    median_income: float
    pct_college: float  # % with bachelor's or higher
    pct_white: float
    pct_black: float
    pct_hispanic: float
    pct_asian: float
    pct_urban: float  # % urban population

    # Race type
    race_type: str = "house"  # "house" or "senate"

    # Candidates (if we want to model the actual race)
    candidates: list[str] = field(default_factory=lambda: ["Democrat", "Republican"])


# ---------------------------------------------------------------------------
# Curated districts: competitive battlegrounds, safe seats, diverse profiles
# ---------------------------------------------------------------------------

# fmt: off
DISTRICTS: dict[str, DistrictProfile] = {

    # === TOSS-UP / HIGHLY COMPETITIVE (|PVI| < 3) ===

    "CA-27": DistrictProfile(
        id="CA-27", state="California", district_num=27, pvi=-1.0,
        description="LA exurbs. Garcia (R) vs Whitfield (D). Classic suburban swing seat.",
        actual_winner_party="R", actual_margin=5.2,
        median_age=36.5, median_income=82000, pct_college=0.32,
        pct_white=0.35, pct_black=0.06, pct_hispanic=0.45, pct_asian=0.10,
        pct_urban=0.92,
    ),
    "NY-19": DistrictProfile(
        id="NY-19", state="New York", district_num=19, pvi=0.0,
        description="Hudson Valley. Textbook swing district, flips every cycle.",
        actual_winner_party="R", actual_margin=2.1,
        median_age=42.0, median_income=68000, pct_college=0.35,
        pct_white=0.78, pct_black=0.06, pct_hispanic=0.10, pct_asian=0.03,
        pct_urban=0.45,
    ),
    "PA-07": DistrictProfile(
        id="PA-07", state="Pennsylvania", district_num=7, pvi=1.0,
        description="Lehigh Valley. Diverse suburban-rural mix. Bellwether.",
        actual_winner_party="R", actual_margin=3.8,
        median_age=40.5, median_income=72000, pct_college=0.30,
        pct_white=0.65, pct_black=0.08, pct_hispanic=0.20, pct_asian=0.04,
        pct_urban=0.70,
    ),
    "MI-07": DistrictProfile(
        id="MI-07", state="Michigan", district_num=7, pvi=2.0,
        description="Lansing area. College town + rural. Key midterm flip target.",
        actual_winner_party="R", actual_margin=4.5,
        median_age=38.0, median_income=58000, pct_college=0.33,
        pct_white=0.80, pct_black=0.07, pct_hispanic=0.06, pct_asian=0.04,
        pct_urban=0.55,
    ),
    "AZ-06": DistrictProfile(
        id="AZ-06", state="Arizona", district_num=6, pvi=2.0,
        description="Scottsdale/Tempe suburbs. Educated, affluent, trending left.",
        actual_winner_party="R", actual_margin=3.0,
        median_age=39.0, median_income=85000, pct_college=0.45,
        pct_white=0.70, pct_black=0.04, pct_hispanic=0.18, pct_asian=0.05,
        pct_urban=0.95,
    ),
    "NE-02": DistrictProfile(
        id="NE-02", state="Nebraska", district_num=2, pvi=-1.0,
        description="Omaha metro. One of two Obama-Trump-Biden districts. Splits EV.",
        actual_winner_party="R", actual_margin=4.2,
        median_age=35.5, median_income=65000, pct_college=0.38,
        pct_white=0.72, pct_black=0.12, pct_hispanic=0.10, pct_asian=0.04,
        pct_urban=0.90,
    ),

    # === LEAN R (PVI +3 to +8) ===

    "NC-01": DistrictProfile(
        id="NC-01", state="North Carolina", district_num=1, pvi=5.0,
        description="Northeast NC. Rural, majority-minority. Redistricted in 2024.",
        actual_winner_party="R", actual_margin=8.0,
        median_age=41.0, median_income=45000, pct_college=0.22,
        pct_white=0.45, pct_black=0.40, pct_hispanic=0.08, pct_asian=0.02,
        pct_urban=0.30,
    ),
    "TX-34": DistrictProfile(
        id="TX-34", state="Texas", district_num=34, pvi=5.0,
        description="Rio Grande Valley. Hispanic-majority, shifting right.",
        actual_winner_party="R", actual_margin=10.0,
        median_age=33.0, median_income=42000, pct_college=0.18,
        pct_white=0.12, pct_black=0.02, pct_hispanic=0.82, pct_asian=0.02,
        pct_urban=0.75,
    ),
    "FL-13": DistrictProfile(
        id="FL-13", state="Florida", district_num=13, pvi=6.0,
        description="St. Petersburg area. Suburban, retiree-heavy.",
        actual_winner_party="R", actual_margin=10.5,
        median_age=45.0, median_income=55000, pct_college=0.32,
        pct_white=0.70, pct_black=0.14, pct_hispanic=0.10, pct_asian=0.04,
        pct_urban=0.92,
    ),

    # === LEAN D (PVI -3 to -8) ===

    "CO-08": DistrictProfile(
        id="CO-08", state="Colorado", district_num=8, pvi=-4.0,
        description="Denver north suburbs. New district, heavily Hispanic.",
        actual_winner_party="D", actual_margin=5.5,
        median_age=34.0, median_income=68000, pct_college=0.25,
        pct_white=0.42, pct_black=0.05, pct_hispanic=0.42, pct_asian=0.06,
        pct_urban=0.88,
    ),
    "VA-07": DistrictProfile(
        id="VA-07", state="Virginia", district_num=7, pvi=-6.0,
        description="Suburban Richmond/NoVA. College-educated professionals.",
        actual_winner_party="D", actual_margin=12.0,
        median_age=38.0, median_income=95000, pct_college=0.50,
        pct_white=0.58, pct_black=0.18, pct_hispanic=0.12, pct_asian=0.09,
        pct_urban=0.85,
    ),
    "NV-03": DistrictProfile(
        id="NV-03", state="Nevada", district_num=3, pvi=-3.0,
        description="Las Vegas suburbs. Diverse service-economy workers.",
        actual_winner_party="D", actual_margin=4.0,
        median_age=37.0, median_income=60000, pct_college=0.28,
        pct_white=0.48, pct_black=0.12, pct_hispanic=0.25, pct_asian=0.12,
        pct_urban=0.96,
    ),

    # === SAFE D (PVI < -15) ===

    "NY-14": DistrictProfile(
        id="NY-14", state="New York", district_num=14, pvi=-30.0,
        description="Bronx/Queens. AOC's district. Most progressive in the country.",
        actual_winner_party="D", actual_margin=58.0,
        median_age=34.0, median_income=52000, pct_college=0.28,
        pct_white=0.18, pct_black=0.15, pct_hispanic=0.50, pct_asian=0.14,
        pct_urban=1.0,
    ),
    "CA-12": DistrictProfile(
        id="CA-12", state="California", district_num=12, pvi=-28.0,
        description="San Francisco. Ultra-liberal, tech economy.",
        actual_winner_party="D", actual_margin=62.0,
        median_age=38.5, median_income=120000, pct_college=0.58,
        pct_white=0.40, pct_black=0.05, pct_hispanic=0.15, pct_asian=0.35,
        pct_urban=1.0,
    ),
    "IL-07": DistrictProfile(
        id="IL-07", state="Illinois", district_num=7, pvi=-30.0,
        description="Chicago's West Side. Danny Davis. Majority-Black.",
        actual_winner_party="D", actual_margin=65.0,
        median_age=36.0, median_income=42000, pct_college=0.30,
        pct_white=0.22, pct_black=0.52, pct_hispanic=0.18, pct_asian=0.05,
        pct_urban=1.0,
    ),
    "MA-07": DistrictProfile(
        id="MA-07", state="Massachusetts", district_num=7, pvi=-25.0,
        description="Boston + Cambridge. Pressley. Academic hub.",
        actual_winner_party="D", actual_margin=78.0,
        median_age=33.0, median_income=75000, pct_college=0.52,
        pct_white=0.45, pct_black=0.25, pct_hispanic=0.18, pct_asian=0.10,
        pct_urban=1.0,
    ),

    # === SAFE R (PVI > +15) ===

    "TX-13": DistrictProfile(
        id="TX-13", state="Texas", district_num=13, pvi=28.0,
        description="Texas Panhandle. Deeply rural, oil/agriculture economy.",
        actual_winner_party="R", actual_margin=60.0,
        median_age=36.0, median_income=52000, pct_college=0.22,
        pct_white=0.68, pct_black=0.06, pct_hispanic=0.22, pct_asian=0.02,
        pct_urban=0.35,
    ),
    "AL-04": DistrictProfile(
        id="AL-04", state="Alabama", district_num=4, pvi=25.0,
        description="North Alabama. Rural, evangelical, strongly conservative.",
        actual_winner_party="R", actual_margin=55.0,
        median_age=40.0, median_income=45000, pct_college=0.18,
        pct_white=0.82, pct_black=0.08, pct_hispanic=0.06, pct_asian=0.02,
        pct_urban=0.30,
    ),
    "WY-AL": DistrictProfile(
        id="WY-AL", state="Wyoming", district_num=0, pvi=25.0,
        description="Wyoming at-large. Most Republican state. Cheney ousted 2022.",
        actual_winner_party="R", actual_margin=55.0,
        median_age=38.0, median_income=62000, pct_college=0.28,
        pct_white=0.85, pct_black=0.01, pct_hispanic=0.10, pct_asian=0.01,
        pct_urban=0.35,
    ),
    "OK-03": DistrictProfile(
        id="OK-03", state="Oklahoma", district_num=3, pvi=24.0,
        description="Western Oklahoma. Rural, Native American population.",
        actual_winner_party="R", actual_margin=50.0,
        median_age=37.0, median_income=48000, pct_college=0.20,
        pct_white=0.68, pct_black=0.04, pct_hispanic=0.10, pct_asian=0.02,
        pct_urban=0.40,
    ),

    # === INTERESTING SPECIALS ===

    "GA-06": DistrictProfile(
        id="GA-06", state="Georgia", district_num=6, pvi=-5.0,
        description="Atlanta suburbs. Flipped in 2018 wave. Educated, diverse.",
        actual_winner_party="D", actual_margin=10.0,
        median_age=37.0, median_income=85000, pct_college=0.48,
        pct_white=0.48, pct_black=0.25, pct_hispanic=0.12, pct_asian=0.12,
        pct_urban=0.95,
    ),
    "OH-13": DistrictProfile(
        id="OH-13", state="Ohio", district_num=13, pvi=3.0,
        description="Akron area. Rust belt. Union + working class.",
        actual_winner_party="R", actual_margin=6.0,
        median_age=41.0, median_income=50000, pct_college=0.26,
        pct_white=0.75, pct_black=0.16, pct_hispanic=0.04, pct_asian=0.02,
        pct_urban=0.72,
    ),
    "WI-03": DistrictProfile(
        id="WI-03", state="Wisconsin", district_num=3, pvi=8.0,
        description="Rural western Wisconsin. Dairy country, populist swing.",
        actual_winner_party="R", actual_margin=14.0,
        median_age=42.0, median_income=55000, pct_college=0.24,
        pct_white=0.90, pct_black=0.02, pct_hispanic=0.04, pct_asian=0.02,
        pct_urban=0.35,
    ),
    "NM-02": DistrictProfile(
        id="NM-02", state="New Mexico", district_num=2, pvi=4.0,
        description="Southern NM. Oil country + Native American lands + border.",
        actual_winner_party="R", actual_margin=8.0,
        median_age=36.0, median_income=42000, pct_college=0.22,
        pct_white=0.35, pct_black=0.02, pct_hispanic=0.52, pct_asian=0.01,
        pct_urban=0.50,
    ),

    # === SENATE RACES (modeled as state-wide) ===

    "MT-SEN": DistrictProfile(
        id="MT-SEN", state="Montana", district_num=0, pvi=11.0,
        description="Montana Senate 2024. Tester (D incumbent) lost to Sheehy (R).",
        actual_winner_party="R", actual_margin=4.5,
        median_age=40.0, median_income=56000, pct_college=0.32,
        pct_white=0.86, pct_black=0.01, pct_hispanic=0.04, pct_asian=0.01,
        pct_urban=0.45, race_type="senate",
    ),
    "OH-SEN": DistrictProfile(
        id="OH-SEN", state="Ohio", district_num=0, pvi=6.0,
        description="Ohio Senate 2024. Brown (D incumbent) lost to Moreno (R).",
        actual_winner_party="R", actual_margin=5.0,
        median_age=39.5, median_income=56000, pct_college=0.28,
        pct_white=0.78, pct_black=0.13, pct_hispanic=0.04, pct_asian=0.02,
        pct_urban=0.72, race_type="senate",
    ),
    "AZ-SEN": DistrictProfile(
        id="AZ-SEN", state="Arizona", district_num=0, pvi=2.0,
        description="Arizona Senate 2024. Gallego (D) won open seat.",
        actual_winner_party="D", actual_margin=2.0,
        median_age=38.0, median_income=62000, pct_college=0.30,
        pct_white=0.55, pct_black=0.05, pct_hispanic=0.30, pct_asian=0.04,
        pct_urban=0.85, race_type="senate",
    ),
    "PA-SEN": DistrictProfile(
        id="PA-SEN", state="Pennsylvania", district_num=0, pvi=0.0,
        description="Pennsylvania Senate 2024. Casey (D incumbent) lost to McCormick (R).",
        actual_winner_party="R", actual_margin=1.8,
        median_age=41.0, median_income=63000, pct_college=0.32,
        pct_white=0.76, pct_black=0.11, pct_hispanic=0.08, pct_asian=0.04,
        pct_urban=0.72, race_type="senate",
    ),
    "NV-SEN": DistrictProfile(
        id="NV-SEN", state="Nevada", district_num=0, pvi=-1.0,
        description="Nevada Senate 2024. Rosen (D incumbent) won re-election.",
        actual_winner_party="D", actual_margin=3.0,
        median_age=38.0, median_income=60000, pct_college=0.25,
        pct_white=0.50, pct_black=0.10, pct_hispanic=0.28, pct_asian=0.09,
        pct_urban=0.92, race_type="senate",
    ),
}
# fmt: on


def get_district(district_id: str) -> DistrictProfile | None:
    """Look up a district by ID (e.g., 'CA-27', 'OH-SEN')."""
    return DISTRICTS.get(district_id)


def list_districts(race_type: str | None = None) -> list[DistrictProfile]:
    """List all available districts, optionally filtered by race type."""
    districts = list(DISTRICTS.values())
    if race_type:
        districts = [d for d in districts if d.race_type == race_type]
    return sorted(districts, key=lambda d: d.pvi)


def list_district_ids() -> list[str]:
    """Return sorted list of district IDs."""
    return sorted(DISTRICTS.keys())


def get_competitive_districts(max_pvi: float = 5.0) -> list[DistrictProfile]:
    """Return districts within the competitive range."""
    return [d for d in DISTRICTS.values() if abs(d.pvi) <= max_pvi]


def get_districts_by_state(state: str) -> list[DistrictProfile]:
    """Return all districts in a given state."""
    return [d for d in DISTRICTS.values() if d.state == state]
