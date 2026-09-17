"""Frozen official-source registry for the 2021-2025 real municipal 311 rebuild.

Every entry is a real, publicly documented municipal open-data endpoint. The
field lists are the frozen experiment fields retrieved verbatim from the
source; no value is transformed at acquisition time.
"""
from __future__ import annotations

from dataclasses import dataclass

YEARS = (2021, 2022, 2023, 2024, 2025)
CITIES = ("nyc", "chicago", "boston", "los_angeles")

SOCRATA_PAGE_SIZE = 50_000
CKAN_PAGE_SIZE = 32_000  # CKAN datastore_search hard maximum


@dataclass(frozen=True)
class SocrataSource:
    city: str
    year: int
    domain: str
    dataset_id: str
    dataset_name: str
    id_field: str
    created_field: str
    fields: tuple[str, ...]
    year_filtered: bool = True
    coverage_note: str | None = None

    @property
    def resource_url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.json"

    @property
    def landing_url(self) -> str:
        return f"https://{self.domain}/d/{self.dataset_id}"


@dataclass(frozen=True)
class CkanSource:
    city: str
    year: int
    domain: str
    resource_id: str
    dataset_name: str
    id_field: str
    created_field: str
    fields: tuple[str, ...]
    coverage_note: str | None = None

    @property
    def resource_url(self) -> str:
        return f"https://{self.domain}/api/3/action/datastore_search"

    @property
    def landing_url(self) -> str:
        return f"https://{self.domain}/datastore/dump/{self.resource_id}"


NYC_FIELDS = (
    ":id",
    "unique_key",
    "created_date",
    "closed_date",
    "status",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "borough",
    "community_board",
    "incident_zip",
    "latitude",
    "longitude",
)

CHICAGO_FIELDS = (
    ":id",
    "sr_number",
    "created_date",
    "closed_date",
    "status",
    "sr_type",
    "sr_short_code",
    "owner_department",
    "created_department",
    "community_area",
    "ward",
    "zip_code",
    "duplicate",
    "legacy_record",
    "parent_sr_number",
    "latitude",
    "longitude",
)

LA_LEGACY_FIELDS = (
    ":id",
    "srnumber",
    "createddate",
    "closeddate",
    "status",
    "requesttype",
    "actiontaken",
    "owner",
    "assignto",
    "ncname",
    "nc",
    "cd",
    "apc",
    "zipcode",
    "latitude",
    "longitude",
)

LA_2025_FIELDS = (
    ":id",
    "casenumber",
    "createddate",
    "closeddate",
    "status",
    "type",
    "action_taken__c",
    "department_name__c",
    "assigned_to__c",
    "locator_sr_neigborhood_council_1",
    "locator_sr_neigborhood_council",
    "locator_council_district",
    "locator_sr_area_planning",
    "zipcode__c",
    "geolocation__latitude__s",
    "geolocation__longitude__s",
)

BOSTON_FIELDS = (
    "_id",
    "case_enquiry_id",
    "open_dt",
    "sla_target_dt",
    "closed_dt",
    "on_time",
    "case_status",
    "closure_reason",
    "case_title",
    "subject",
    "reason",
    "type",
    "queue",
    "department",
    "neighborhood",
    "ward",
    "location_zipcode",
    "latitude",
    "longitude",
    "source",
)

LA_DATASETS = {
    2021: ("97z7-y5bt", "MyLA311 Service Request Data 2021"),
    2022: ("i5ke-k6by", "MyLA311 Service Request Data 2022"),
    2023: ("4a4x-mna2", "MyLA311 Service Request Data 2023"),
    2024: ("b7dx-7gc3", "MyLA311 Service Request Data 2024"),
    2025: ("73a2-6ar5", "MyLA311 Cases March 2025 to December 2025"),
}

BOSTON_RESOURCES = {
    2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
    2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
    2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
    2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
    2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
}

LA_2025_NOTE = (
    "Official Los Angeles 2025 source (73a2-6ar5, 'MyLA311 Cases March 2025 to "
    "December 2025') covers March-December 2025 only; January-February 2025 is not "
    "published in this dataset. The 2025 schema also differs from 2021-2024 "
    "(casenumber/type/action_taken__c/department_name__c/geolocation__*__s)."
)


def build_sources() -> list[SocrataSource | CkanSource]:
    out: list[SocrataSource | CkanSource] = []
    for year in YEARS:
        out.append(
            SocrataSource(
                city="nyc",
                year=year,
                domain="data.cityofnewyork.us",
                dataset_id="erm2-nwe9",
                dataset_name="311 Service Requests from 2020 to Present",
                id_field="unique_key",
                created_field="created_date",
                fields=NYC_FIELDS,
            )
        )
    for year in YEARS:
        out.append(
            SocrataSource(
                city="chicago",
                year=year,
                domain="data.cityofchicago.org",
                dataset_id="v6vf-nfxy",
                dataset_name="311 Service Requests",
                id_field="sr_number",
                created_field="created_date",
                fields=CHICAGO_FIELDS,
            )
        )
    for year in YEARS:
        out.append(
            CkanSource(
                city="boston",
                year=year,
                domain="data.boston.gov",
                resource_id=BOSTON_RESOURCES[year],
                dataset_name=f"311 Service Requests {year}",
                id_field="case_enquiry_id",
                created_field="open_dt",
                fields=BOSTON_FIELDS,
            )
        )
    for year in YEARS:
        dataset_id, name = LA_DATASETS[year]
        if year == 2025:
            out.append(
                SocrataSource(
                    city="los_angeles",
                    year=year,
                    domain="data.lacity.org",
                    dataset_id=dataset_id,
                    dataset_name=name,
                    id_field="casenumber",
                    created_field="createddate",
                    fields=LA_2025_FIELDS,
                    coverage_note=LA_2025_NOTE,
                )
            )
        else:
            out.append(
                SocrataSource(
                    city="los_angeles",
                    year=year,
                    domain="data.lacity.org",
                    dataset_id=dataset_id,
                    dataset_name=name,
                    id_field="srnumber",
                    created_field="createddate",
                    fields=LA_LEGACY_FIELDS,
                )
            )
    return out


SOURCES = build_sources()
SOURCE_BY_KEY = {(s.city, s.year): s for s in SOURCES}
