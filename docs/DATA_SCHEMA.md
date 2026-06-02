# Data Schema

GovTrust-FL uses a canonical municipal service-request schema after city-specific harmonization.

| Column | Description |
| --- | --- |
| `request_id` | City-provided request identifier. |
| `created_date` | Request creation timestamp. |
| `closed_date` | Request closure timestamp, required for delayed-resolution experiments. |
| `status` | Request status before cleaning. |
| `category` | Complaint or service-request category. |
| `descriptor` | More specific request descriptor or text label. |
| `agency` | Responsible agency or department when available. |
| `latitude` | Request latitude. |
| `longitude` | Request longitude. |
| `area` | Borough, ward, neighborhood, community area, or similar geography. |
| `city` | Canonical city key: `nyc`, `chicago`, `boston`, or `los_angeles`. |

Post-cleaning and labeling add:

| Column | Description |
| --- | --- |
| `resolution_hours` | Difference between `closed_date` and `created_date` in hours. |
| `delay_threshold_hours` | City-category Q3 threshold for `resolution_hours`. |
| `delayed` | Main binary target: `1` when `resolution_hours > delay_threshold_hours`. |
| `service_routing_target` | Secondary routing target from agency, with category fallback by default. |

Leakage-sensitive columns such as `closed_date`, `resolution_hours`, `delay_threshold_hours`, and final status must not be used as creation-time model features.
