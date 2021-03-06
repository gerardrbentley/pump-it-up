# Waterpoint Data

From [DrivenData Problem Description](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/)

## Features

The goal is to predict the operating condition of a waterpoint for each record in the dataset.
The following set of features are available for each point:

- `amount_tsh`: Total static head (amount water available to waterpoint)
- `date_recorded`: The date the row was entered
- `funder`: Who funded the well
- `gps_height`: Altitude of the well
- `installer`: Organization that installed the well
- `longitude`: GPS coordinate
- `latitude`: GPS coordinate
- `wpt_name`: Name of the waterpoint if there is one
- `num_private`:
- `basin`: Geographic water basin
- `subvillage`: Geographic location
- `region`: Geographic location
- `region_code`: Geographic location (coded)
- `district_code`: Geographic location (coded)
- `lga`: Geographic location
- `ward`: Geographic location
- `population`: Population around the well
- `public_meeting`: True/False
- `recorded_by`: Group entering this row of data
- `scheme_management`: Who operates the waterpoint
- `scheme_name`: Who operates the waterpoint
- `permit`: If the waterpoint is permitted
- `construction_year`: Year the waterpoint was constructed
- `extraction_type`: The kind of extraction the waterpoint uses
- `extraction_type_group`: The kind of extraction the waterpoint uses
- `extraction_type_class`: The kind of extraction the waterpoint uses
- `management`: How the waterpoint is managed
- `management_group`: How the waterpoint is managed
- `payment`: What the water costs
- `payment_type`: What the water costs
- `water_quality`: The quality of the water
- `quality_group`: The quality of the water
- `quantity`: The quantity of water
- `quantity_group`: The quantity of water
- `source`: The source of the water
- `source_type`: The source of the water
- `source_class`: The source of the water
- `waterpoint_type`: The kind of waterpoint
- `waterpoint_type_group`: The kind of waterpoint

## Labels

- `functional`: The waterpoint is operational and there are no repairs needed
- `functional needs repair`: The waterpoint is operational, but needs repairs
- `non functional`: The waterpoint is not operational

## Output Format

The format for the submission file is the row id of the input and the predicted label

## Performance Metric

The output is scored with classification rate, the percentage of correctly assigned labels to the test set.
The goal is to maximize this score.
Being a percentage, it can be from 0.0 to 1.0
