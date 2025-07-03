# tcia-clinical-validator
 A streamlit validator to ensure Common Data Element compliance for submitting clinical data to TCIA.

Link: https://tcia-clinical-validator-boyp2sjsdzdgawrqppupip.streamlit.app/ 

# TCIA Clinical Data Validator: Application Documentation

## 1. Introduction

The TCIA Clinical Data Validator is a Streamlit-based web application designed to help users prepare their clinical data for submission to The Cancer Imaging Archive (TCIA). It guides users through a multi-step process to ensure clinical data conforms to the National Cancer Institute's Cancer Data Standards Registry and Repository (caDSR) standards. This includes critical steps such as intelligent column mapping, robust data cleaning, and comprehensive validation of both categorical and numerical values. The application's primary goal is to streamline the data submission process by proactively identifying and assisting in the correction of common data quality issues, thereby ensuring clinical data is formatted to an NIH-recognized standard and adheres to organizational principles.

## 2. Core Functionality

The application is structured into a series of steps, each focusing on a specific aspect of data validation and standardization.

### 2.1. Data Input & Pre-processing

**Supported File Types:** The validator accepts clinical data in common spreadsheet formats:
* **CSV (.csv)**
* **Excel (.xlsx):** Supports single or multiple sheets. If multiple sheets are detected, the user can select the primary sheet for validation and choose to include other sheets in the final output.
* **Tab-Separated Values (.tsv)**

**Input Methods:** Users can upload files directly or provide a URL to a publicly accessible file.

**Initial Cleaning:** Upon loading, the application automatically performs initial data cleaning, including:
* Removing leading and trailing whitespace from all string values.
* Converting all non-age-related columns to string type for consistent text processing.
* Automatic capitalization correction for known column names (e.g., 'race' will be corrected to 'Race').
* Removal of duplicate rows within the dataset.

### 2.2. Column Mapping and Management

**Automatic Column Renaming:** The validator attempts to automatically correct the capitalization of column headers to match TCIA's allowable column names. For example, if a column is named "race" (lowercase), it will be automatically renamed to "Race".

**Identification of Unexpected Columns:** Any columns in the uploaded data that do not match the predefined list of allowable TCIA columns (even after capitalization correction) are flagged as "unexpected".

**User-Guided Column Actions:** For each unexpected column, users are provided with options:
* **Map to an Allowable Column:** Select a correct TCIA column name from a dropdown list.
* **Leave Unmodified:** Keep the column as is, without renaming.
* **Delete Column:** Remove the column entirely from the dataset.

**Required Columns:** The application checks for the presence of 'Project Short Name' and 'Case ID', which are mandatory for TCIA submissions. If 'Case ID' is missing, the user cannot proceed. If 'Project Short Name' is missing or contains invalid entries, the user will be prompted to provide a valid project short name.

### 2.3. Categorical and Numeric Data Validation

The validator checks specific columns against predefined permissible values and data types.

**Race, Ethnicity, Sex at Birth Validation:**
* **Permissible Values:** These columns are validated against fixed lists of acceptable values (e.g., "American Indian or Alaska Native", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White", "More Than One Race", "Unknown", "Not Reported" for Race; similar lists for Ethnicity and Sex at Birth).
* **Automatic Capitalization Correction:** Similar to column names, the application attempts to correct capitalization within these columns (e.g., "white" will be corrected to "White").
* **User Correction for Invalid Entries:** If values do not match permissible options, the user is prompted to manually select the correct value from a dropdown of valid options. For the 'Race' column, multiple selections are allowed since multiple race values can be semicolon-separated.

**Age Data Validation:**
* **Target Columns:** 'Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', and 'Age at Imaging' are checked.
* **Numeric Check:** Ensures that values in these columns are numeric.
* **Null/Empty Value Handling:** Empty or null values are identified and reported, along with their row indices, for user review.
* **User Correction for Non-Numeric Values:** For non-numeric entries, the user is prompted to input a valid numeric correction.

**Age Unit of Measure (Age UOM):**
* **Required if Age Columns Present:** If any age-related columns are in the data but 'Age UOM' is missing, the user is prompted to select a default unit (Day, Month, or Year).
* **Validation against Permissible Units:** Values in the 'Age UOM' column are validated against 'Day', 'Month', and 'Year'.
* **Future Conversion to Years:** (Still working on this feature)

**Primary Diagnosis and Primary Site Validation:**
* **External Data Sources:** Permissible values for these columns are loaded from external Excel files, ensuring compliance with specific ontologies (caDSR).
* **Similarity-Based Suggestions:** For invalid entries, the application provides a prioritized list of suggestions from the permissible values using string similarity algorithms (e.g., Levenshtein distance, word matching, acronyms). This helps users quickly find the correct term.
* **User Mapping:** Users can select the correct term from the suggestions or choose to keep the current value.

## 3. Output & Export

**Standardized Data Download:** After all validation and correction steps are completed, the user can download the processed DataFrame as an Excel (.xlsx) file.

**Customizable Filename:** The application suggests a default filename based on the 'Project Short Name' but allows the user to customize it.

**Column Reordering:** The final output DataFrame's columns are reordered to a preferred TCIA standard sequence for consistency.

**Preservation of Other Sheets:** If the original input was a multi-sheet Excel file and the user chose to keep other sheets, these sheets are included in the downloaded Excel file alongside the 'Standardized Data' sheet.

## 4. User Interface (UI) & Experience (UX)

**Step-by-Step Workflow:** The application guides the user through the validation process via a clear, sequential step mechanism, with progress indicated by changing subheaders.

**Session State Management:** Streamlit's session state is used to maintain data and user selections across reruns, providing a continuous user experience. A "Restart" button is available to clear all session data and start fresh.

**Dynamic Logo:** The TCIA logo displayed in the header dynamically adjusts based on the user's system theme (dark/light mode).

**Informative Messaging:** The application provides clear success, info, warning, and error messages to inform the user about data issues, automatic corrections, and required actions.

**Interactive Widgets:** Dropdowns, text inputs, multiselect boxes, and buttons are used to enable user interaction and correction of data.

## 5. Data Handling and Privacy

**In-Memory Processing:** The application processes data primarily in memory (within the Streamlit session) and does not store user data persistently on a server.

**Local Download:** The final standardized data is downloaded directly to the user's local machine.

## 6. Screenshots of Process

*(image syntax: `![Screenshot Description](path/to/image.png)`)*

*This is being worked on right now.

First we boot up the app that contains the clinical validation and your screen should look like the screenshot bellow.

<img width="802" alt="Screenshot 2025-06-27 at 9 06 34 AM" src="https://github.com/user-attachments/assets/79d701b3-3c3f-4011-91eb-f68a7f10fc89" />

At the top bar you may add any clinical file within the xlsx, csv, or tsv file format:

<img width="802" alt="Screenshot 2025-06-27 at 9 07 04 AM" src="https://github.com/user-attachments/assets/a896aa33-ed0d-4aeb-8c6a-caad963dff05" />

You may choose the Data Entry to fit the reading of the data but after uploading the documents you may be brought to this mapping column page:

<img width="771" alt="Screenshot 2025-06-27 at 9 10 47 AM" src="https://github.com/user-attachments/assets/8456c3b8-5c19-4226-b123-afd760e33222" />

You
