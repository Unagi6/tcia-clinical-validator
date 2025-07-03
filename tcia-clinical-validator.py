import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
from difflib import SequenceMatcher

st.set_page_config(page_title="TCIA Clinical Data Validator")

# Permissible columns and enumerations
required_columns = ['Project Short Name', 'Case ID']
allowable_columns = [
    'Project Short Name', 'Case ID', 'Race', 'Ethnicity', 'Sex at Birth',
    'Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery','Age UOM',
    'Primary Diagnosis', 'Primary Site'
]

def reset_session_state():
    """Reset all session state variables to their initial values"""
    # Core step tracking
    st.session_state.step = 1
    st.session_state.project_short_name = ''
    st.session_state.age_uom = ''

    # Column mapping related states
    st.session_state.columns_mapped = False
    st.session_state.column_mapping = {}
    st.session_state.mapping_applied = False

    # Categorical validation states
    if 'kept_values' in st.session_state:
        del st.session_state.kept_values
    if 'fix_column_states' in st.session_state:
        del st.session_state.fix_column_states

    # Clear validation mappings
    if 'mapping_complete' in st.session_state:
        del st.session_state.mapping_complete
    if 'value_mappings' in st.session_state:
        del st.session_state.value_mappings

    # Primary Diagnosis and Primary Site mapping states - Added explicit clearing
    if 'primary_diagnosis_mapped' in st.session_state:
        del st.session_state.primary_diagnosis_mapped
    if 'primary_diagnosis_mappings' in st.session_state:
        del st.session_state.primary_diagnosis_mappings
    if 'primary_site_mapped' in st.session_state:
        del st.session_state.primary_site_mapped
    if 'primary_site_mappings' in st.session_state:
        del st.session_state.primary_site_mappings

    # Clear the dataframe if it exists
    if 'df' in st.session_state:
        del st.session_state.df

    # Clear any skip flags or other dynamic states
    keys_to_remove = []
    for key in st.session_state.keys():
        if (key.startswith('skip_') or
            key.startswith('Race_') or
            key.startswith('Age_') or
            key.startswith('Primary_Diagnosis_') or
            key.startswith('Primary_Site_') or
            key.startswith('fix_')):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del st.session_state[key]

# Function to read and process permissible value lists for primary diagnosis and primary site
def load_permissible_values(file_path):
    try:
        df = pd.read_excel(file_path)
        # Assuming 'Permissible Value' is the column name
        values = df['Permissible Value'].dropna().unique().tolist()
        # Sort values for easier lookup
        return sorted(values)
    except Exception as e:
        st.error(f"Error loading permissible values from {file_path}: {str(e)}")
        return []

# Load permissible values at app startup
@st.cache_data
def initialize_permissible_values():
    primary_diagnosis_values = load_permissible_values('primary_diagnosis_caDSR_14905532.xlsx')
    primary_site_values = load_permissible_values('primary_site_caDSR_14883047.xlsx')
    return primary_diagnosis_values, primary_site_values

# Load the permissible values for Primary Diagnosis and Primary Site
permissible_primary_diagnosis, permissible_primary_site = initialize_permissible_values()

permissible_race = [
    "American Indian or Alaska Native", "Asian", "Black or African American",
    "Native Hawaiian or Other Pacific Islander", "Not Allowed To Collect",
    "Not Reported", "Unknown", "White"
]
permissible_ethnicity = [
    "Hispanic or Latino", "Not Allowed To Collect",
    "Not Hispanic or Latino", "Not Reported", "Unknown"
]
permissible_sex_at_birth = [
    "Don't know", "Female", "Intersex", "Male",
    "None of these describe me", "Prefer not to answer", "Unknown"
]
permissible_age_uom = ['Day', 'Month', 'Year']

# Conversion factors for Age UOM
age_uom_factors = {
    'Day': 1 / 365,
    'Month': 1 / 12,
    'Year': 1
}

# convert non-age columns to strings
def convert_to_strings(df):
    age_columns = ['Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging']
    for col in df.columns:
        if col not in age_columns:
            df[col] = df[col].astype(str)
    return df

# Helper to validate and clean data
def validate_and_clean_data(df):
    report = []

    # Convert non-age columns to strings
    df = convert_to_strings(df)

    # Track unique capitalization corrections
    capitalization_fixes = {
        'Race': set(),
        'Ethnicity': set(),
        'Sex at Birth': set()
    }

    # Handle Race column (special case for multiple values)
    if 'Race' in df.columns:
        def fix_race_values(race_str):
            if pd.isna(race_str):
                return race_str
            fixed_races = []
            for race in str(race_str).split(';'):
                race = race.strip()
                correct_race = get_correct_value(race, permissible_race)
                if correct_race and correct_race != race:
                    capitalization_fixes['Race'].add((race, correct_race))
                fixed_races.append(correct_race if correct_race else race)
            return ';'.join(sorted(set(fixed_races)))

        df['Race'] = df['Race'].apply(fix_race_values)

    # Handle Ethnicity and Sex at Birth
    for col in ['Ethnicity', 'Sex at Birth']:
        if col in df.columns:
            valid_values = permissible_ethnicity if col == 'Ethnicity' else permissible_sex_at_birth

            def fix_value(val):
                if pd.isna(val):
                    return val
                correct_val = get_correct_value(val, valid_values)
                if correct_val and correct_val != val:
                    capitalization_fixes[col].add((val, correct_val))
                return correct_val if correct_val else val

            df[col] = df[col].apply(fix_value)

    # Report unique capitalization fixes
    for col, fixes in capitalization_fixes.items():
        if fixes:
            if col == 'Race':
                fix_details = [f"'{old}' → '{new}'" for old, new in fixes]
                report.append(f"Automatically corrected capitalization of {len(fixes)} unique Race values: {', '.join(fix_details)}")
            else:
                fix_details = [f"'{old}' → '{new}'" for old, new in fixes]
                report.append(f"Automatically corrected capitalization of {len(fixes)} unique values in {col} column: {', '.join(fix_details)}")

    # Drop duplicate rows and report their original row numbers
    duplicate_rows = df[df.duplicated()].index.tolist()
    if duplicate_rows:
        df.drop_duplicates(inplace=True)
        report.append(f"Removed {len(duplicate_rows)} duplicate rows: {duplicate_rows}")

    return df, report

# Validate numeric columns (updated to handle nulls)
def validate_numeric_columns(df, numeric_columns):
    numeric_issues = {}

    for col in numeric_columns:
        if col in df.columns:
            # Convert empty strings and whitespace-only strings to NaN
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

            # Count null values (including NaN, None, and empty strings)
            null_mask = df[col].isna()
            null_count = null_mask.sum()

            # Find non-numeric values (excluding nulls)
            non_null_mask = ~null_mask
            non_null_values = df[col][non_null_mask]
            numeric_mask = pd.to_numeric(non_null_values, errors='coerce').notna()
            non_numeric = df[col][non_null_mask & ~numeric_mask]

            if null_count > 0 or len(non_numeric) > 0:
                issues = {}
                if null_count > 0:
                    issues['null_count'] = null_count
                    # Store the row indices where nulls occur
                    issues['null_indices'] = df[null_mask].index.tolist()
                if len(non_numeric) > 0:
                    issues['invalid_values'] = non_numeric
                numeric_issues[col] = issues

    return numeric_issues

def validate_categorical_column(df, column, valid_values):
    """
    Validates a categorical column, taking into account previous capitalization fixes.
    Returns a Series of boolean values where False indicates an invalid value.
    """
    def is_valid(value):
        if pd.isna(value):
            return True
        if column == 'Race':
            return all(get_correct_value(race.strip(), valid_values) is not None
                      for race in str(value).split(';'))
        return get_correct_value(value, valid_values) is not None

    return df[column].apply(is_valid)

# helper function to validate Project Short Name
def is_valid_project_short_name(name):
    return bool(re.match(r'^[a-zA-Z0-9\s_-]{1,30}$', name))

# helper function to find the correct capitalization of a column name
def get_correct_column_name(col):
    lower_allowable = {c.lower(): c for c in allowable_columns}
    return lower_allowable.get(col.lower(), col)

# helper function to get correct capitalization for categorical values
def get_correct_value(value, valid_values):
    """
    Find the correctly capitalized value from valid_values, matching case-insensitively.
    Returns None if no match is found.
    """
    lower_valid = {v.lower(): v for v in valid_values}
    return lower_valid.get(str(value).lower())

def get_prioritized_options(value, valid_options, n_suggestions=5):
    """
    Returns a prioritized list of valid options based on multiple matching strategies.

    Args:
        value (str): The input value to find matches for
        valid_options (list): List of valid options to match against
        n_suggestions (int): Number of close matches to return before remaining options

    Returns:
        list: Prioritized list of options with best matches first
    """
    def clean_string(s):
        # Convert to lowercase and remove special characters
        return re.sub(r'[^a-z0-9\s]', '', str(s).lower())

    def get_similarity_score(option):
        # Get base similarity score
        base_score = SequenceMatcher(None, clean_string(value), clean_string(option)).ratio()

        # Boost score for matches at start of words
        words_value = set(clean_string(value).split())
        words_option = set(clean_string(option).split())
        word_start_matches = sum(1 for w1 in words_value
                               for w2 in words_option
                               if w2.startswith(w1) or w1.startswith(w2))

        # Boost score for acronym matches
        value_acronym = ''.join(word[0] for word in clean_string(value).split() if word)
        option_acronym = ''.join(word[0] for word in clean_string(option).split() if word)
        acronym_match = SequenceMatcher(None, value_acronym, option_acronym).ratio()

        # Boost score for partial word matches
        shared_words = words_value.intersection(words_option)
        word_match_score = len(shared_words) / max(len(words_value), len(words_option)) if words_value else 0

        # Calculate weighted final score
        final_score = (base_score * 0.4 +  # Base string similarity
                      (word_start_matches * 0.1) +  # Word start matches
                      (acronym_match * 0.2) +  # Acronym similarity
                      (word_match_score * 0.3))  # Word match score

        return final_score

    # Score all options
    scored_options = [(option, get_similarity_score(option)) for option in valid_options]

    # Sort by score in descending order
    scored_options.sort(key=lambda x: x[1], reverse=True)

    # Get top N matches and remaining options
    top_matches = [option for option, _ in scored_options[:n_suggestions]]
    remaining_options = [option for option, _ in scored_options[n_suggestions:]]

    # Construct final list with 'Keep current value' first, then top matches, then remaining options
    return ['Keep current value'] + top_matches + remaining_options

<<<<<<< Updated upstream
# Function to reorder columns
=======

>>>>>>> Stashed changes
def reorder_columns(df):
    preferred_order = [
        'Project Short Name', 'Case ID', 'Primary Diagnosis', 'Primary Site',
        'Race', 'Ethnicity', 'Sex at Birth', 'Age UOM',
        'Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging'
    ]
    existing_columns = [col for col in preferred_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in existing_columns]
    return df[existing_columns + other_columns]

# helper function to ingest spreadsheet file to dataframe
def process_file(file_or_url, is_url=False):
    """Helper function to process uploaded files or URLs"""
    try:
        if is_url:
            file_name = file_or_url
            if not any(file_name.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.tsv']):
                st.error("URL must point to a .csv, .xlsx, or .tsv file")
                return None, False
        else:
            file_name = file_or_url.name  # Get the name from UploadedFile object

        # Initialize other_sheets as None
        other_sheets = None

        # Determine file type and read accordingly
        if file_name.lower().endswith('.csv'):
            df = pd.read_csv(file_or_url)
            proceed_to_next = True
        elif file_name.lower().endswith('.xlsx'):
            excel_file = pd.ExcelFile(file_or_url)
            sheet_names = excel_file.sheet_names
            if len(sheet_names) > 1:
                selected_tab = st.selectbox("Select Sheet to Analyze", sheet_names)
                keep_other_sheets = st.checkbox("Keep other sheets in final output", value=True)

                # Read the selected sheet
                df = pd.read_excel(file_or_url, sheet_name=selected_tab)

                # If keeping other sheets, store them
                if keep_other_sheets:
                    other_sheets = {}
                    other_sheet_names = [s for s in sheet_names if s != selected_tab]
                    for sheet in other_sheet_names:
                        other_sheets[sheet] = pd.read_excel(file_or_url, sheet_name=sheet)

                proceed_to_next = st.button("Next")
            else:
                df = pd.read_excel(file_or_url)
                proceed_to_next = True
        elif file_name.lower().endswith('.tsv'):
            df = pd.read_csv(file_or_url, delimiter='\t')
            proceed_to_next = True
        else:
            st.error("Unsupported file format. Please upload a .csv, .xlsx, or .tsv file")
            return None, False, None

        return df, proceed_to_next, other_sheets

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, False, None

# Main Streamlit app
# Custom CSS to switch logo based on the user's theme preference
st.markdown(
    """
    <style>
    @media (prefers-color-scheme: dark) {
        .logo {
            content: url(https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-02.png);
        }
    }
    @media (prefers-color-scheme: light) {
        .logo {
            content: url(https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-01.png);
        }
    }
    </style>
    <header class="main-header">
        <img class="logo" alt="App Logo">
    </header>
    """,
    unsafe_allow_html=True
)

# main column title
st.title("Clinical Data Validator")

# Initialize session state to track steps
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'project_short_name' not in st.session_state:
    st.session_state.project_short_name = ''
if 'age_uom' not in st.session_state:
    st.session_state.age_uom = ''
if 'other_sheets' not in st.session_state:
    st.session_state.other_sheets = None

# Step 1: File Upload and Import
if st.session_state.step == 1:
    st.subheader("Step 1: Upload your CSV, XLSX, or TSV file")
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "tsv"])

    if uploaded_file:
        df, proceed_to_next, other_sheets = process_file(uploaded_file)
    else:
        url = st.text_input("...or provide the URL of the file")
        if url:
            df, proceed_to_next, other_sheets = process_file(url, is_url=True)

    # Process results if we have them
    if 'df' in locals() and df is not None and proceed_to_next:
        st.success("File imported successfully!")
        # Remove leading and trailing spaces from all strings in the dataframe
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        # Store the data and other sheets in session state
        st.session_state.df = df
        st.session_state.other_sheets = other_sheets
        st.session_state.step = 2
        st.rerun()

# Step 2: Analyze and Map Columns
elif st.session_state.step == 2:
    st.subheader("Step 2: Map column names to TCIA column names")
    df = st.session_state.df

    # Clean and validate the data first
    #df, validation_report = validate_and_clean_data(df)

    # Display any cleaning operations that were performed
    #if validation_report:
    #    for message in validation_report:
    #        st.info(message)

    # Automatically correct capitalization for columns that match allowable columns
    columns_to_rename = {}
    for col in df.columns:
        # Convert column name to string to ensure it's not a UploadedFile object
        col_str = str(col)
        correct_name = get_correct_column_name(col_str)
        if correct_name != col_str:
            columns_to_rename[col_str] = correct_name

    if columns_to_rename:
        df.rename(columns=columns_to_rename, inplace=True)
        st.info(f"The following columns were automatically renamed to correct capitalization: {', '.join(columns_to_rename.values())}")

    # Find truly unexpected columns (those that don't match any allowable column, regardless of capitalization)
    unexpected_columns = [
        str(col) for col in df.columns
        if str(col).lower() not in [c.lower() for c in allowable_columns]
    ]

    # Initialize session state variables if not present
    if 'columns_mapped' not in st.session_state:
        st.session_state.columns_mapped = False
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'mapping_applied' not in st.session_state:
        st.session_state.mapping_applied = False

    if unexpected_columns:
        if not st.session_state.mapping_applied:
            st.warning("The following unexpected columns were found.")
            column_mapping = {}
            for col in unexpected_columns:
                option = st.selectbox(
                    f"How should '{col}' be mapped?",
                    allowable_columns + ["Leave unmodified", "Delete column"],
                    key=col,
                    index=len(allowable_columns)
                )
                column_mapping[col] = option

            if st.button("Apply column mapping"):
                st.session_state.column_mapping = column_mapping
                st.session_state.mapping_applied = True

                # Apply the mappings
                for col, action in column_mapping.items():
                    if action == "Delete column":
                        df.drop(columns=[col], inplace=True)
                    elif action != "Leave unmodified":
                        df.rename(columns={col: action}, inplace=True)

                st.session_state.df = df
                st.rerun()
        else:
            # Group columns by action type
            to_delete = []
            to_remain = []
            to_remap = {}

            for col, action in st.session_state.column_mapping.items():
                if action == "Delete column":
                    to_delete.append(col)
                elif action == "Leave unmodified":
                    to_remain.append(col)
                else:
                    to_remap[col] = action

            # Show summary of applied mappings grouped by action
            st.markdown("### Column Mapping Summary:")

            if to_delete:
                st.info(f"The following columns will be deleted: {', '.join(f'`{col}`' for col in to_delete)}")

            if to_remain:
                st.info(f"The following columns will remain unchanged: {', '.join(f'`{col}`' for col in to_remain)}")

            if to_remap:
                remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                st.info(f"The following columns will be remapped: {', '.join(remap_summary)}")

            st.session_state.columns_mapped = True
    else:
        # Only show success message if there are no unexpected columns
        # and automatic renaming (if any) has been completed
        st.success("All columns are correctly named or have been automatically corrected.")
        st.session_state.columns_mapped = True

    # Update the session state with the cleaned data
    st.session_state.df = df

    # Only show "Next step" button if all columns are mapped
    if st.session_state.columns_mapped:
        if st.button("Next step"):
            # Reset mapping state for if user comes back to this step
            st.session_state.mapping_applied = False
            st.session_state.column_mapping = {}
            st.session_state.step = 3
            st.rerun()

# Step 3: Check Required Columns and Validate Project Short Name
elif st.session_state.step == 3:
    st.subheader("Step 3: Check for Minimum Required Columns")
    df = st.session_state.df

    # Convert non-age columns to strings
    df = convert_to_strings(df)

    missing_case_id = 'Case ID' not in df.columns
    missing_project_short_name = 'Project Short Name' not in df.columns
    age_columns = ['Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging']
    existing_age_columns = [col for col in age_columns if col in df.columns]
    missing_age_uom = 'Age UOM' not in df.columns and existing_age_columns

    if missing_case_id:
        st.error("The 'Case ID' column is missing from your spreadsheet. This is a required column.")
        if st.button("Restart"):
            st.session_state.step = 1
            if 'df' in st.session_state:
                del st.session_state.df
            st.rerun()
    else:
        project_short_name_valid = True
        age_uom_valid = True

        # Handle Project Short Name
        name_updates = {}
        if missing_project_short_name:
            st.warning("The 'Project Short Name' column is missing from your spreadsheet.")
            project_short_name = st.text_input("Please specify a Project Short Name:", value=st.session_state.project_short_name)
            if project_short_name:
                if is_valid_project_short_name(project_short_name):
                    st.session_state.project_short_name = project_short_name
                else:
                    st.error("Invalid Project Short Name. It should be 1-30 characters long and contain only letters, numbers, dashes, and underscores.")
                    project_short_name_valid = False
            else:
                project_short_name_valid = False
        else:
            # Check existing Project Short Names
            invalid_names = df[~df['Project Short Name'].apply(is_valid_project_short_name)]['Project Short Name'].unique()
            if len(invalid_names) > 0:
                st.warning("Some Project Short Names are invalid. Please update them:")
                for name in invalid_names:
                    new_name = st.text_input(f"New name for '{name}' (1-30 characters, letters, numbers, dashes, underscores):")
                    if new_name:
                        if is_valid_project_short_name(new_name):
                            name_updates[name] = new_name
                        else:
                            st.error(f"'{new_name}' is not a valid Project Short Name.")
                            project_short_name_valid = False
                    else:
                        project_short_name_valid = False
            #else:
                #st.success("All Project Short Names are valid.")

        # Handle missing Age UOM
        if missing_age_uom:
            st.warning("The 'Age UOM' column is missing, but age-related columns are present.")
            age_uom = st.selectbox("Please select the Age Unit of Measure:", options=permissible_age_uom, index=0 if not st.session_state.age_uom else permissible_age_uom.index(st.session_state.age_uom))
            if age_uom:
                st.session_state.age_uom = age_uom
            else:
                age_uom_valid = False
        else:
            age_uom_valid = True

        st.success("Column validation successful.")

        # Single "Next step" button for both updating and continuing
        if st.button("Next step"):
            if project_short_name_valid and age_uom_valid:
                # Apply updates to Project Short Names if necessary
                if missing_project_short_name:
                    df['Project Short Name'] = st.session_state.project_short_name
                for old_name, new_name in name_updates.items():
                    df.loc[df['Project Short Name'] == old_name, 'Project Short Name'] = new_name

                # Apply Age UOM changes if necessary
                if missing_age_uom:
                    df['Age UOM'] = st.session_state.age_uom

                st.session_state.df = df
                st.success("Changes applied successfully!")
                st.session_state.step = 4
                st.rerun()
            else:
                if not project_short_name_valid:
                    st.error("Please provide a valid Project Short Name.")
                if not age_uom_valid:
                    st.error("Please select an Age Unit of Measure.")

# Step 4: Validate Race, Ethnicity and Age Data
elif st.session_state.step == 4:
    st.subheader("Step 4: Validate Race, Ethnicity, and Age Data")
    df = st.session_state.df

    # Clean and validate the data
    df, validation_report = validate_and_clean_data(df)

    # Display any cleaning operations that were performed
    if validation_report:
        for message in validation_report:
            st.info(message)

    # Dictionary to store all corrections needed for invalid values
    all_corrections = {}

    # 1. Validate Race column (after capitalization fixes)
    if 'Race' in df.columns:
        invalid_races = ~validate_categorical_column(df, 'Race', permissible_race)
        invalid_race_values = df[invalid_races]['Race'].unique()

        if len(invalid_race_values) > 0:
            st.markdown("#### Invalid Race values found (after capitalization fixes):")
            race_corrections = {}
            for value in invalid_race_values:
                st.write(f"Invalid value: '{value}'")
                correct_races = st.multiselect(
                    f"Select correct races for '{value}':",
                    options=permissible_race,
                    key=f"Race_{value}"
                )
                if correct_races:
                    race_corrections[value] = ';'.join(correct_races)

            if race_corrections:
                all_corrections['Race'] = race_corrections

    # 2. Validate other categorical columns (after capitalization fixes)
    categorical_columns = {
        'Ethnicity': permissible_ethnicity,
        'Sex at Birth': permissible_sex_at_birth,
        'Age UOM': permissible_age_uom
    }

    for col, valid_values in categorical_columns.items():
        if col in df.columns:
            invalid_mask = ~validate_categorical_column(df, col, valid_values)
            invalid_values = df[invalid_mask][col].unique()

            if len(invalid_values) > 0:
                st.markdown(f"#### Invalid {col} values found (after capitalization fixes):")
                corrections = {}
                for value in invalid_values:
                    correct_value = st.selectbox(
                        f"Correct value for '{value}' in {col}:",
                        options=valid_values,
                        key=f"{col}_{value}"
                    )
                    if correct_value:
                        corrections[value] = correct_value

                if corrections:
                    all_corrections[col] = corrections

    # 3. Validate numeric columns
    numeric_columns = ['Age at Diagnosis', 'Age at Enrollment',
                      'Age at Surgery', 'Age at Earliest Imaging']

    # Only validate numeric columns if we're not in the process of applying corrections
    if 'applying_corrections' not in st.session_state:
        numeric_issues = validate_numeric_columns(df, numeric_columns)

        for col, issues in numeric_issues.items():
            st.markdown(f"#### Issues found in {col}:")

            # Report null values with more detail
            if 'null_count' in issues:
                null_count = issues['null_count']
                null_indices = issues['null_indices']
                st.warning(f"{null_count} empty or null values found in rows: {', '.join(map(str, null_indices))}. <br><br>If this is unexpected, please fix your spreadsheet before trying again.")

            # Handle invalid non-null values
            if 'invalid_values' in issues:
                invalid_values = issues['invalid_values']
                st.error(f"{len(invalid_values)} non-numeric values found")

                corrections = {}
                for idx, value in invalid_values.items():
                    correct_value = st.text_input(
                        f"Correct value for '{value}' in row {idx}:",
                        key=f"{col}_{idx}"
                    )
                    if correct_value:
                        try:
                            float(correct_value)
                            corrections[value] = correct_value
                        except ValueError:
                            st.error(f"'{correct_value}' is not a valid numeric value.")

                if corrections:
                    all_corrections[col] = corrections

            if not any(issues):
                st.success(f"All values in {col} are valid!")

    # Apply corrections button
    # Function to apply all corrections
    def apply_corrections():
        st.session_state.applying_corrections = True
        for col, correct_dict in all_corrections.items():
            df[col] = df[col].replace(correct_dict)
        st.session_state.df = df
        st.success("Corrections applied successfully!")
        st.rerun()

    if all_corrections:
        if st.button("Apply All Corrections"):
            apply_corrections()
    #else:
        # Only show success message if we're not in the process of applying corrections
        #if 'applying_corrections' not in st.session_state:
        #    st.success("All race, ethnicity and age data is valid!")

    # Clear the applying_corrections flag if it exists
    if 'applying_corrections' in st.session_state:
        del st.session_state.applying_corrections

    # Only show "Next step" button if no corrections are needed
    remaining_corrections = {k: v for k, v in all_corrections.items()
                           if not k.startswith('skip_')}
    if not remaining_corrections:
        st.success("All race, ethnicity and age data is valid!")
        if st.button("Next step"):
            # Clear skip flags for next run
            for key in list(st.session_state.keys()):
                if key.startswith('skip_'):
                    del st.session_state[key]
            st.session_state.step = 5
            st.rerun()

# Step 5: Primary Site Validation
elif st.session_state.step == 5:
    st.subheader("Step 5: Validate Primary Site")
    df = st.session_state.df

    if 'Primary Site' not in df.columns:
        st.info("No Primary Site column found in the data. Proceeding to next step.")
        if st.button("Next step"):
            st.session_state.step = 6
            st.rerun()
    else:
        # Initialize session state for Primary Site mapping
        if 'primary_site_mapped' not in st.session_state:
            st.session_state.primary_site_mapped = False
        if 'primary_site_mappings' not in st.session_state:
            st.session_state.primary_site_mappings = {}

        # Get invalid values
        invalid_values = df[~df['Primary Site'].isin(permissible_primary_site)]['Primary Site'].unique()

        if len(invalid_values) == 0:
            st.success("All Primary Site values are valid!")
            if st.button("Next step"):
                st.session_state.step = 6
                st.rerun()
        else:
            if not st.session_state.primary_site_mapped:
                st.markdown(f"#### Found {len(invalid_values)} non-standard Primary Site values")

                # Show mapping interface
                mappings = {}
                for value in invalid_values:
                    # Create selectbox with close matches first, then all options
                    options = get_prioritized_options(value, permissible_primary_site)

                    selected_value = st.selectbox(
                        f"Map '{value}' to:",
                        options=options,
                        key=f"primary_site_{value}"
                    )

                    if selected_value != 'Keep current value':
                        mappings[value] = selected_value

                # Button to confirm mappings
                if st.button("Confirm Primary Site mappings"):
                    st.session_state.primary_site_mappings = mappings
                    st.session_state.primary_site_mapped = True

                    # Apply mappings
                    if mappings:
                        df['Primary Site'] = df['Primary Site'].replace(mappings)
                        st.session_state.df = df

                    st.rerun()
            else:
                # Show mapping summary
                st.markdown("#### Primary Site Mapping Summary:")

                # Group values by action
                to_keep = [val for val in invalid_values if val not in st.session_state.primary_site_mappings]
                to_remap = st.session_state.primary_site_mappings

                if to_keep:
                    st.info(f"Values to keep unchanged: {', '.join(f'`{val}`' for val in to_keep)}")

                if to_remap:
                    remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                    st.info(f"Values that were remapped: {', '.join(remap_summary)}")

                # Button to reset mappings
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Map additional values"):
                        st.session_state.primary_site_mapped = False
                        st.session_state.primary_site_mappings = {}
                        st.rerun()

                with col2:
                    if st.button("Next step"):
                        st.session_state.step = 6
                        st.rerun()

# Step 6: Primary Diagnosis Validation
elif st.session_state.step == 6:
    st.subheader("Step 6: Validate Primary Diagnosis")
    df = st.session_state.df

    if 'Primary Diagnosis' not in df.columns:
        st.info("No Primary Diagnosis column found in the data. Proceeding to next step.")
        if st.button("Next step"):
            st.session_state.step = 7
            st.rerun()
    else:
        # Initialize session state for Primary Diagnosis mapping
        if 'primary_diagnosis_mapped' not in st.session_state:
            st.session_state.primary_diagnosis_mapped = False
        if 'primary_diagnosis_mappings' not in st.session_state:
            st.session_state.primary_diagnosis_mappings = {}

        # Get invalid values
        invalid_values = df[~df['Primary Diagnosis'].isin(permissible_primary_diagnosis)]['Primary Diagnosis'].unique()

        if len(invalid_values) == 0:
            st.success("All Primary Diagnosis values are valid!")
            if st.button("Next step"):
                st.session_state.step = 7
                st.rerun()
        else:
            if not st.session_state.primary_diagnosis_mapped:
                st.markdown(f"#### Found {len(invalid_values)} non-standard Primary Diagnosis values")

                # Show mapping interface
                mappings = {}
                for value in invalid_values:
                    # Create selectbox with close matches first, then all options
                    options = get_prioritized_options(value, permissible_primary_diagnosis)

                    selected_value = st.selectbox(
                        f"Map '{value}' to:",
                        options=options,
                        key=f"primary_diagnosis_{value}"
                    )

                    if selected_value != 'Keep current value':
                        mappings[value] = selected_value

                # Button to confirm mappings
                if st.button("Confirm Primary Diagnosis mappings"):
                    st.session_state.primary_diagnosis_mappings = mappings
                    st.session_state.primary_diagnosis_mapped = True

                    # Apply mappings
                    if mappings:
                        df['Primary Diagnosis'] = df['Primary Diagnosis'].replace(mappings)
                        st.session_state.df = df

                    st.rerun()
            else:
                # Show mapping summary
                st.markdown("#### Primary Diagnosis Mapping Summary:")

                # Group values by action
                to_keep = [val for val in invalid_values if val not in st.session_state.primary_diagnosis_mappings]
                to_remap = st.session_state.primary_diagnosis_mappings

                if to_keep:
                    st.info(f"Values to keep unchanged: {', '.join(f'`{val}`' for val in to_keep)}")

                if to_remap:
                    remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                    st.info(f"Values that were remapped: {', '.join(remap_summary)}")

                # Button to reset mappings
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reset mappings"):
                        st.session_state.primary_diagnosis_mapped = False
                        st.session_state.primary_diagnosis_mappings = {}
                        st.rerun()

                with col2:
                    if st.button("Next step"):
                        st.session_state.step = 7
                        st.rerun()

# Step 7: Download Standardized Data
elif st.session_state.step == 7:
    st.subheader("Step 7: Download Standardized Data")
    df = st.session_state.df

    # Get the default filename based on first Project Short Name value
    default_filename = f"{df['Project Short Name'].iloc[0]}-Clinical-Standardized.xlsx"

    # Create a text input for custom filename with the default value
    custom_filename = st.text_input(
        "Filename:",
        value=default_filename,
        help="You can modify the filename if desired"
    )
    st.markdown("You must press ENTER after setting a new file name.")

    # Ensure the filename ends with .xlsx
    if not custom_filename.endswith('.xlsx'):
        custom_filename += '.xlsx'

    # Reorder columns
    df = reorder_columns(df)
    output = BytesIO()

    # If we have other sheets, write them all to the Excel file
    if st.session_state.other_sheets:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Standardized Data', index=False)
            for sheet_name, sheet_data in st.session_state.other_sheets.items():
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        st.info("The downloaded file will include your standardized data sheet along with all other sheets from the original file.")
    else:
        # Single sheet export
        df.to_excel(output, index=False)

    st.download_button(
        "Download Standardized XLSX file",
        data=output.getvalue(),
        file_name=custom_filename,
        help="Download the standardized data in Excel format"
    )

    if st.button("Restart"):
        reset_session_state()
        st.rerun()
