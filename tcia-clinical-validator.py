import streamlit as st 
import pandas as pd 
import numpy as np 
from io import BytesIO 
import re 
from difflib import SequenceMatcher 

# Set basic page configuration for the Streamlit app
st.set_page_config(page_title="TCIA Clinical Data Validator")

# --- Global Variables ---
# Define the core required columns that must be present in the submitted data
required_columns = ['Project Short Name', 'Case ID']

# Define a comprehensive list of all permissible column names the validator recognizes.
# This list is used for column mapping and renaming.
allowable_columns = [
    'Project Short Name', 'Case ID', 'Race', 'Ethnicity', 'Sex at Birth',
    'Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery','Age UOM', 'Age at Imaging',
    'Primary Diagnosis', 'Primary Site'
]

# --- Session State Management Function ---
def reset_session_state():
    """
    Resets all Streamlit session state variables to their initial values.
    This function is crucial for restarting the application's workflow,
    clearing any previously loaded data, user inputs, or validation states.
    """
    # Core step tracking: Resets the current step to 1 (File Upload)
    st.session_state.step = 1
    # Resets user-defined project short name and age unit of measure
    st.session_state.project_short_name = ''
    st.session_state.age_uom = ''

    # Column mapping related states: Resets flags and dictionary used for column renaming/deletion
    st.session_state.columns_mapped = False
    st.session_state.column_mapping = {}
    st.session_state.mapping_applied = False

    # Categorical validation states: Clears temporary states from categorical value validation
    if 'kept_values' in st.session_state:
        del st.session_state.kept_values
    if 'fix_column_states' in st.session_state:
        del st.session_state.fix_column_states

    # Clear previous validation mappings: Ensures fresh start for value-level mappings
    if 'mapping_complete' in st.session_state:
        del st.session_state.mapping_complete
    if 'value_mappings' in st.session_state:
        del st.session_state.value_mappings

    # Primary Diagnosis and Primary Site mapping states: Explicitly clears mappings for these specific columns
    if 'primary_diagnosis_mapped' in st.session_state:
        del st.session_state.primary_diagnosis_mapped
    if 'primary_diagnosis_mappings' in st.session_state:
        del st.session_state.primary_diagnosis_mappings
    if 'primary_site_mapped' in st.session_state:
        del st.session_state.primary_site_mapped
    if 'primary_site_mappings' in st.session_state:
        del st.session_state.primary_site_mappings

    # Clear the main DataFrame from session state
    if 'df' in st.session_state:
        del st.session_state.df

    # Clear any dynamically created session state keys (e.g., for skip buttons or specific value fixes)
    keys_to_remove = []
    for key in st.session_state.keys():
        if (key.startswith('skip_') or # Flags for skipping validation steps
            key.startswith('Race_') or # States related to Race column corrections
            key.startswith('Age_') or # States related to Age column corrections
            key.startswith('Primary_Diagnosis_') or # States related to Primary Diagnosis corrections
            key.startswith('Primary_Site_') or # States related to Primary Site corrections
            key.startswith('fix_')): # General fix states
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del st.session_state[key]

# --- Data Loading and Permissible Value Functions ---
def load_permissible_values(file_path):
    """
    Loads permissible values from a specified Excel file.
    It expects a column named 'Permissible Value' in the Excel sheet.
    """
    try:
        df = pd.read_excel(file_path)
        # Extract unique, non-null values from the 'Permissible Value' column and sort them
        values = df['Permissible Value'].dropna().unique().tolist()
        return sorted(values)
    except Exception as e:
        # Display an error message if loading fails
        st.error(f"Error loading permissible values from {file_path}: {str(e)}")
        return []

@st.cache_data # Cache the function output to avoid reloading data on rerun
def initialize_permissible_values():
    """
    Loads the permissible values for 'Primary Diagnosis' and 'Primary Site'
    from their respective caDSR Excel files. This ensures these lists are
    loaded only once for the application's lifetime.
    """
    primary_diagnosis_values = load_permissible_values('primary_diagnosis_caDSR_14905532.xlsx')
    primary_site_values = load_permissible_values('primary_site_caDSR_14883047.xlsx')
    return primary_diagnosis_values, primary_site_values

# Load the permissible values for Primary Diagnosis and Primary Site at application startup
permissible_primary_diagnosis, permissible_primary_site = initialize_permissible_values()

# Define hardcoded lists of permissible values for common categorical columns
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

# Define conversion factors for Age Unit of Measure (UOM) to convert to years
age_uom_factors = {
    'Day': 1 / 365, # 1 day = 1/365 years
    'Month': 1 / 12, # 1 month = 1/12 years
    'Year': 1 # 1 year = 1 year (base unit)
}

# --- Data Cleaning and Validation Helper Functions ---
def convert_to_strings(df):
    """
    Converts all DataFrame columns to string type, except for specified age columns.
    This ensures consistent handling of categorical/text data.
    """
    age_columns = ['Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging']
    for col in df.columns:
        if col not in age_columns:
            df[col] = df[col].astype(str) # Convert to string
    return df

def validate_and_clean_data(df):
    """
    Performs initial data cleaning and validation, including handling capitalization
    for certain categorical columns and removing duplicate rows.
    """
    report = [] # List to store messages about performed cleaning operations

    # Convert non-age columns to strings to ensure consistent data types for string operations
    df = convert_to_strings(df)

    # Track unique capitalization fixes made for reporting to the user
    capitalization_fixes = {
        'Race': set(),
        'Ethnicity': set(),
        'Sex at Birth': set()
    }

    # Handle Race column (special case for multiple semicolon-separated values)
    if 'Race' in df.columns:
        def fix_race_values(race_str):
            """Internal helper to process semicolon-separated race values."""
            if pd.isna(race_str): # Handle NaN values
                return race_str
            fixed_races = []
            for race in str(race_str).split(';'): # Split by semicolon
                race = race.strip() # Remove leading/trailing whitespace
                correct_race = get_correct_value(race, permissible_race) # Find correct capitalization
                if correct_race and correct_race != race:
                    capitalization_fixes['Race'].add((race, correct_race)) # Log the fix
                fixed_races.append(correct_race if correct_race else race) # Use corrected value or original
            return ';'.join(sorted(set(fixed_races))) # Rejoin, remove duplicates, and sort

        df['Race'] = df['Race'].apply(fix_race_values) # Apply the fixing function to the Race column

    # Handle Ethnicity and Sex at Birth columns
    for col in ['Ethnicity', 'Sex at Birth']:
        if col in df.columns:
            # Select the appropriate list of valid values for the current column
            valid_values = permissible_ethnicity if col == 'Ethnicity' else permissible_sex_at_birth

            def fix_value(val):
                """Internal helper to fix capitalization for a single value."""
                if pd.isna(val):
                    return val
                correct_val = get_correct_value(val, valid_values) # Find correct capitalization
                if correct_val and correct_val != val:
                    capitalization_fixes[col].add((val, correct_val)) # Log the fix
                return correct_val if correct_val else val # Use corrected value or original

            df[col] = df[col].apply(fix_value) # Apply the fixing function to the column

    # Report unique capitalization fixes that were automatically performed
    for col, fixes in capitalization_fixes.items():
        if fixes:
            if col == 'Race':
                fix_details = [f"'{old}' → '{new}'" for old, new in fixes]
                report.append(f"Automatically corrected capitalization of {len(fixes)} unique Race values: {', '.join(fix_details)}")
            else:
                fix_details = [f"'{old}' → '{new}'" for old, new in fixes]
                report.append(f"Automatically corrected capitalization of {len(fixes)} unique values in {col} column: {', '.join(fix_details)}")

    # Drop duplicate rows and report their original row numbers
    duplicate_rows = df[df.duplicated()].index.tolist() # Get indices of duplicate rows
    if duplicate_rows:
        df.drop_duplicates(inplace=True) # Remove duplicates
        report.append(f"Removed {len(duplicate_rows)} duplicate rows: {duplicate_rows}") # Report action

    return df, report

def validate_numeric_columns(df, numeric_columns):
    """
    Validates specified numeric columns for non-numeric values and nulls.
    Identifies and reports issues without applying corrections.
    """
    numeric_issues = {} # Dictionary to store issues found per numeric column

    for col in numeric_columns:
        if col in df.columns:
            # Convert empty strings and whitespace-only strings to NaN for consistent null detection
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

            # Count null values (NaN, None, and empty strings now)
            null_mask = df[col].isna()
            null_count = null_mask.sum()

            # Find non-numeric values (excluding nulls)
            non_null_mask = ~null_mask # Mask for non-null values
            non_null_values = df[col][non_null_mask] # Extract non-null values
            # Attempt to convert to numeric; 'coerce' turns errors into NaN
            numeric_mask = pd.to_numeric(non_null_values, errors='coerce').notna()
            # Identify values that are non-null AND failed numeric conversion
            non_numeric = df[col][non_null_mask & ~numeric_mask]

            if null_count > 0 or len(non_numeric) > 0:
                issues = {}
                if null_count > 0:
                    issues['null_count'] = null_count
                    issues['null_indices'] = df[null_mask].index.tolist() # Store indices of nulls
                if len(non_numeric) > 0:
                    issues['invalid_values'] = non_numeric # Store the actual non-numeric values
                numeric_issues[col] = issues # Add issues for this column to the main dictionary

    return numeric_issues

def validate_categorical_column(df, column, valid_values):
    """
    Validates a categorical column against a list of permissible values.
    Handles capitalization and semicolon-separated values for 'Race'.
    Returns a Series of boolean values (True for valid, False for invalid).
    """
    def is_valid(value):
        """Internal helper to check if a single value (or set of values) is valid."""
        if pd.isna(value): # NaN values are considered valid in this context
            return True
        if column == 'Race': # Special handling for 'Race' due to multiple values
            return all(get_correct_value(race.strip(), valid_values) is not None # Check each individual race value
                       for race in str(value).split(';'))
        # For other categorical columns, just check if the value (case-insensitively corrected) is valid
        return get_correct_value(value, valid_values) is not None

    return df[column].apply(is_valid) # Apply the validation check to the entire column

# --- Column and Value Matching Helper Functions ---
def is_valid_project_short_name(name):
    """
    Validates if a given string conforms to the Project Short Name pattern:
    1-30 characters, alphanumeric, spaces, dashes, and underscores allowed.
    """
    return bool(re.match(r'^[a-zA-Z0-9\s_-]{1,30}$', name))

def get_correct_column_name(col):
    """
    Finds the correctly capitalized column name from `allowable_columns`
    based on a case-insensitive match. Returns the original column name if no match found.
    """
    # Create a dictionary mapping lowercase allowable names to their correct capitalization
    lower_allowable = {c.lower(): c for c in allowable_columns}
    # Return the correctly capitalized name if a lowercase match exists, otherwise return original
    return lower_allowable.get(col.lower(), col)

def get_correct_value(value, valid_values):
    """
    Finds the correctly capitalized value from a list of `valid_values`,
    matching the input `value` case-insensitively.
    Returns None if no match is found.
    """
    # Create a dictionary mapping lowercase valid values to their correct capitalization
    lower_valid = {v.lower(): v for v in valid_values}
    # Return the correctly capitalized value if a lowercase match exists, otherwise None
    return lower_valid.get(str(value).lower())

def get_prioritized_options(value, valid_options, n_suggestions=5):
    """
    Returns a prioritized list of valid options for a given input value based on multiple
    string matching strategies (Levenshtein distance, word matching, acronyms).
    This helps suggest the most relevant corrections to the user.
    """
    def clean_string(s):
        """Internal helper to clean string for similarity comparison (lowercase, remove special chars)."""
        return re.sub(r'[^a-z0-9\s]', '', str(s).lower())

    def get_similarity_score(option):
        """Internal helper to calculate a weighted similarity score between the input value and an option."""
        # Calculate base string similarity using SequenceMatcher (ratio of matching characters)
        base_score = SequenceMatcher(None, clean_string(value), clean_string(option)).ratio()

        # Boost score for matches at the start of words (e.g., 'Lung Adenocarcinoma' vs 'Lung ADC')
        words_value = set(clean_string(value).split())
        words_option = set(clean_string(option).split())
        word_start_matches = sum(1 for w1 in words_value
                                 for w2 in words_option
                                 if w2.startswith(w1) or w1.startswith(w2))

        # Boost score for acronym matches (e.g., 'GBM' vs 'Glioblastoma Multiforme')
        value_acronym = ''.join(word[0] for word in clean_string(value).split() if word)
        option_acronym = ''.join(word[0] for word in clean_string(option).split() if word)
        acronym_match = SequenceMatcher(None, value_acronym, option_acronym).ratio()

        # Boost score for partial word matches (e.g., 'Adeno' vs 'Adenocarcinoma')
        shared_words = words_value.intersection(words_option)
        # Avoid division by zero if words_value is empty
        word_match_score = len(shared_words) / max(len(words_value), len(words_option)) if words_value else 0

        # Calculate weighted final score based on various matching heuristics
        final_score = (base_score * 0.4 +            # 40% importance to overall string similarity
                       (word_start_matches * 0.1) +  # 10% importance to word start matches
                       (acronym_match * 0.2) +       # 20% importance to acronym matches
                       (word_match_score * 0.3))     # 30% importance to shared word matches

        return final_score

    # Calculate scores for all valid options
    scored_options = [(option, get_similarity_score(option)) for option in valid_options]

    # Sort options by their similarity score in descending order
    scored_options.sort(key=lambda x: x[1], reverse=True)

    # Extract the top N suggestions and the remaining options
    top_matches = [option for option, _ in scored_options[:n_suggestions]]
    remaining_options = [option for option, _ in scored_options[n_suggestions:]]

    # Construct the final list: 'Keep current value' always first, then top matches, then the rest
    return ['Keep current value'] + top_matches + remaining_options

def reorder_columns(df):
    """
    Reorders the DataFrame columns into a preferred standard order for TCIA.
    Columns not in the preferred order are appended at the end.
    """
    preferred_order = [
        'Project Short Name', 'Case ID', 'Primary Diagnosis', 'Primary Site',
        'Race', 'Ethnicity', 'Sex at Birth', 'Age UOM',
        'Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging'
    ]
    # Filter preferred_order to include only columns actually present in the DataFrame
    existing_columns = [col for col in preferred_order if col in df.columns]
    # Get any other columns not in the preferred list
    other_columns = [col for col in df.columns if col not in existing_columns]
    # Return DataFrame with columns in preferred order followed by other columns
    return df[existing_columns + other_columns]

def process_file(file_or_url, is_url=False):
    """
    Helper function to read and process uploaded spreadsheet files (CSV, XLSX, TSV)
    or files from a given URL. Handles multi-sheet Excel files.
    """
    try:
        if is_url:
            file_name = file_or_url # URL acts as the filename for extension check
            if not any(file_name.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.tsv']):
                st.error("URL must point to a .csv, .xlsx, or .tsv file")
                return None, False, None # Return None for df, False for proceed, None for other_sheets
        else:
            file_name = file_or_url.name # Get the name from Streamlit's UploadedFile object

        other_sheets = None # Initialize other_sheets to None, will be populated for XLSX if needed

        # Determine file type based on extension and read into a pandas DataFrame
        if file_name.lower().endswith('.csv'):
            df = pd.read_csv(file_or_url)
            proceed_to_next = True
        elif file_name.lower().endswith('.xlsx'):
            excel_file = pd.ExcelFile(file_or_url) # Create an ExcelFile object to access sheets
            sheet_names = excel_file.sheet_names # Get all sheet names
            if len(sheet_names) > 1:
                # If multiple sheets, allow user to select one for analysis
                selected_tab = st.selectbox("Select Sheet to Analyze", sheet_names)
                # Option to keep other sheets in the final output Excel file
                keep_other_sheets = st.checkbox("Keep other sheets in final output", value=True)

                # Read only the selected sheet into the main DataFrame
                df = pd.read_excel(file_or_url, sheet_name=selected_tab)

                # If selected, read and store other sheets in a dictionary
                if keep_other_sheets:
                    other_sheets = {}
                    other_sheet_names = [s for s in sheet_names if s != selected_tab]
                    for sheet in other_sheet_names:
                        other_sheets[sheet] = pd.read_excel(file_or_url, sheet_name=sheet)

                # A button is required to proceed when a sheet is selected in multi-sheet files
                proceed_to_next = st.button("Next")
            else:
                # If only one sheet, read it directly and proceed automatically
                df = pd.read_excel(file_or_url)
                proceed_to_next = True
        elif file_name.lower().endswith('.tsv'):
            df = pd.read_csv(file_or_url, delimiter='\t') # Read tab-separated values
            proceed_to_next = True
        else:
            # Handle unsupported file formats
            st.error("Unsupported file format. Please upload a .csv, .xlsx, or .tsv file")
            return None, False, None

        return df, proceed_to_next, other_sheets # Return DataFrame, a flag to proceed, and other sheets

    except Exception as e:
        # Catch and display any errors during file processing
        st.error(f"Error processing file: {str(e)}")
        return None, False, None

# --- Main Streamlit Application UI ---

# Custom CSS to dynamically switch the TCIA logo based on the user's system theme (dark/light mode)
st.markdown(
    """
    <style>
    @media (prefers-color-scheme: dark) {
        .logo {
            content: url(https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-02.png); /* Dark mode logo */
        }
    }
    @media (prefers-color-scheme: light) {
        .logo {
            content: url(https://www.cancerimagingarchive.net/wp-content/uploads/2021/06/TCIA-Logo-01.png); /* Light mode logo */
        }
    }
    </style>
    <header class="main-header">
        <img class="logo" alt="App Logo">
    </header>
    """,
    unsafe_allow_html=True # Allow HTML injection for custom styling
)

# Main title of the Streamlit application
st.title("Clinical Data Validator")

# Initialize session state variables if they don't exist.
# These variables persist across app reruns and maintain the application's state.
if 'step' not in st.session_state:
    st.session_state.step = 1 # Tracks the current step in the validation workflow
if 'project_short_name' not in st.session_state:
    st.session_state.project_short_name = '' # Stores user-defined project short name
if 'age_uom' not in st.session_state:
    st.session_state.age_uom = '' # Stores user-selected age unit of measure
if 'other_sheets' not in st.session_state:
    st.session_state.other_sheets = None # Stores DataFrames of other sheets from an Excel file

# --- Application Steps (Conditional Rendering based on st.session_state.step) ---

# Step 1: File Upload and Import
if st.session_state.step == 1:
    st.subheader("Step 1: Upload your CSV, XLSX, or TSV file")
    # File uploader widget for user to upload a file
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "tsv"])

    df = None # Initialize df to None for scope if no file is uploaded
    proceed_to_next = False
    other_sheets = None

    if uploaded_file:
        # Process the uploaded file
        df, proceed_to_next, other_sheets = process_file(uploaded_file)
    else:
        # Alternative input: URL for the file
        url = st.text_input("...or provide the URL of the file")
        if url:
            # Process file from URL
            df, proceed_to_next, other_sheets = process_file(url, is_url=True)

    # If a DataFrame is successfully loaded and the 'proceed_to_next' flag is True
    if 'df' in locals() and df is not None and proceed_to_next:
        st.success("File imported successfully!")
        # Remove leading and trailing spaces from all string values in the DataFrame
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        # Store the processed DataFrame and other sheets in session state
        st.session_state.df = df
        st.session_state.other_sheets = other_sheets
        st.session_state.step = 2 # Advance to the next step
        st.rerun() # Rerun the app to display the next step's UI

# Step 2: Analyze and Map Columns
elif st.session_state.step == 2:
    st.subheader("Step 2: Map column names to TCIA column names")
    df = st.session_state.df # Retrieve DataFrame from session state

    # Automatically correct capitalization for column names that match `allowable_columns`
    columns_to_rename = {}
    for col in df.columns:
        col_str = str(col) # Ensure column name is treated as a string
        correct_name = get_correct_column_name(col_str)
        if correct_name != col_str:
            columns_to_rename[col_str] = correct_name

    if columns_to_rename:
        df.rename(columns=columns_to_rename, inplace=True) # Apply renaming
        st.info(f"The following columns were automatically renamed to correct capitalization: {', '.join(columns_to_rename.values())}")

    # Identify columns that are genuinely not in the allowable list (even after capitalization fixes)
    unexpected_columns = [
        str(col) for col in df.columns
        if str(col).lower() not in [c.lower() for c in allowable_columns]
    ]

    # Initialize session state variables for column mapping if not present
    if 'columns_mapped' not in st.session_state:
        st.session_state.columns_mapped = False # Flag if all columns have been mapped/addressed
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {} # Stores user-defined mappings
    if 'mapping_applied' not in st.session_state:
        st.session_state.mapping_applied = False # Flag if mapping changes have been applied

    if unexpected_columns:
        # If mapping has not yet been applied, show the mapping interface
        if not st.session_state.mapping_applied:
            st.warning("The following unexpected columns were found.")
            column_mapping = {} # Temporary dictionary for current mapping selections
            for col in unexpected_columns:
                # For each unexpected column, provide a selectbox to map it to an allowable column,
                # or choose to leave it unmodified or delete it.
                option = st.selectbox(
                    f"How should '{col}' be mapped?",
                    allowable_columns + ["Leave unmodified", "Delete column"], # Options for mapping
                    key=col, # Unique key for the widget
                    index=len(allowable_columns) # Default to "Leave unmodified"
                )
                column_mapping[col] = option

            # Button to apply the chosen column mappings
            if st.button("Apply column mapping"):
                st.session_state.column_mapping = column_mapping # Store mappings in session state
                st.session_state.mapping_applied = True # Set flag to indicate mapping applied

                # Apply the chosen actions (rename or delete) to the DataFrame
                for col, action in column_mapping.items():
                    if action == "Delete column":
                        df.drop(columns=[col], inplace=True) # Delete the column
                    elif action != "Leave unmodified": # If not 'Leave unmodified', it's a rename
                        df.rename(columns={col: action}, inplace=True) # Rename the column

                st.session_state.df = df # Update DataFrame in session state
                st.rerun() # Rerun to show updated state
        else:
            # If mappings have been applied, show a summary of the actions taken
            to_delete = []
            to_remain = []
            to_remap = {}

            for col, action in st.session_state.column_mapping.items():
                if action == "Delete column":
                    to_delete.append(col)
                elif action == "Leave unmodified":
                    to_remain.append(col)
                else: # It's a rename action
                    to_remap[col] = action

            st.markdown("### Column Mapping Summary:")

            if to_delete:
                st.info(f"The following columns were deleted: {', '.join(f'`{col}`' for col in to_delete)}")
            if to_remain:
                st.info(f"The following columns remain unchanged: {', '.join(f'`{col}`' for col in to_remain)}")
            if to_remap:
                remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                st.info(f"The following columns were remapped: {', '.join(remap_summary)}")

            st.session_state.columns_mapped = True # Mark columns as addressed
    else:
        # If no unexpected columns were found initially (or after auto-correction)
        st.success("All columns are correctly named or have been automatically corrected.")
        st.session_state.columns_mapped = True # Mark columns as addressed

    # Update the DataFrame in session state after all potential column operations
    st.session_state.df = df

    # Show "Next step" button only if all columns have been mapped or confirmed as valid
    if st.session_state.columns_mapped:
        if st.button("Next step"):
            # Reset mapping state for the next time the user might return to this step
            st.session_state.mapping_applied = False
            st.session_state.column_mapping = {}
            st.session_state.step = 3 # Advance to the next step
            st.rerun()

# Step 3: Check Required Columns and Validate Project Short Name & Age UOM
elif st.session_state.step == 3:
    st.subheader("Step 3: Check for Minimum Required Columns and Validate Project Details")
    df = st.session_state.df # Retrieve DataFrame from session state

    # Convert non-age columns to strings for consistent validation
    df = convert_to_strings(df)

    # Check for presence of required columns
    missing_case_id = 'Case ID' not in df.columns
    missing_project_short_name = 'Project Short Name' not in df.columns

    # Check for age columns and Age UOM
    age_columns = ['Age at Diagnosis', 'Age at Enrollment', 'Age at Surgery', 'Age at Earliest Imaging']
    existing_age_columns = [col for col in age_columns if col in df.columns]
    missing_age_uom = 'Age UOM' not in df.columns and existing_age_columns # UOM required if any age column exists

    if missing_case_id:
        st.error("The 'Case ID' column is missing from your spreadsheet. This is a required column.")
        if st.button("Restart"): # Option to restart if a critical required column is missing
            st.session_state.step = 1
            if 'df' in st.session_state:
                del st.session_state.df
            st.rerun()
    else:
        project_short_name_valid = True # Flag to track validity of project short name
        age_uom_valid = True # Flag to track validity of age UOM

        # Handle missing or invalid 'Project Short Name'
        name_updates = {} # Dictionary to store updates for existing invalid names
        if missing_project_short_name:
            st.warning("The 'Project Short Name' column is missing from your spreadsheet.")
            # Prompt user to input a Project Short Name
            project_short_name = st.text_input("Please specify a Project Short Name:", value=st.session_state.project_short_name)
            if project_short_name:
                if is_valid_project_short_name(project_short_name):
                    st.session_state.project_short_name = project_short_name # Store valid input
                else:
                    st.error("Invalid Project Short Name. It should be 1-30 characters long and contain only letters, numbers, dashes, and underscores.")
                    project_short_name_valid = False # Mark as invalid
            else:
                project_short_name_valid = False # Input is empty, so invalid

        else: # 'Project Short Name' column exists
            # Check for invalid values within the existing 'Project Short Name' column
            invalid_names = df[~df['Project Short Name'].apply(is_valid_project_short_name)]['Project Short Name'].unique()
            if len(invalid_names) > 0:
                st.warning("Some Project Short Names are invalid. Please update them:")
                for name in invalid_names:
                    # For each invalid name, prompt for a new valid one
                    new_name = st.text_input(f"New name for '{name}' (1-30 characters, letters, numbers, dashes, underscores):", key=f"psn_fix_{name}")
                    if new_name:
                        if is_valid_project_short_name(new_name):
                            name_updates[name] = new_name # Store valid update
                        else:
                            st.error(f"'{new_name}' is not a valid Project Short Name.")
                            project_short_name_valid = False
                    else:
                        project_short_name_valid = False # New name input is empty

        # Handle missing 'Age UOM' if age columns are present
        if missing_age_uom:
            st.warning("The 'Age UOM' column is missing, but age-related columns are present.")
            # Prompt user to select an Age Unit of Measure
            age_uom = st.selectbox(
                "Please select the Age Unit of Measure:",
                options=permissible_age_uom,
                index=0 if not st.session_state.age_uom else permissible_age_uom.index(st.session_state.age_uom)
            )
            if age_uom:
                st.session_state.age_uom = age_uom # Store selected UOM
            else:
                age_uom_valid = False
        else:
            age_uom_valid = True # If Age UOM column exists, assume it's valid for now (further validation in step 4)

        # Indicate that column validation is successful
        st.success("Required column and project name validation successful.")

        # Single "Next step" button for both applying updates and proceeding
        if st.button("Next step"):
            if project_short_name_valid and age_uom_valid:
                # Apply updates to 'Project Short Names' if a new name was provided for a missing column
                if missing_project_short_name:
                    df['Project Short Name'] = st.session_state.project_short_name
                # Apply updates for existing invalid Project Short Names
                for old_name, new_name in name_updates.items():
                    df.loc[df['Project Short Name'] == old_name, 'Project Short Name'] = new_name

                # Apply 'Age UOM' if it was missing and a selection was made
                if missing_age_uom:
                    df['Age UOM'] = st.session_state.age_uom

                st.session_state.df = df # Update DataFrame in session state
                st.success("Changes applied successfully!")
                st.session_state.step = 4 # Advance to the next step
                st.rerun()
            else:
                # Show errors if validation failed
                if not project_short_name_valid:
                    st.error("Please provide a valid Project Short Name.")
                if not age_uom_valid:
                    st.error("Please select an Age Unit of Measure.")

# Step 4: Validate Race, Ethnicity, and Age Data
elif st.session_state.step == 4:
    st.subheader("Step 4: Validate Race, Ethnicity, and Age Data")
    df = st.session_state.df # Retrieve DataFrame from session state

    # Perform initial data cleaning (capitalization, duplicates)
    df, validation_report = validate_and_clean_data(df)

    # Display any automated cleaning operations performed
    if validation_report:
        for message in validation_report:
            st.info(message)

    all_corrections = {} # Dictionary to accumulate all manual corrections needed across columns

    # 1. Validate 'Race' column (after capitalization fixes)
    if 'Race' in df.columns:
        # Identify invalid race values (those not matching permissible values after capitalization fixes)
        invalid_races = ~validate_categorical_column(df, 'Race', permissible_race)
        invalid_race_values = df[invalid_races]['Race'].unique()

        if len(invalid_race_values) > 0:
            st.markdown("#### Invalid Race values found (after capitalization fixes):")
            race_corrections = {} # Store corrections specifically for Race
            for value in invalid_race_values:
                st.write(f"Invalid value: '{value}'")
                # Provide a multiselect box for user to choose correct race(s)
                correct_races = st.multiselect(
                    f"Select correct races for '{value}':",
                    options=permissible_race,
                    key=f"Race_{value}"
                )
                if correct_races:
                    race_corrections[value] = ';'.join(correct_races) # Store as semicolon-separated string

            if race_corrections:
                all_corrections['Race'] = race_corrections # Add to overall corrections

    # 2. Validate other categorical columns ('Ethnicity', 'Sex at Birth', 'Age UOM')
    categorical_columns = {
        'Ethnicity': permissible_ethnicity,
        'Sex at Birth': permissible_sex_at_birth,
        'Age UOM': permissible_age_uom
    }

    for col, valid_values in categorical_columns.items():
        if col in df.columns:
            # Identify invalid values in the current categorical column
            invalid_mask = ~validate_categorical_column(df, col, valid_values)
            invalid_values = df[invalid_mask][col].unique()

            if len(invalid_values) > 0:
                st.markdown(f"#### Invalid {col} values found (after capitalization fixes):")
                corrections = {} # Store corrections for the current column
                for value in invalid_values:
                    # Provide a selectbox for user to choose a correct value
                    correct_value = st.selectbox(
                        f"Correct value for '{value}' in {col}:",
                        options=valid_values,
                        key=f"{col}_{value}"
                    )
                    if correct_value:
                        corrections[value] = correct_value

                if corrections:
                    all_corrections[col] = corrections # Add to overall corrections

    # 3. Validate numeric columns (Age columns)
    numeric_columns = ['Age at Diagnosis', 'Age at Enrollment',
                       'Age at Surgery', 'Age at Earliest Imaging']

    # Only validate numeric columns if not currently applying corrections (to prevent infinite loops)
    if 'applying_corrections' not in st.session_state:
        numeric_issues = validate_numeric_columns(df, numeric_columns)

        for col, issues in numeric_issues.items():
            st.markdown(f"#### Issues found in {col}:")

            # Report null values with more detail (indices)
            if 'null_count' in issues:
                null_count = issues['null_count']
                null_indices = issues['null_indices']
                st.warning(f"{null_count} empty or null values found in rows: {', '.join(map(str, null_indices))}. <br><br>If this is unexpected, please fix your spreadsheet before trying again.")

            # Handle invalid non-null numeric values
            if 'invalid_values' in issues:
                invalid_values = issues['invalid_values']
                st.error(f"{len(invalid_values)} non-numeric values found")

                corrections = {} # Store corrections for the current numeric column
                for idx, value in invalid_values.items():
                    # Prompt user to input a correct numeric value
                    correct_value = st.text_input(
                        f"Correct value for '{value}' in row {idx}:",
                        key=f"{col}_{idx}" # Unique key for each input widget
                    )
                    if correct_value:
                        try:
                            float(correct_value) # Attempt to convert to float to validate
                            corrections[value] = correct_value
                        except ValueError:
                            st.error(f"'{correct_value}' is not a valid numeric value.") # Display error if not numeric

                if corrections:
                    all_corrections[col] = corrections # Add to overall corrections

            # If no issues were found for this numeric column
            if not any(issues): # Check if the 'issues' dict is empty (no nulls or invalid values)
                st.success(f"All values in {col} are valid!")

    # Function to apply all gathered corrections to the DataFrame
    def apply_corrections():
        st.session_state.applying_corrections = True # Set flag to indicate corrections are being applied
        for col, correct_dict in all_corrections.items():
            df[col] = df[col].replace(correct_dict) # Replace values in the DataFrame
        st.session_state.df = df # Update DataFrame in session state
        st.success("Corrections applied successfully!")
        st.rerun() # Rerun to reflect changes and potentially re-evaluate validations

    # Display "Apply All Corrections" button if any corrections are pending
    if all_corrections:
        if st.button("Apply All Corrections"):
            apply_corrections()

    # Clear the 'applying_corrections' flag after a rerun or if not in the process
    if 'applying_corrections' in st.session_state:
        del st.session_state.applying_corrections

    # Only show "Next step" button if no remaining corrections are needed from the user
    remaining_corrections = {k: v for k, v in all_corrections.items()
                             if not k.startswith('skip_')} # Exclude any 'skip' flags from consideration
    if not remaining_corrections:
        st.success("All race, ethnicity and age data is valid!")
        if st.button("Next step"):
            # Clear temporary skip flags before moving to the next step
            for key in list(st.session_state.keys()):
                if key.startswith('skip_'):
                    del st.session_state[key]
            st.session_state.step = 5 # Advance to the next step
            st.rerun()

# Step 5: Primary Site Validation
elif st.session_state.step == 5:
    st.subheader("Step 5: Validate Primary Site")
    df = st.session_state.df # Retrieve DataFrame from session state

    if 'Primary Site' not in df.columns:
        st.info("No Primary Site column found in the data. Proceeding to next step.")
        if st.button("Next step"):
            st.session_state.step = 6 # Skip to next step if column is missing
            st.rerun()
    else:
        # Initialize session state for Primary Site mapping
        if 'primary_site_mapped' not in st.session_state:
            st.session_state.primary_site_mapped = False # Flag if mapping is complete
        if 'primary_site_mappings' not in st.session_state:
            st.session_state.primary_site_mappings = {} # Stores user-selected mappings

        # Get unique invalid values in 'Primary Site' column
        invalid_values = df[~df['Primary Site'].isin(permissible_primary_site)]['Primary Site'].unique()

        if len(invalid_values) == 0:
            st.success("All Primary Site values are valid!")
            if st.button("Next step"):
                st.session_state.step = 6 # Advance if all valid
                st.rerun()
        else:
            # If invalid values exist and mapping hasn't been confirmed yet
            if not st.session_state.primary_site_mapped:
                st.markdown(f"#### Found {len(invalid_values)} non-standard Primary Site values")

                mappings = {} # Temporary dictionary for current mapping selections
                for value in invalid_values:
                    # Get prioritized suggestions for the invalid value
                    options = get_prioritized_options(value, permissible_primary_site)

                    # Selectbox for user to map invalid value to a valid one
                    selected_value = st.selectbox(
                        f"Map '{value}' to:",
                        options=options,
                        key=f"primary_site_{value}" # Unique key for widget
                    )

                    if selected_value != 'Keep current value': # If user selected a new value
                        mappings[value] = selected_value

                # Button to confirm and apply Primary Site mappings
                if st.button("Confirm Primary Site mappings"):
                    st.session_state.primary_site_mappings = mappings # Store mappings
                    st.session_state.primary_site_mapped = True # Mark as mapped

                    if mappings: # Apply mappings to DataFrame if any were made
                        df['Primary Site'] = df['Primary Site'].replace(mappings)
                        st.session_state.df = df # Update DataFrame

                    st.rerun() # Rerun to show summary or next step
            else:
                # If mappings have been confirmed, show a summary of what was mapped/kept
                st.markdown("#### Primary Site Mapping Summary:")

                # Categorize values for summary display
                to_keep = [val for val in invalid_values if val not in st.session_state.primary_site_mappings]
                to_remap = st.session_state.primary_site_mappings

                if to_keep:
                    st.info(f"Values to keep unchanged: {', '.join(f'`{val}`' for val in to_keep)}")
                if to_remap:
                    remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                    st.info(f"Values that were remapped: {', '.join(remap_summary)}")

                # Buttons for user to either map more values or proceed
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Map additional values"):
                        st.session_state.primary_site_mapped = False # Reset flag to show mapping interface again
                        st.session_state.primary_site_mappings = {} # Clear previous mappings
                        st.rerun()
                with col2:
                    if st.button("Next step"):
                        st.session_state.step = 6 # Advance to next step
                        st.rerun()

# Step 6: Primary Diagnosis Validation
elif st.session_state.step == 6:
    st.subheader("Step 6: Validate Primary Diagnosis")
    df = st.session_state.df # Retrieve DataFrame from session state

    if 'Primary Diagnosis' not in df.columns:
        st.info("No Primary Diagnosis column found in the data. Proceeding to next step.")
        if st.button("Next step"):
            st.session_state.step = 7 # Skip to next step if column is missing
            st.rerun()
    else:
        # Initialize session state for Primary Diagnosis mapping
        if 'primary_diagnosis_mapped' not in st.session_state:
            st.session_state.primary_diagnosis_mapped = False # Flag if mapping is complete
        if 'primary_diagnosis_mappings' not in st.session_state:
            st.session_state.primary_diagnosis_mappings = {} # Stores user-selected mappings

        # Get unique invalid values in 'Primary Diagnosis' column
        invalid_values = df[~df['Primary Diagnosis'].isin(permissible_primary_diagnosis)]['Primary Diagnosis'].unique()

        if len(invalid_values) == 0:
            st.success("All Primary Diagnosis values are valid!")
            if st.button("Next step"):
                st.session_state.step = 7 # Advance if all valid
                st.rerun()
        else:
            # If invalid values exist and mapping hasn't been confirmed yet
            if not st.session_state.primary_diagnosis_mapped:
                st.markdown(f"#### Found {len(invalid_values)} non-standard Primary Diagnosis values")

                mappings = {} # Temporary dictionary for current mapping selections
                for value in invalid_values:
                    # Get prioritized suggestions for the invalid value
                    options = get_prioritized_options(value, permissible_primary_diagnosis)

                    # Selectbox for user to map invalid value to a valid one
                    selected_value = st.selectbox(
                        f"Map '{value}' to:",
                        options=options,
                        key=f"primary_diagnosis_{value}" # Unique key for widget
                    )

                    if selected_value != 'Keep current value': # If user selected a new value
                        mappings[value] = selected_value

                # Button to confirm and apply Primary Diagnosis mappings
                if st.button("Confirm Primary Diagnosis mappings"):
                    st.session_state.primary_diagnosis_mappings = mappings # Store mappings
                    st.session_state.primary_diagnosis_mapped = True # Mark as mapped

                    if mappings: # Apply mappings to DataFrame if any were made
                        df['Primary Diagnosis'] = df['Primary Diagnosis'].replace(mappings)
                        st.session_state.df = df # Update DataFrame

                    st.rerun() # Rerun to show summary or next step
            else:
                # If mappings have been confirmed, show a summary of what was mapped/kept
                st.markdown("#### Primary Diagnosis Mapping Summary:")

                # Categorize values for summary display
                to_keep = [val for val in invalid_values if val not in st.session_state.primary_diagnosis_mappings]
                to_remap = st.session_state.primary_diagnosis_mappings

                if to_keep:
                    st.info(f"Values to keep unchanged: {', '.join(f'`{val}`' for val in to_keep)}")
                if to_remap:
                    remap_summary = [f"`{old}` → `{new}`" for old, new in to_remap.items()]
                    st.info(f"Values that were remapped: {', '.join(remap_summary)}")

                # Buttons for user to either reset mappings or proceed
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reset mappings"): # Option to restart mapping for this step
                        st.session_state.primary_diagnosis_mapped = False
                        st.session_state.primary_diagnosis_mappings = {}
                        st.rerun()
                with col2:
                    if st.button("Next step"): # Advance to next step
                        st.session_state.step = 7
                        st.rerun()

# Step 7: Download Standardized Data
elif st.session_state.step == 7:
    st.subheader("Step 7: Download Standardized Data")
    df = st.session_state.df # Retrieve the final, standardized DataFrame from session state

    # Generate a default filename based on the first 'Project Short Name' in the data
    default_filename = f"{df['Project Short Name'].iloc[0]}-Clinical-Standardized.xlsx"

    # Allow user to customize the filename via a text input, pre-filled with the default
    custom_filename = st.text_input(
        "Filename:",
        value=default_filename,
        help="You can modify the filename if desired"
    )
    st.markdown("You must press ENTER after setting a new file name.")

    # Ensure the custom filename always ends with .xlsx
    if not custom_filename.endswith('.xlsx'):
        custom_filename += '.xlsx'

    # Reorder columns of the DataFrame to the preferred TCIA standard order
    df = reorder_columns(df)
    output = BytesIO() # Create an in-memory binary stream to write the Excel file

    # Check if there were other sheets from the original Excel file to include in the output
    if st.session_state.other_sheets:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write the main standardized DataFrame to a sheet named 'Standardized Data'
            df.to_excel(writer, sheet_name='Standardized Data', index=False)
            # Write any other preserved sheets from the original file
            for sheet_name, sheet_data in st.session_state.other_sheets.items():
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        st.info("The downloaded file will include your standardized data sheet along with all other sheets from the original file.")
    else:
        # If no other sheets, just write the main DataFrame to the Excel file
        df.to_excel(output, index=False)

    # Streamlit download button to allow user to download the generated Excel file
    st.download_button(
        "Download Standardized XLSX file",
        data=output.getvalue(), # Get the bytes from the BytesIO stream
        file_name=custom_filename, # Use the user-defined filename
        help="Download the standardized data in Excel format"
    )

    # Button to restart the application workflow
    if st.button("Restart"):
        reset_session_state() # Call the function to clear all session state
        st.rerun() # Rerun the app to go back to Step 1

