import pathlib
import textwrap
import time
import os
import google.generativeai as genai
import streamlit as st
from IPython import display
from IPython.display import Markdown
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


import pandas as pd
import os


def find_schema_differences(file1_name: str, file2_name: str):
    """
    This function simulates Gemini for finding schema differences
    between two dataframes loaded from CSV files.

    Args:
        file1_name (str): Name of the first CSV file.
        file2_name (str): Name of the second CSV file.

    Returns:
        str: A string describing the schema differences,
            or an empty string if the files are not found or identical.
    """
    folder_path = os.getenv("folder_path")

    # Simulate searching
    file1_path = os.path.join(folder_path, file1_name)
    file2_path = os.path.join(folder_path, file2_name)
    if not (os.path.exists(file1_path) and os.path.exists(file2_path)):
        return "Files not found."

    try:
        df1 = pd.read_csv(file1_path)
    except pd.errors.ParserError as e:
        return f"Error parsing file {file1_path}: {e}"

    try:
        df2 = pd.read_csv(file2_path)
    except pd.errors.ParserError as e:
        return f"Error parsing file {file2_path}: {e}"

    # Find all columns present in either dataframe
    all_columns = set(df1.columns).union(set(df2.columns))

    # Find missing columns in each dataframe
    missing_in_df1 = all_columns - set(df1.columns)
    missing_in_df2 = all_columns - set(df2.columns)

    # Generate report on missing columns
    differences = []
    if missing_in_df1:
        differences.append(
            f"Missing columns in {file1_name}: {', '.join(missing_in_df1)}"
        )
    if missing_in_df2:
        differences.append(
            f"Missing columns in {file2_name}: {', '.join(missing_in_df2)}"
        )

    return (
        "\n".join(differences) if differences else "Dataframes have identical schemas."
    )


def goto_next_sub_directory(subdirectory: str):
    folder_path = os.getenv("folder_path")
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return False
    subdirectory_path = os.path.join(folder_path, subdirectory)
    os.environ["folder_path"] = subdirectory_path
    print(os.getenv("folder_path"))
    return True


def goto_previous_directory():
    folder_path = os.getenv("folder_path")
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return False


def list_all_files():
    """
    This function lists files recursively within a folder and its subfolders,
    printing the subfolder name and filename.

    Args:
        folder_path (str): Path to the folder to search.
    """

    folder_path = os.getenv("folder_path")

    all_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # Construct relative path within subfolder
            relative_path = os.path.join(os.path.relpath(root, folder_path), filename)
            all_files.append(relative_path)
    return all_files


def read_csv_data(filename: str, num_records: int):
    """
    This function reads a CSV file into a DataFrame and returns the specified number of records.

    Args:
        filename: The path to the CSV file.
        num_records: The number of records to display.

    Returns:
        A Pandas DataFrame containing the specified number of records from the beginning of the file.
    """
    folder_path = os.getenv("folder_path")

    num_records = int(num_records)

    file_path = os.path.join(folder_path, filename)

    try:
        # Read the CSV file into a DataFrame
        initial_df = pd.read_csv(file_path, nrows=10)

        inferred_dtypes = initial_df.dtypes.to_dict()

        for col, dtype in inferred_dtypes.items():
            if dtype == "object":
                inferred_dtypes[col] = "str"

        data = pd.read_csv(file_path, dtype=inferred_dtypes)

        if num_records > len(data):
            print(
                f"Requested number of records ({num_records}) exceeds the total number of records ({len(data)})."
            )
            print(f"Returning all {len(data)} records instead.")
            num_records = len(data)

        # Return the specified number of records as a DataFrame
        return data.head(num_records)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None


model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[
        find_schema_differences,
        list_all_files,
        read_csv_data,
        goto_next_sub_directory,
    ],
)

chat = model.start_chat(enable_automatic_function_calling=True)

# response = chat.send_message(
#     "what is the schema difference between  employees.csv and employees.csv in subFolder?"
# )
# print(response.text)
st.title("Find schema difference between two files")
prompt = st.text_area("Enter your prompt")
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write(chat.send_message(prompt).text)
    else:
        st.write("Please enter a prompt")
