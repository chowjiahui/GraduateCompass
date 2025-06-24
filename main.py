import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
import google.generativeai as genai
from pydantic import BaseModel, Field

import os
import time


# --- Pydantic Model for Structured Data (Our "Tool") ---
# We define the structure of the data we want the LLM to extract.
# This is our "function" for the LLM's tool-use feature.
class Profile(BaseModel):
    """
    A class to hold extracted information about a person's profile.
    """
    name: str = Field(description="The full name of the person.")
    job_title: str = Field(description="The current job title of the person.")
    company: str = Field(description="The current company the person works at.")


# --- Helper Functions ---

def setup_api_keys():
    """
    Sets up API keys from Streamlit secrets.
    Handles potential errors if keys are not found.
    """
    try:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=google_api_key)
        return True
    except KeyError as e:
        st.error(f"API Key not found in secrets: {e}. Please add it to your .streamlit/secrets.toml file.")
        return False


def search_linkedin_profiles(university, program, num_pages):
    """
    Uses SerpApi to search for LinkedIn profiles, handling pagination for deep search.
    """
    all_results = []
    query = f'linkedin "{program}" "{university}" graduate'
    st.write(f"🔍 **Performing search with query:** `{query}`")

    for page in range(num_pages):
        start_index = page * 10
        st.info(f"Fetching page {page + 1} of {num_pages} from Google...")

        params = {
            "api_key": os.environ.get("SERP_API_KEY"),
            "engine": "google",
            "q": query,
            "start": start_index,
            "num": 10  # Google returns 10 results per page by default
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            if "error" in results:
                st.error(f"SerpApi Error: {results['error']}")
                break  # Stop searching if there's an error

            if "organic_results" in results:
                all_results.extend(results["organic_results"])
            else:
                # No more results to fetch
                st.warning("No more results found. Ending search.")
                break

        except Exception as e:
            st.error(f"An exception occurred during the SerpApi search: {e}")
            break

        time.sleep(1)  # Be a good citizen and don't hammer the API

    return all_results


def extract_profile_data(gemini_model, result):
    """
    Uses Gemini 2.5 Flash with Tool Use to extract structured data from a text snippet.
    """
    # The snippet is often more useful than the raw content for extraction
    text_to_parse = f"{result.get('title', '')} - {result.get('snippet', '')}"
    link = result.get('link', '')

    try:
        response = gemini_model.generate_content(
            f"Extract the person's name, job title, and company from the following text. "
            f"Text: {text_to_parse}",
            tools=[Profile]
        )
        tool_call = response.candidates[0].content.parts[0].function_call

        if tool_call:
            args = tool_call.args
            profile_data = {
                "name": args.get("name", "N/A"),
                "job_title": args.get("job_title", "N/A"),
                "company": args.get("company", "N/A"),
                "linkedin_url": link
            }
            return profile_data
    except Exception as e:
        # This will catch errors if the model fails to generate a valid tool call
        print(f"Error parsing text with Gemini: {text_to_parse} | Error: {e}")  # Log to console for debugging
    return None


# --- Main Streamlit App ---

st.set_page_config(page_title="Graduate Career Compass", layout="wide")
st.title("🎓 Graduate Career Compass")
st.markdown("Analyze career outcomes of US Master's programs based on public LinkedIn data.")

# Set up API keys using Streamlit Secrets
keys_ready = setup_api_keys()
if keys_ready:
    with st.sidebar:
        st.header("Search Criteria")
        university_input = st.text_input("University Name", placeholder="e.g., Stanford University")
        program_input = st.text_input("Master's Program Name",
                                      placeholder="e.g., Master of Science in Computer Science")

        # Add a slider to control search depth
        search_pages = st.slider(
            "Search Depth (Pages)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many pages of Google results to analyze. More pages = more data but longer processing time and higher cost."
        )

        analyze_button = st.button("Analyze Career Outcomes", type="primary")

    if analyze_button and university_input and program_input:
        st.session_state.clear()
        st.session_state.profiles = []

        with st.status("Analyzing...", expanded=True) as status:
            search_results = search_linkedin_profiles(university_input, program_input, search_pages)

            if not search_results:
                status.update(label="No results found. Try broadening your search terms.", state="error")
            else:
                gemini_flash_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", tools=[Profile])
                status.update(label=f"Found {len(search_results)} potential profiles. Now extracting data with AI...")

                progress_bar = st.progress(0, text="Extracting profile data...")

                for i, result in enumerate(search_results):
                    profile = extract_profile_data(gemini_flash_model, result)
                    if profile:
                        st.session_state.profiles.append(profile)

                    time.sleep(0.2)
                    progress_bar.progress((i + 1) / len(search_results),
                                          text=f"Processing profile {i + 1}/{len(search_results)}")

                status.update(label="Analysis complete!", state="complete")

    if 'profiles' in st.session_state and st.session_state.profiles:
        df = pd.DataFrame(st.session_state.profiles)
        df.replace('N/A', pd.NA, inplace=True)
        df.dropna(subset=['company', 'job_title'], inplace=True)

        if df.empty:
            st.warning("Could not extract enough valid data to generate a report.")
        else:
            st.header(f"Results for {program_input} at {university_input}", divider="rainbow")
            st.info(f"Successfully extracted and cleaned data for **{len(df)}** profiles.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 15 Companies")
                company_counts = df['company'].value_counts().nlargest(15)
                st.dataframe(company_counts)
            with col2:
                st.subheader("Top 15 Job Roles")
                role_counts = df['job_title'].value_counts().nlargest(15)
                st.dataframe(role_counts)

            st.subheader("Extracted Graduate Profiles")
            st.dataframe(
                df,
                column_config={
                    "linkedin_url": st.column_config.LinkColumn("LinkedIn Profile", display_text="🔗 View Profile")},
                hide_index=True,
                use_container_width=True
            )
else:
    st.warning("Please configure your API keys in the `.streamlit/secrets.toml` file to use the app.")