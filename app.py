import pandas as pd
import openai
import time
import os
import re
import json
import backoff
import nest_asyncio
import tiktoken
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import io
import base64
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI, OpenAIError, RateLimitError

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("screening.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Apply nest_asyncio for async operation in notebooks/Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Daily Vlog Video Screener",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the top logo
st.image("labda_logo.png", width=600)

# Title and description
st.title("📹 Daily Vlog Video Screener")
st.markdown("""
This tool automatically screens YouTube videos to identify genuine "day in the life" or daily vlog content.
Upload your Excel file containing YouTube video metadata, and the tool will analyze each video
using advanced AI to determine if it meets the inclusion criteria.
""")

# Initialize session state for keyboard shortcuts and other stateful features
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = []
if 'custom_categories' not in st.session_state:
    st.session_state.custom_categories = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'keyboard_shortcuts_enabled' not in st.session_state:
    st.session_state.keyboard_shortcuts_enabled = False
if 'notification_email' not in st.session_state:
    st.session_state.notification_email = ""
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0

# Sidebar for configuration
st.sidebar.header("Configuration")

# API Key input (with warning)
# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
#                                      help="Required to use the OpenAI GPT models for screening")


# Use pre-configured API key for the research consortium
import os
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
st.sidebar.success("✅ API key pre-configured for consortium members")

# Model selection
model_name = st.sidebar.selectbox(
    "Select OpenAI Model", 
    options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0,
    help="GPT-4o provides the best results but costs more. GPT-4o-mini is a good balance."
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    max_tokens = st.slider("Max Tokens", min_value=1000, max_value=8000, value=3000, step=500,
                          help="Maximum number of tokens to use in each prompt (including transcript)")
    batch_size = st.slider("Batch Size", min_value=5, max_value=50, value=20, step=5,
                          help="Number of videos to process in each batch")
    workers = st.slider("Concurrent Workers", min_value=1, max_value=5, value=3, step=1,
                       help="Number of videos to process concurrently. Higher values may hit rate limits.")
    debug_mode = st.checkbox("Debug Mode", value=False, 
                            help="Show detailed logs and error messages")
    st.session_state.keyboard_shortcuts_enabled = st.checkbox("Enable Keyboard Shortcuts", value=False,
                                                           help="Enable keyboard shortcuts for power users")

# Email notification settings
with st.sidebar.expander("Notification Settings"):
    notification_enabled = st.checkbox("Enable Email Notifications", value=False,
                                     help="Send email notifications when processing completes")
    if notification_enabled:
        st.session_state.notification_email = st.text_input("Email Address", 
                                                         value=st.session_state.notification_email,
                                                         help="Email to receive notifications")
        smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com", 
                                  help="SMTP server for sending emails")
        smtp_port = st.number_input("SMTP Port", value=587, help="SMTP server port")
        smtp_username = st.text_input("SMTP Username", help="SMTP username (often your email)")
        smtp_password = st.text_input("SMTP Password", type="password", 
                                    help="SMTP password or app password")
        
# Custom categories
with st.sidebar.expander("Custom Categories"):
    new_category = st.text_input("Add New Category", help="Add a custom category beyond INCLUDE/NOT SURE/EXCLUDE")
    if st.button("Add Category") and new_category and new_category not in st.session_state.custom_categories:
        st.session_state.custom_categories.append(new_category)
        
    st.write("Current Custom Categories:")
    for i, category in enumerate(st.session_state.custom_categories):
        col1, col2 = st.columns([3, 1])
        col1.write(f"• {category}")
        if col2.button("Remove", key=f"remove_{i}"):
            st.session_state.custom_categories.remove(category)
            st.experimental_rerun()

# Bulk actions
with st.sidebar.expander("Bulk Actions", expanded=False):
    if st.session_state.final_df is not None:
        st.write(f"Selected: {len(st.session_state.selected_videos)} videos")
        
        # Bulk action options
        bulk_action = st.selectbox("Select Action", 
                                 options=["Change Category", "Download Selected", "Remove Selected"])
        
        if bulk_action == "Change Category":
            # Get all available categories including custom ones
            all_categories = ["INCLUDE", "NOT SURE", "EXCLUDE"] + st.session_state.custom_categories
            new_category = st.selectbox("New Category", options=all_categories)
            
            if st.button("Apply Change") and st.session_state.selected_videos:
                # Apply the category change to selected videos
                if st.session_state.final_df is not None:
                    for video_id in st.session_state.selected_videos:
                        idx = st.session_state.final_df[st.session_state.final_df['id'] == video_id].index
                        if len(idx) > 0:
                            st.session_state.final_df.loc[idx, 'decision'] = new_category
                    st.success(f"Changed {len(st.session_state.selected_videos)} videos to {new_category}")
                    # Clear selections after action
                    st.session_state.selected_videos = []
                    st.experimental_rerun()
                    
        elif bulk_action == "Download Selected" and st.button("Download"):
            if st.session_state.selected_videos and st.session_state.final_df is not None:
                selected_df = st.session_state.final_df[st.session_state.final_df['id'].isin(st.session_state.selected_videos)]
                if not selected_df.empty:
                    # Generate download link for selected videos
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        selected_df.to_excel(writer, index=False)
                    buffer.seek(0)
                    b64 = base64.b64encode(buffer.read()).decode()
                    download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="selected_videos.xlsx">Download Selected Videos</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
        
        elif bulk_action == "Remove Selected" and st.button("Remove"):
            if st.session_state.selected_videos and st.session_state.final_df is not None:
                # Remove selected videos from the dataframe
                st.session_state.final_df = st.session_state.final_df[~st.session_state.final_df['id'].isin(st.session_state.selected_videos)]
                st.success(f"Removed {len(st.session_state.selected_videos)} videos")
                # Clear selections after action
                st.session_state.selected_videos = []
                st.experimental_rerun()
    else:
        st.write("Bulk actions will be available after processing videos")

# Caching settings
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
transcript_cache_file = os.path.join(cache_dir, 'transcript_cache.json')

# Load transcript cache if it exists
transcript_cache = {}
if os.path.exists(transcript_cache_file):
    try:
        with open(transcript_cache_file, 'r', encoding='utf-8') as f:
            transcript_cache = json.load(f)
        st.sidebar.info(f"Loaded {len(transcript_cache)} transcripts from cache")
    except Exception as e:
        st.sidebar.warning(f"Could not load transcript cache: {e}")

# Clear cache button
if st.sidebar.button("Clear Cache"):
    try:
        # Clear transcript cache
        if os.path.exists(transcript_cache_file):
            os.remove(transcript_cache_file)
            
        # Clear batch results
        for file in os.listdir(cache_dir):
            if file.startswith('batch_') or file.endswith('.xlsx'):
                os.remove(os.path.join(cache_dir, file))
                
        # Clear progress file
        if os.path.exists('screened_youtube_videos_progress.xlsx'):
            os.remove('screened_youtube_videos_progress.xlsx')
            
        # Reset transcript cache dictionary
        transcript_cache = {}
        
        st.sidebar.success("Cache cleared successfully!")
    except Exception as e:
        st.sidebar.error(f"Error clearing cache: {e}")

# Add a usage counter in the sidebar:
st.sidebar.info(f"API calls this session: {st.session_state.api_calls}")

# Helper functions
def count_tokens(text, model_name):
    """Count tokens in a string using the appropriate tokenizer."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback to approximate token count
        return len(text.split()) * 1.3

def truncate_to_token_limit(text, max_tokens=2000, model_name="gpt-4o-mini"):
    """Truncate text to token limit while preserving sentence boundaries."""
    if not text:
        return ""
    
    # First check if we're already under the limit
    token_count = count_tokens(text, model_name)
    if token_count <= max_tokens:
        return text
    
    # If we need to truncate, do it by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    current_token_count = 0
    
    for sentence in sentences:
        sentence_token_count = count_tokens(sentence, model_name)
        if current_token_count + sentence_token_count <= max_tokens:
            result += sentence + " "
            current_token_count += sentence_token_count
        else:
            break
    
    return result.strip() + " [truncated]"

def extract_video_id(url):
    """Extract the YouTube video ID from a URL."""
    # Regular expression pattern for YouTube URLs
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
        r'(?:youtube\.com\/shorts\/)([^&\n?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

async def get_transcript(video_id):
    """Get transcript using youtube_transcript_api without YouTube API."""
    # Check cache first
    if video_id in transcript_cache:
        return transcript_cache[video_id]
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=['en', 'nl', 'fr', 'sw', 'auto']  # Include auto-generated
        )
        
        # Combine transcript pieces into a single text
        full_transcript = " ".join([entry['text'] for entry in transcript_list])
        
        # Cache the result
        transcript_cache[video_id] = full_transcript
        
        # Periodically save cache
        if len(transcript_cache) % 10 == 0:
            with open(transcript_cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_cache, f, ensure_ascii=False)
        
        return full_transcript
    except Exception as e:
        logger.warning(f"Could not get transcript for video {video_id}: {e}")
        # Cache the negative result too to avoid retrying
        transcript_cache[video_id] = None
        return None

# Create a robust chat function with retries
@backoff.on_exception(
    backoff.expo, 
    (OpenAIError, RateLimitError, Exception), 
    max_tries=5,
    on_backoff=lambda details: logger.warning(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries calling OpenAI API"
    )
)
def chat_with_openai(api_key, messages, model="gpt-4o-mini"):
    """Robust chat function with retries and error handling."""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Increment API call counter ---------API Tracking
        st.session_state.api_calls += 1

        # Get the content and immediately try to parse it as JSON
        content = response.choices[0].message.content
        
        # Log the raw response for debugging
        if debug_mode:
            logger.info(f"Raw OpenAI response: {content}")
        
        # Try to parse the JSON with improved error handling
        try:
            parsed_json = json.loads(content)
            
            # Verify that parsed_json is a dictionary
            if not isinstance(parsed_json, dict):
                logger.error(f"Expected dictionary but got {type(parsed_json)}")
                return {
                    "decision": "NOT SURE",
                    "confidence": 3,
                    "cues_found": ["Error: Invalid JSON structure"],
                    "exclusion_evidence": [],
                    "reasoning": "The LLM response was not in the expected format."
                }
                
            return parsed_json
            
        except json.JSONDecodeError as e:
            # More detailed logging of the error
            error_position = f"line {e.lineno}, column {e.colno}"
            logger.error(f"JSON parse error at {error_position}: {e}")
            logger.error(f"Raw content causing error: {content}")
            
            # Return a default structured response instead of raising
            return {
                "decision": "NOT SURE",
                "confidence": 1,
                "cues_found": ["JSON parse error"],
                "exclusion_evidence": [],
                "reasoning": f"Failed to parse LLM response: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        # Return a default structured response instead of raising
        return {
            "decision": "NOT SURE",
            "confidence": 1,
            "cues_found": ["API call error"],
            "exclusion_evidence": [],
            "reasoning": f"OpenAI API call failed: {str(e)}"
        }

async def screen_video_with_llm(video_data, transcript=None, model_name="gpt-4o-mini"):
    """Use LLM to determine if a video meets the inclusion criteria."""
    # Extract basic metadata
    video_id = video_data.get("id", "Unknown")
    title = video_data.get("title", "Unknown")
    
    logger.info(f"Processing video: {video_id} - {title}")
    
    try:
        # Prepare the prompt with video information
        prompt = f"""
        I need to determine if this YouTube video is a genuine "day in the life" or daily vlog video.

        Here's the information about the video:
        
        Title: {video_data.get("title", "Unknown")}
        Description: {video_data.get("text", "No description available")[:800] + "..." if video_data.get("text") and len(video_data.get("text", "")) > 800 else video_data.get("text", "No description available")}
        Duration: {video_data.get("duration", "Unknown")}
        Channel Name: {video_data.get("channelName", "Unknown")}
        """
        
        # Add hashtags if available
        hashtags = []
        for i in range(10):  # Check for up to 10 hashtags
            hashtag_key = f"hashtags/{i}"
            if hashtag_key in video_data and video_data[hashtag_key] and not pd.isna(video_data[hashtag_key]):
                hashtags.append(str(video_data[hashtag_key]))
        
        if hashtags:
            prompt += f"\nHashtags: {', '.join(hashtags)}\n"
        
        # Add transcript if available (shortened)
        if transcript:
            transcript_snippet = transcript[:500] + "..." if len(transcript) > 500 else transcript
            prompt += f"\nTranscript snippet: {transcript_snippet}\n"
        
        # Add the vlog screener prompt
        prompt += """
        ################################################################
        #  HIGH-PRECISION VLOG SCREENER – MAXIMUM SPECIFICITY (v1-2025) #
        ################################################################
        
        TASK: You are an expert screener for daily vlog content with MAXIMUM PRECISION.
        Your primary goal is to ONLY include videos that are DEFINITIVELY "day in the life" vlogs.
        
        CONTEXT:
        - Prioritize SPECIFICITY over sensitivity - it is better to miss potentially eligible videos than to include non-vlogs
        - When evaluating sparse data, absence of clear vlog indicators should be treated as evidence AGAINST inclusion
        - When in doubt, ALWAYS favor EXCLUSION - false negatives are STRONGLY PREFERRED over false positives
        - Only include videos when multiple strong indicators are present and aligned
        
        ------------------------------------
        DECISION FRAMEWORK:
        
        - **INCLUDE** ONLY when ALL of these apply:
          - EXPLICIT single-day narrative with clear temporal progression
          - Strong first-person language ("I", "my", "me") PLUS multiple specific daily activities
          - Video structure follows a chronological sequence of ordinary activities
          - Personal/individual channel with established vlogging history
          - Duration typically between 10-30 minutes (sweet spot for genuine day vlogs)
          - Contains at least 3 distinct indicators of authentic daily life content
          - NO significant contradictory evidence present
        
        - **NOT SURE** when:
          - Strong indicators present but missing critical elements
          - Mixed content where daily vlog elements compete with other formats
          - Temporal structure unclear but personal narrative exists
          - Duration and format align with vlogs but content focus is ambiguous
          - Contains both inclusion and exclusion evidence of similar strength
        
        - **EXCLUDE** when ANY of these apply (30-40% certainty is sufficient):
          - No explicit indication of single-day narrative
          - Spans multiple days or summarizes a period longer than 24 hours
          - Primarily focused on a single activity, event, or topic rather than daily life
          - Primarily tutorial, reaction, review, or entertainment-focused content
          - Professional/commercial production style rather than personal documentation
          - Duration under 5 minutes or over 45 minutes without clear vlog structure
          - Holiday, travel, or tourism content focused on attractions rather than personal experience
          - Primarily showcasing products, locations, or experiences without daily life context
          - Channel primarily produces non-vlog content (gaming, educational, news, etc.)
        
        ------------------------------------
        NON-VLOG INDICATORS ACROSS LANGUAGES
        (Apply to ALL languages)
        
        - Format indicators: tutorial format, reaction video, product review, news report, commentary
        - Focus indicators: single-topic focus, specialized content, professional presentation
        - Production indicators: highly edited, scripted delivery, professional equipment/lighting
        - Commercial indicators: sponsored content, brand deals as primary focus
        - Event indicators: special occasions, one-time experiences, unusual activities
        
        ------------------------------------
        TRUE VLOG CONFIRMATION CHECKLIST
        (Require multiple items for INCLUSION)
        
        1. Temporal markers: Morning → afternoon → evening progression
        2. Mundane activities: Multiple ordinary activities (eating, commuting, working)
        3. Personal perspective: First-person narration throughout with authentic reactions
        4. Setting variety: Multiple everyday locations (home, work, errands)
        5. Natural transitions: Logical flow between daily activities
        6. Balance of planned/unplanned: Mix of routine and spontaneous moments
        7. Personal context: References to ongoing life circumstances
        8. Authentic interaction: Natural engagement with environment/others
        
        ------------------------------------
        CRUCIAL GUIDANCE:
        - Require STRONG EVIDENCE from MULTIPLE categories for inclusion
        - When information is limited, DEFAULT to NOT SURE or EXCLUDE
        - A single exclusion indicator outweighs multiple weak inclusion indicators
        - Apply a LOW threshold for exclusion (30-40% certainty is sufficient)
        - Apply extremely HIGH threshold for inclusion (80%+ certainty required)
        - Classification requires EXPLICIT positive evidence, not absence of negative evidence
        
        ------------------------------------
        OUTPUT (JSON)
        
        {
          "decision": "INCLUDE|NOT SURE|EXCLUDE",
          "confidence": 1-5,
          "inclusion_factors": ["list", "of", "confirmed", "vlog", "indicators"],
          "exclusion_factors": ["list", "of", "non-vlog", "indicators"],
          "evaluation": "Detailed analysis of specific evidence considered and how it was weighed"
        }
        
        REMEMBER: The goal is PRECISION - only include videos that are definitively day-in-life vlogs with high confidence.
        """   
       
        # Call OpenAI API with error handling and retries
        try:
            json_result = chat_with_openai(openai_api_key, [{"role": "user", "content": prompt}], model_name)
            
            # Process each field individually and convert to appropriate types
            processed_result = {}
            
            # Decision (string)
            if "decision" in json_result:
                processed_result["decision"] = str(json_result["decision"]).upper()
                if processed_result["decision"] not in ["INCLUDE", "EXCLUDE", "NOT SURE"]:
                    processed_result["decision"] = "NOT SURE"
            else:
                processed_result["decision"] = "NOT SURE"
            
            # Confidence (integer)
            if "confidence" in json_result:
                try:
                    processed_result["confidence"] = int(float(json_result["confidence"]))
                    processed_result["confidence"] = max(1, min(5, processed_result["confidence"]))
                except (ValueError, TypeError):
                    processed_result["confidence"] = 3
            else:
                processed_result["confidence"] = 3

            # Updated code for Inclusion factors (maps to cues_found in the database):
            if "inclusion_factors" in json_result and json_result["inclusion_factors"]:
                if isinstance(json_result["inclusion_factors"], list):
                    # Join list items into a single string with semicolons
                    cues_list = [str(item) for item in json_result["inclusion_factors"] if item is not None]
                    processed_result["cues_found"] = "; ".join(cues_list)
                else:
                    processed_result["cues_found"] = str(json_result["inclusion_factors"])
            else:
                processed_result["cues_found"] = "No specific inclusion factors identified"
            
            # Updated code for Exclusion factors (maps to exclusion_evidence in the database):
            if "exclusion_factors" in json_result and json_result["exclusion_factors"]:
                if isinstance(json_result["exclusion_factors"], list):
                    # Join list items into a single string with semicolons
                    evidence_list = [str(item) for item in json_result["exclusion_factors"] if item is not None]
                    processed_result["exclusion_evidence"] = "; ".join(evidence_list)
                else:
                    processed_result["exclusion_evidence"] = str(json_result["exclusion_factors"])
            else:
                processed_result["exclusion_evidence"] = ""
            
            # Updated code for Evaluation (maps to reasoning in the database):
            if "evaluation" in json_result and json_result["evaluation"]:
                processed_result["reasoning"] = str(json_result["evaluation"])
            else:
                processed_result["reasoning"] = "No detailed evaluation provided"
          
            logger.info(f"Screening complete for {video_id}: {processed_result['decision']} (confidence: {processed_result['confidence']})")
            return processed_result
            
        except Exception as e:
            # Log the full exception for debugging
            logger.exception(f"Error during OpenAI API call for video {video_id}")
            
            # Return a default response with string fields
            return {
                "decision": "NOT SURE",
                "confidence": 1,
                "cues_found": "Error in API call",
                "exclusion_evidence": "",
                "reasoning": f"Error during processing: {str(e)}"
            }
            
    except Exception as e:
        # Catch any other exceptions in the overall function
        logger.exception(f"Unexpected error processing video {video_id}")
        return {
            "decision": "NOT SURE",
            "confidence": 1,
            "cues_found": "Unexpected error",
            "exclusion_evidence": "",
            "reasoning": f"Unexpected error: {str(e)}"
        }

async def process_video(row, progress_callback=None):
    """Process a single video row from the dataframe."""
    video_id = row["id"]
    video_url = row["url"]
    
    try:
        # Convert row to dictionary for easier handling
        video_data = row.to_dict()
        
        # Extract transcript without using YouTube API
        transcript = None
        extracted_id = extract_video_id(video_url)
        if extracted_id:
            transcript = await get_transcript(extracted_id)
        
        # Screen with LLM using metadata and transcript
        result = await screen_video_with_llm(video_data, transcript, model_name)
        
        # Extract fields from result - result now contains strings, not lists
        result_dict = {
            "id": video_id,
            "title": row["title"],
            "url": video_url,
            "decision": result["decision"],
            "confidence": result["confidence"],
            "cues_found": result["cues_found"],  # Already a string
            "exclusion_evidence": result["exclusion_evidence"],  # Already a string
            "reasoning": result["reasoning"],
            "transcript_available": "Yes" if transcript else "No"
        }
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback()
            
        return result_dict
    except Exception as e:
        logger.exception(f"Error processing {video_id}")
        # Update progress if callback provided
        if progress_callback:
            progress_callback()
            
        return {
            "id": video_id,
            "title": row["title"] if "title" in row else "Unknown",
            "url": video_url,
            "decision": "NOT SURE",
            "confidence": 1,
            "cues_found": "Error processing",  # String, not list
            "exclusion_evidence": "",  # Empty string, not list
            "reasoning": f"Failed to process: {str(e)}",
            "transcript_available": "No"
        }

async def process_batch(batch_df, batch_num, total_batches, progress_bar=None):
    """Process a batch of videos and save results."""
    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_df)} videos")
    
    tasks = []
    
    # Create a callback for updating the progress bar
    def update_progress():
        if progress_bar is not None:
            progress_bar.progress((batch_num - 1) / total_batches + 1 / (total_batches * len(batch_df)))
    
    for _, row in batch_df.iterrows():
        tasks.append(process_video(row, update_progress))
    
    # Process videos concurrently but with controlled concurrency
    semaphore = asyncio.Semaphore(workers)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    bounded_tasks = [process_with_semaphore(task) for task in tasks]
    results = []
    
    for task in asyncio.as_completed(bounded_tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            logger.error(f"Error in task: {e}", exc_info=True)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save batch results
    batch_file = os.path.join(cache_dir, f"batch_{batch_num}_results.csv")
    results_df.to_csv(batch_file, index=False)
    logger.info(f"Saved batch results to {batch_file}")
    
    return results_df

async def process_videos(df, progress_bar=None):
    """Process all videos in the dataframe."""
    # Check for progress file
    progress_file = os.path.join(cache_dir, "screened_youtube_videos_progress.xlsx")
    all_results = []
    
    if os.path.exists(progress_file):
        try:
            progress_df = pd.read_excel(progress_file)
            already_processed = set(progress_df['id'].tolist())
            st.info(f"Found {len(already_processed)} already processed videos")
            all_results.append(progress_df)
            
            # Filter out already processed videos
            df_filtered = df[~df['id'].isin(already_processed)]
            st.info(f"Remaining videos to process: {len(df_filtered)}")
        except Exception as e:
            st.warning(f"Error loading progress file: {e}")
            df_filtered = df
    else:
        df_filtered = df
    
    # Break into smaller batches
    df_batches = [df_filtered[i:i+batch_size] for i in range(0, len(df_filtered), batch_size)]
    num_batches = len(df_batches)
    
    if len(df_filtered) == 0:
        st.success("All videos have already been processed!")
        # Load the final results
        final_df = pd.read_excel(progress_file)
        return final_df
    
    st.info(f"Processing {len(df_filtered)} videos in {num_batches} batches")
    
    for i, batch_df in enumerate(df_batches):
        batch_num = i + 1
        
        # Process this batch
        batch_results = await process_batch(batch_df, batch_num, num_batches, progress_bar)
        all_results.append(batch_results)
        
        # Save progress after each batch
        progress_df = pd.concat(all_results)
        progress_df.to_excel(progress_file, index=False)
        
        if progress_bar:
            progress_bar.progress(batch_num / num_batches)
        
        # Add a delay between batches to avoid rate limits
        if i < len(df_batches) - 1:
            delay = 5
            logger.info(f"Waiting {delay} seconds before starting next batch...")
            await asyncio.sleep(delay)
    
# Combine all results
    if all_results:
        final_results = pd.concat(all_results)
        
        # Merge with original input data to include any columns not in the results
        merge_columns = ['id']
        columns_to_exclude = list(final_results.columns)
        columns_to_include = [col for col in df.columns if col not in columns_to_exclude or col in merge_columns]
        
        # Use left join on final_results to keep only processed videos
        final_df = final_results.merge(
            df[columns_to_include], 
            on="id", 
            how="left"
        )
        
        # Save to new Excel file
        output_file = os.path.join(cache_dir, "screened_youtube_videos_final.xlsx")
        final_df.to_excel(output_file, index=False)
        logger.info(f"Final results saved to {output_file}")
        
        return final_df
    else:
        st.warning("No results were produced!")
        return None

def display_video_example(video_data, width=360, height=200):
    """Display a video thumbnail with basic info and decision."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        video_id = extract_video_id(video_data['url'])
        if video_id:
            # Add checkbox for bulk selection
            is_selected = video_id in st.session_state.selected_videos
            if st.checkbox("Select", value=is_selected, key=f"select_{video_id}"):
                if video_id not in st.session_state.selected_videos:
                    st.session_state.selected_videos.append(video_id)
            else:
                if video_id in st.session_state.selected_videos:
                    st.session_state.selected_videos.remove(video_id)
                    
            # Display thumbnail
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
            st.image(thumbnail_url, use_column_width=True)
            
            # Add embedded YouTube player
            st.write("Preview:")
            st.components.v1.iframe(
                f"https://www.youtube.com/embed/{video_id}", 
                width=250, 
                height=150,
                scrolling=True
            )
        else:
            st.write("No thumbnail available")
    
    with col2:
        st.markdown(f"**{video_data['title']}**")
        
        # Display decision with dropdown for custom categories if present
        all_categories = ["INCLUDE", "NOT SURE", "EXCLUDE"] + st.session_state.custom_categories
        current_decision = video_data['decision']
        new_decision = st.selectbox(
            "Decision", 
            options=all_categories,
            index=all_categories.index(current_decision) if current_decision in all_categories else 0,
            key=f"decision_{video_id}"
        )
        
        # If decision was changed, update it
        if new_decision != current_decision and st.session_state.final_df is not None:
            idx = st.session_state.final_df[st.session_state.final_df['id'] == video_id].index
            if len(idx) > 0:
                st.session_state.final_df.loc[idx, 'decision'] = new_decision
        
        st.markdown(f"**Confidence**: {video_data['confidence']}")
        
        # Display cues if available
        if 'cues_found' in video_data and video_data['cues_found']:
            st.markdown(f"**Cues**: {video_data['cues_found']}")
        
        # Display exclusion evidence if available and not empty
        if 'exclusion_evidence' in video_data and video_data['exclusion_evidence']:
            st.markdown(f"**Exclusion Evidence**: {video_data['exclusion_evidence']}")
        
        st.markdown(f"**Reasoning**: {video_data['reasoning']}")
        if 'transcript_available' in video_data:
            st.markdown(f"**Transcript**: {video_data['transcript_available']}")
        
        # Add a link to the video
        st.markdown(f"[Watch Video]({video_data['url']})")

def send_notification_email(subject, message, to_email):
    """Send notification email when batch processing completes."""
    try:
        # Only send if notification is enabled and SMTP details are provided
        if not (notification_enabled and st.session_state.notification_email and
                smtp_server and smtp_port and smtp_username and smtp_password):
            return False
            
        # Create email
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add message body
        msg.attach(MIMEText(message, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False

def generate_download_link(df, filename="screened_videos.xlsx"):
    """Generate a download link for the dataframe."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href

def debug_llm_responses(df):
    """Analyze why videos are being excluded or marked as 'not sure'."""
    if 'decision' not in df.columns:
        st.error("No decision column found in results")
        return
        
    # Count decisions
    decisions = df['decision'].value_counts()
    st.write("### Decision Breakdown")
    st.write(decisions)
    
    # Calculate percentages
    total = len(df)
    decision_pcts = df['decision'].value_counts(normalize=True) * 100
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99']
    ax.pie(
        decision_pcts, 
        labels=decision_pcts.index, 
        autopct='%1.1f%%', 
        colors=colors,
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)
    
    # Sample excluded videos
    if 'EXCLUDE' in decisions and decisions['EXCLUDE'] > 0:
        excluded = df[df['decision'] == 'EXCLUDE'].sample(min(5, len(df[df['decision'] == 'EXCLUDE'])))
        
        st.write("### Sample Excluded Videos")
        for _, row in excluded.iterrows():
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Reason:** {row.get('reasoning', 'No reason provided')}")
            if 'exclusion_evidence' in row and row['exclusion_evidence']:
                st.write(f"**Exclusion Evidence:** {row['exclusion_evidence']}")
            st.write(f"**Duration:** {row.get('duration', 'Unknown')}")
            st.write(f"**URL:** {row['url']}")
            st.write("---")
    
    # Sample not sure videos
    if 'NOT SURE' in decisions and decisions['NOT SURE'] > 0:
        not_sure = df[df['decision'] == 'NOT SURE'].sample(min(5, len(df[df['decision'] == 'NOT SURE'])))
        
        st.write("### Sample 'Not Sure' Videos")
        for _, row in not_sure.iterrows():
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Reason:** {row.get('reasoning', 'No reason provided')}")
            st.write(f"**Duration:** {row.get('duration', 'Unknown')}")
            st.write(f"**URL:** {row['url']}")
            st.write("---")
    
    # Check for errors
    if 'reasoning' in df.columns:
        error_count = df[df['reasoning'].str.contains('Error', na=False)].shape[0]
        if error_count > 0:
            st.write(f"### Error Analysis")
            st.write(f"Found {error_count} videos with error messages in reasoning.")
            st.write("Sample error messages:")
            
            error_samples = df[df['reasoning'].str.contains('Error', na=False)].sample(min(5, error_count))
            for _, row in error_samples.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Error:** {row['reasoning']}")
                st.write("---")

def check_openai_api():
    """Test the OpenAI API connection and return True if working."""
    if not openai_api_key:
        st.error("OpenAI API key is required")
        return False
        
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Simple test message"}],
            max_tokens=5
        )
        st.success("✅ OpenAI API connection successful!")
        return True
    except Exception as e:
        st.error(f"❌ OpenAI API connection failed: {str(e)}")
        return False

# Main app function
def run_app():

    #Optional API usage tracking
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = 0

    # For keyboard shortcuts (if enabled)
    if st.session_state.keyboard_shortcuts_enabled:
        st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+F for search
            if (e.ctrlKey && e.key === 'f') {
                document.getElementById('search-box').focus();
                e.preventDefault();
            }
            // Ctrl+S to save
            if (e.ctrlKey && e.key === 's') {
                document.getElementById('download-button').click();
                e.preventDefault();
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Display keyboard shortcut help
        with st.expander("Keyboard Shortcuts"):
            st.markdown("""
            - **Ctrl+F**: Focus search box
            - **Ctrl+S**: Download results
            - **Space**: Toggle video selection (when focused)
            """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload YouTube Videos Excel File", type=["xlsx", "xls"], 
                                   help="Upload an Excel file containing YouTube video metadata")

    if uploaded_file is not None:
        # Check if API key is provided and test connection
        if not check_openai_api():
            return
        
        # Load the data
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded {len(df)} videos from the Excel file")
            
            # Fill NaN values to avoid type errors
            df = df.fillna("")
            
            # Display a sample of the data
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            # Check required columns
            required_columns = ['id', 'title', 'url']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"The Excel file is missing required columns: {', '.join(missing_columns)}")
                return
            
            # Start processing button
            if st.button("Start Screening Videos"):
                st.info("Starting the screening process. This may take a while depending on the number of videos.")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a spinner while processing
                with st.spinner('Processing videos...'):
                    # Process the videos
                    final_df = asyncio.run(process_videos(df, progress_bar))
                    
                    if final_df is not None:
                        # Store in session state for later use
                        st.session_state.final_df = final_df
                        
                        # Update progress and status
                        progress_bar.progress(1.0)
                        status_text.success("Processing complete!")
                        
                        # Send notification if enabled
                        if notification_enabled and st.session_state.notification_email:
                            notification_sent = send_notification_email(
                                "Video Screening Complete",
                                f"Your batch of {len(final_df)} videos has finished processing.\n\n"
                                f"Results summary:\n"
                                f"- INCLUDE: {len(final_df[final_df['decision'] == 'INCLUDE'])}\n"
                                f"- NOT SURE: {len(final_df[final_df['decision'] == 'NOT SURE'])}\n"
                                f"- EXCLUDE: {len(final_df[final_df['decision'] == 'EXCLUDE'])}\n",
                                st.session_state.notification_email
                            )
                            if notification_sent:
                                st.success(f"Notification email sent to {st.session_state.notification_email}")
                            else:
                                st.warning("Failed to send notification email. Check SMTP settings.")
                        
                        # Display results
                        st.header("Screening Results")
                        
                        # Add search functionality
                        st.subheader("Search Results")
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            search_term = st.text_input("Search in results", 
                                                      placeholder="Enter title, description, or cues", 
                                                      key="search-box")
                        with col2:
                            search_field = st.selectbox("Field", 
                                                      options=["All", "Title", "Description", "Cues", "Reasoning"])
                        with col3:
                            search_button = st.button("Search")
                            
                        if search_term and search_button:
                            # Map selection to DataFrame columns
                            field_mapping = {
                                "Title": "title",
                                "Description": "text",
                                "Cues": "cues_found",
                                "Reasoning": "reasoning"
                            }
                            
                            if search_field == "All":
                                # Search in all text fields
                                mask = (
                                    final_df["title"].str.contains(search_term, case=False, na=False) |
                                    final_df["text"].str.contains(search_term, case=False, na=False) |
                                    final_df["cues_found"].str.contains(search_term, case=False, na=False) |
                                    final_df["reasoning"].str.contains(search_term, case=False, na=False)
                                )
                            else:
                                # Search in specific field
                                column = field_mapping.get(search_field)
                                if column in final_df.columns:
                                    mask = final_df[column].str.contains(search_term, case=False, na=False)
                                else:
                                    st.warning(f"Column {column} not found in results.")
                                    mask = pd.Series(False, index=final_df.index)
                            
                            # Store search results in session state
                            st.session_state.search_results = final_df[mask]
                            if not st.session_state.search_results.empty:
                                st.success(f"Found {len(st.session_state.search_results)} matching videos")
                            else:
                                st.warning("No matches found")
                        
                        # Display search results if available
                        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
                            st.subheader("Search Results")
                            for _, video in st.session_state.search_results.iterrows():
                                display_video_example(video)
                                st.divider()
                            
                            # Clear search button
                            if st.button("Clear Search"):
                                st.session_state.search_results = None
                                st.experimental_rerun()
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Total Videos", len(final_df))
                        col2.metric("Include", len(final_df[final_df['decision'] == 'INCLUDE']))
                        col3.metric("Exclude", len(final_df[final_df['decision'] == 'EXCLUDE']))
                        col4.metric("Not Sure", len(final_df[final_df['decision'] == 'NOT SURE']))
                        
                        # Add custom categories to metrics if they exist
                        if st.session_state.custom_categories:
                            custom_cols = st.columns(min(4, len(st.session_state.custom_categories)))
                            for i, category in enumerate(st.session_state.custom_categories):
                                custom_cols[i % 4].metric(
                                    category, 
                                    len(final_df[final_df['decision'] == category])
                                )
                        
                        # Analysis tab
                        with st.expander("Decision Analysis"):
                            debug_llm_responses(final_df)
                        
                        # Sample videos for each decision category
                        st.subheader("Sample Results")
                        
                        # Create tabs for all categories including custom ones
                        all_categories = ["Include", "Exclude", "Not Sure"] + st.session_state.custom_categories
                        tabs = st.tabs(all_categories)
                        
                        # Standard categories
                        with tabs[0]:
                            include_samples = final_df[final_df['decision'] == 'INCLUDE'].sample(min(3, len(final_df[final_df['decision'] == 'INCLUDE'])))
                            if not include_samples.empty:
                                for _, video in include_samples.iterrows():
                                    display_video_example(video)
                                    st.divider()
                            else:
                                st.info("No videos were included.")
                        
                        with tabs[1]:
                            exclude_samples = final_df[final_df['decision'] == 'EXCLUDE'].sample(min(3, len(final_df[final_df['decision'] == 'EXCLUDE'])))
                            if not exclude_samples.empty:
                                for _, video in exclude_samples.iterrows():
                                    display_video_example(video)
                                    st.divider()
                            else:
                                st.info("No videos were excluded.")
                        
                        with tabs[2]:
                            not_sure_samples = final_df[final_df['decision'] == 'NOT SURE'].sample(min(3, len(final_df[final_df['decision'] == 'NOT SURE'])))
                            if not not_sure_samples.empty:
                                for _, video in not_sure_samples.iterrows():
                                    display_video_example(video)
                                    st.divider()
                            else:
                                st.info("No videos were marked as 'Not Sure'.")
                        
                        # Custom category tabs
                        for i, category in enumerate(st.session_state.custom_categories):
                            with tabs[i+3]:
                                category_samples = final_df[final_df['decision'] == category].sample(
                                    min(3, len(final_df[final_df['decision'] == category]))
                                ) if category in final_df['decision'].values else pd.DataFrame()
                                
                                if not category_samples.empty:
                                    for _, video in category_samples.iterrows():
                                        display_video_example(video)
                                        st.divider()
                                else:
                                    st.info(f"No videos in category '{category}'.")
                        
                        # Download link
                        st.subheader("Download Results")
                        st.markdown(generate_download_link(final_df, "screened_youtube_videos.xlsx"), 
                                   unsafe_allow_html=True)
                        
                        # Save transcript cache
                        with open(transcript_cache_file, 'w', encoding='utf-8') as f:
                            json.dump(transcript_cache, f, ensure_ascii=False)
                    else:
                        status_text.error("Processing failed or produced no results.")
        
        except Exception as e:
            st.error(f"Error processing the Excel file: {e}")
            logger.exception("Error in main app")
    
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload an Excel file containing YouTube video data to begin.")
        
        with st.expander("File Format Requirements"):
            st.markdown("""
            Your Excel file should contain the following columns:
            
            - `id`: YouTube video ID or unique identifier
            - `title`: Video title
            - `url`: Complete YouTube video URL
            - `text`: Video description
            - `channelName`: YouTube channel name
            - `duration`: Video length
            - `date`: Upload date (optional)
            - `viewCount`: Number of views (optional)
            
            Additional columns like hashtags, location, etc. will also be used if available.
            """)
        
        with st.expander("Example Excel Structure"):
            example_data = {
                'id': ['iotARqK_D_w', 'bM6AJjRshWc'],
                'title': ['Living in Nairobi-The Pros & Cons!', '[Adulting in Nairobi] - | my first vlog |'],
                'url': ['https://www.youtube.com/watch?v=iotARqK_D_w', 'https://www.youtube.com/watch?v=bM6AJjRshWc'],
                'text': ['In this episode, I dive into...', 'hi! I\'m no professional but...'],
                'channelName': ['Weyni Tesfai', 'kennedy films'],
                'duration': ['00:19:48', '00:08:40'],
                'date': ['2024-11-15T20:01:59.000Z', '2022-11-22T06:15:02.000Z'],
                'viewCount': [100974, 114]
            }
            st.dataframe(pd.DataFrame(example_data))

        # Show a sample of what the app does
        st.subheader("How it works")
        st.markdown("""
        This tool uses artificial intelligence to identify genuine "day in the life" or daily vlog videos 
        by analyzing video titles, descriptions, hashtags, and transcripts when available.
        
        The AI classifier categorizes videos as:
        - **INCLUDE**: Strong evidence this is a day-in-life video
        - **NOT SURE**: Some indicators but inconclusive
        - **EXCLUDE**: Clear evidence this is not a day-in-life video
        
        The tool also extracts specific cues and indicators that helped make the decision.
        """)
        
        # Add the bottom acknowledgement logo
        st.image("acknowledgement_logo.png", width=700)

# Run the app
if __name__ == "__main__":
    run_app()
