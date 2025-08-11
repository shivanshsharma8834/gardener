import asyncio
import logging
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
from textwrap import dedent
import httpx
import readabilipy
import json

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
logger.info("Loading environment variables...")
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
logger.info("Environment variables loaded successfully")

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Imgflip API Integration ---
class ImgflipAPI:
    def __init__(self):
        self.username = os.environ.get("IMGFLIP_USERNAME")
        self.password = os.environ.get("IMGFLIP_PASSWORD")
        self.base_url = "https://api.imgflip.com"
        self.recently_used_templates = []  # Track recently used templates
        
        if not self.username or not self.password:
            logger.warning("Imgflip credentials not found in environment variables")
    
    async def get_popular_memes(self) -> list[dict]:
        """Get list of popular meme templates from Imgflip."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/get_memes")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data.get("data", {}).get("memes", [])
                    else:
                        logger.error(f"Imgflip API error: {data.get('error_message', 'Unknown error')}")
                        return []
                else:
                    logger.error(f"Failed to fetch memes: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching popular memes: {e}")
            return []
    
    async def get_trending_memes(self) -> list[dict]:
        """Get list of trending meme templates from Imgflip with enhanced filtering."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/get_memes")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        all_memes = data.get("data", {}).get("memes", [])
                        
                        # Filter and prioritize trending memes
                        trending_memes = self._filter_trending_memes(all_memes)
                        logger.info(f"Found {len(trending_memes)} trending memes out of {len(all_memes)} total")
                        return trending_memes
                    else:
                        logger.error(f"Imgflip API error: {data.get('error_message', 'Unknown error')}")
                        return []
                else:
                    logger.error(f"Failed to fetch memes: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching trending memes: {e}")
            return []
    
    def _filter_trending_memes(self, all_memes: list[dict]) -> list[dict]:
        """Filter memes to prioritize trending and recent content."""
        import random
        from datetime import datetime, timedelta
        
        # Keywords that indicate trending/recent content
        trending_keywords = [
            'trending', 'viral', 'popular', 'new', 'latest', 'recent', 'hot',
            '2024', '2023', 'current', 'modern', 'contemporary', 'fresh',
            'tiktok', 'instagram', 'social', 'internet', 'digital', 'online'
        ]
        
        # Score each meme based on trending indicators
        scored_memes = []
        for meme in all_memes:
            score = 0
            name_lower = meme.get('name', '').lower()
            box_count = meme.get('box_count', 0)
            
            # Higher score for memes with trending keywords
            for keyword in trending_keywords:
                if keyword in name_lower:
                    score += 10
            
            # Prefer memes with 2 text boxes (most common format)
            if box_count == 2:
                score += 5
            elif box_count == 1:
                score += 3
            
            # Prefer memes with shorter names (often more recent)
            if len(name_lower) < 30:
                score += 2
            
            # Add some randomness to avoid always picking the same ones
            score += random.randint(0, 5)
            
            scored_memes.append((meme, score))
        
        # Sort by score (highest first) and take top trending ones
        scored_memes.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 50 trending memes, then shuffle for variety
        trending_memes = [meme for meme, score in scored_memes[:50]]
        random.shuffle(trending_memes)
        
        return trending_memes
    
    async def get_recent_memes(self, days_back: int = 30) -> list[dict]:
        """Get memes that are likely to be more recent based on naming patterns."""
        try:
            all_memes = await self.get_popular_memes()
            
            # Filter for memes that might be more recent
            recent_keywords = [
                '2024', '2023', 'new', 'latest', 'recent', 'fresh', 'modern',
                'tiktok', 'instagram', 'viral', 'trending', 'current'
            ]
            
            recent_memes = []
            for meme in all_memes:
                name_lower = meme.get('name', '').lower()
                if any(keyword in name_lower for keyword in recent_keywords):
                    recent_memes.append(meme)
            
            # If we don't have enough recent memes, add some popular ones
            if len(recent_memes) < 20:
                recent_memes.extend(all_memes[:30])
            
            logger.info(f"Found {len(recent_memes)} recent memes")
            return recent_memes
            
        except Exception as e:
            logger.error(f"Error fetching recent memes: {e}")
            return await self.get_popular_memes()
    
    async def get_memes_by_type(self, meme_type: str = "trending") -> list[dict]:
        """Get memes based on specified type (trending, recent, popular, all)."""
        try:
            if meme_type == "trending":
                return await self.get_trending_memes()
            elif meme_type == "recent":
                return await self.get_recent_memes()
            elif meme_type == "popular":
                return await self.get_popular_memes()
            else:
                # Default to trending
                return await self.get_trending_memes()
        except Exception as e:
            logger.error(f"Error fetching memes by type {meme_type}: {e}")
            return await self.get_popular_memes()
    
    async def create_meme(self, template_id: str, text0: str = "", text1: str = "") -> str | None:
        """Create a meme using Imgflip API and return the URL."""
        if not self.username or not self.password:
            logger.error("Imgflip credentials not available")
            return None
            
        try:
            data = {
                "template_id": template_id,
                "username": self.username,
                "password": self.password,
                "text0": text0,
                "text1": text1
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/caption_image", data=data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        return result.get("data", {}).get("url")
                    else:
                        logger.error(f"Imgflip creation error: {result.get('error_message', 'Unknown error')}")
                        return None
                else:
                    logger.error(f"Failed to create meme: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error creating meme: {e}")
            return None
    
    async def download_meme_image(self, meme_url: str) -> bytes | None:
        """Download the meme image from URL and return as bytes."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(meme_url, timeout=30)
                if response.status_code == 200:
                    return response.content
                else:
                    logger.error(f"Failed to download meme image: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading meme image: {e}")
            return None
    
    def get_diverse_templates(self, all_templates: list, count: int = 20) -> list:
        """Get a diverse selection of templates, avoiding recently used ones."""
        import random
        
        # Filter out recently used templates (last 5)
        available_templates = [t for t in all_templates if t['id'] not in self.recently_used_templates[-5:]]
        
        if len(available_templates) < count:
            # If we don't have enough diverse templates, reset and use all
            self.recently_used_templates = []
            available_templates = all_templates
        
        # Get a mix of trending and diverse templates
        trending_count = min(10, count // 2)  # 50% trending
        diverse_count = count - trending_count  # 50% diverse
        
        trending_templates = available_templates[:trending_count]
        diverse_templates = random.sample(available_templates[trending_count:], diverse_count)
        
        selection = trending_templates + diverse_templates
        random.shuffle(selection)  # Shuffle for variety
        
        return selection
    
    def track_template_usage(self, template_id: str):
        """Track that a template was used to avoid repetition."""
        self.recently_used_templates.append(template_id)
        # Keep only last 10 to avoid memory issues
        if len(self.recently_used_templates) > 10:
            self.recently_used_templates = self.recently_used_templates[-10:]

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text
            content_type = response.headers.get("content-type", "")

        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    logger.info("Tool called: validate()")
    logger.info(f"Returning MY_NUMBER: {MY_NUMBER}")
    return MY_NUMBER

# --- Tool: about ---
@mcp.tool
async def about() -> dict[str, str]:
    """Return server name and description for MemeChat MCP."""
    logger.info("Tool called: about()")
    server_name = "MemeChat MCP"
    server_description = dedent(
        """
        MemeChat MCP focuses on meme creation and image captioning.
        It provides tools to:
        - create memes using Imgflip templates with AI-generated captions,
        - turn screenshots into memes,
        - add captions to user images,
        - convert images to black and white.

        It uses Gemini for caption generation and supports Gen Z style with optional Hinglish.
        """
    ).strip()

    return {
        "name": server_name,
        "description": server_description,
    }

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)



# --- Tool: create_imgflip_meme (for using Imgflip templates) ---
ImgflipMemeDescription = RichToolDescription(
    description="Create memes using Imgflip templates with AI-generated captions. REQUIRED: Only use this tool when user wants to create a meme using popular meme templates (like Drake, Two Buttons, etc.).",
    use_when="ONLY use this tool when user explicitly asks to create a meme using meme templates or popular meme formats. Do NOT use for custom image captioning - use add_caption_to_image instead. CRITICAL: 'prompt' = meme type/concept (e.g., 'confused person'), 'meme_caption' = context/inspiration for AI to generate appropriate captions. NEVER use top_text or bottom_text parameters - they do NOT exist in this tool. Let AI generate captions that fit the template layout.",
    side_effects="Returns a meme image created using Imgflip API with AI-generated captions. Returns both text info and the meme image directly.",
)

# --- Tool: screenshot_meme_generator (for analyzing screenshots and creating memes) ---
ScreenshotMemeDescription = RichToolDescription(
    description="SPECIFICALLY for screenshots: Analyze user's screenshot content and create a meme using Imgflip templates. REQUIRED: User must provide a screenshot (puch_image_data parameter). This tool analyzes what's in the screenshot (apps, websites, error messages, social media, etc.) and creates a meme about that content.",
    use_when="ONLY use this tool when user provides a screenshot AND wants to create a meme ABOUT what's shown in the screenshot. This tool is specifically for screenshots of digital content (apps, websites, error messages, social media posts, etc.) that the user wants turned into a meme. Do NOT use for regular images or photos - use add_caption_to_image for those. Do NOT use for general meme creation without screenshots - use create_imgflip_meme for that.",
    side_effects="Returns a meme image created using Imgflip API with captions generated based on screenshot content analysis. Returns both text info and the meme image directly.",
)



@mcp.tool(description=ImgflipMemeDescription.model_dump_json())
async def create_imgflip_meme(
    prompt: Annotated[str, Field(description="REQUIRED: Description of what type of meme to create (e.g., 'confused person', 'surprised reaction', 'Drake format', 'hinglish meme banado'). This describes the meme concept, NOT the caption text. This is the ONLY required parameter. DO NOT use top_text or bottom_text - they do not exist in this tool.")],
    user_context: Annotated[str | None, Field(description="OPTIONAL: Additional context about the user, situation, or target audience (e.g., 'for my tech team', 'about remote work', 'for Gen Z audience', 'hinglish audience ke liye')")] = None,
    auto_pick_template: Annotated[bool, Field(description="OPTIONAL: Let AI automatically pick the best template (default: True)")] = True,
    template_id: Annotated[str | None, Field(description="OPTIONAL: Specific Imgflip template ID to use (only if auto_pick_template is False)")] = None,
    meme_caption: Annotated[str | None, Field(description="OPTIONAL: Context or inspiration for the meme caption (e.g., 'When you finally understand the joke', 'jab tumhe samajh aa jata hai'). This will be used as input for AI to generate appropriate captions that fit the template layout. Do NOT pass this as direct text placement. DO NOT use top_text or bottom_text - they do not exist in this tool.")] = None,
    gen_z_style: Annotated[bool, Field(description="OPTIONAL: Generate captions in Gen Z style (default: True)")] = True,
    hinglish_support: Annotated[bool, Field(description="OPTIONAL: Enable Hinglish (Hindi in English script) support for captions (default: False)")] = False,
) -> list[TextContent | ImageContent]:
    """
    Create memes using Imgflip templates with AI-generated captions.
    
    NOTE: This tool does NOT accept top_text or bottom_text parameters.
    All captions are generated automatically from the user's prompt and meme_caption context.
    """
    logger.info("Tool called: create_imgflip_meme()")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"User context: {user_context}")
    logger.info(f"Auto pick template: {auto_pick_template}")
    logger.info(f"Template ID: {template_id}")
    logger.info(f"User meme caption: {meme_caption}")
    logger.info(f"Gen Z style: {gen_z_style}")
    logger.info(f"Hinglish support: {hinglish_support}")
    
    try:
        # Setup Imgflip API
        imgflip = ImgflipAPI()
        
        # Setup Gemini for caption generation
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="GEMINI_API_KEY not found in environment variables"))
        
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_api_key)
        
        # Step 1: Get available templates
        logger.info("Fetching trending meme templates...")
        templates = await imgflip.get_trending_memes()
        if not templates:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to fetch meme templates from Imgflip"))
        
        logger.info(f"Found {len(templates)} templates")
        
        # Step 2: Select template with variety
        selected_template = None
        if auto_pick_template:
            # Use AI to pick the best template with variety
            logger.info("Using AI to pick the best template with variety...")
            
            # Get diverse templates, avoiding recently used ones
            selection_templates = imgflip.get_diverse_templates(templates, count=20)
            template_list = "\n".join([f"- {t['name']} (ID: {t['id']}): {t['url']}" for t in selection_templates])
            
            context_info = f"\nUser Context: {user_context}" if user_context else ""
            
            template_selection_prompt = f"""
            Based on this meme prompt: "{prompt}"{context_info}
            
            Available diverse meme templates (mix of popular and unique):
            {template_list}
            
            CRITICAL: The user wants to create a meme about "{prompt}". Select a template that would work BEST for making a meme about "{prompt}".
            
            IMPORTANT: Select a template that is:
            1. RELEVANT to "{prompt}" - the template should work well for creating captions about "{prompt}"
            2. APPROPRIATE for the intended humor style about "{prompt}"
            3. VARIED from common overused templates (avoid always picking Drake, Two Buttons, etc.)
            4. SUITABLE for the target audience and situation
            5. WELL-FORMATTED for creating captions about "{prompt}"
            
            Selection criteria:
            - Choose templates that would make "{prompt}" funny and relatable
            - Consider how "{prompt}" could be integrated into the template's visual elements
            - Pick templates that allow for good caption placement about "{prompt}"
            - Avoid overused templates unless they're perfect for "{prompt}"
            - Consider template popularity vs uniqueness balance
            - Think about which template would make "{prompt}" the most hilarious
            
            Return ONLY the template ID number, nothing else.
            """
            
            template_response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[types.Content(parts=[types.Part(text=template_selection_prompt)])]
            )
            
            selected_id = template_response.text.strip()
            logger.info(f"AI selected template ID: {selected_id}")
            
            # Find the selected template
            for template in templates:
                if str(template['id']) == selected_id:
                    selected_template = template
                    break
            
            if not selected_template:
                logger.warning("AI-selected template not found, using first available")
                selected_template = templates[0]
        else:
            # Use provided template ID
            if template_id:
                for template in templates:
                    if str(template['id']) == template_id:
                        selected_template = template
                        break
                
                if not selected_template:
                    raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Template ID {template_id} not found"))
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="Template ID required when auto_pick_template is False"))
        
        logger.info(f"Selected template: {selected_template['name']} (ID: {selected_template['id']})")
        
        # Track template usage to avoid repetition
        imgflip.track_template_usage(str(selected_template['id']))
        
        # Step 3: Generate captions from user's message and template analysis
        logger.info("Generating captions from user's message and template analysis...")
        
        # Download and analyze template image
        template_image_bytes = await imgflip.download_meme_image(selected_template['url'])
        
        if template_image_bytes:
            logger.info("Template image downloaded, analyzing with Gemini...")
            
            # Create content for template analysis
            template_image_part = types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=template_image_bytes
                )
            )
            
            # Determine user's caption context from the prompt
            user_caption_context = meme_caption if meme_caption else prompt
            
            template_analysis_prompt = f"""
            Analyze this meme template image and provide detailed context for caption generation.
            
            Template: {selected_template['name']}
            User's meme concept: "{prompt}"
            User's caption context: "{user_caption_context}"
            
            CRITICAL: The user specifically wants a meme about "{prompt}". Your analysis should focus on how to create captions that are ABOUT "{prompt}" while working with this template's visual elements.
            
            Provide a detailed analysis including:
            
            1. VISUAL ELEMENTS: Describe exactly what you see in the template:
               - Characters: Who/what are the main characters? What are they doing?
               - Actions: What actions or movements are happening?
               - Expressions: What emotions or expressions are shown?
               - Objects: What objects, props, or background elements are present?
               - Layout: How is the image structured (top/bottom, left/right, etc.)?
            
            2. TEXT AREAS: Analyze the template's text placement areas:
               - How many text boxes does this template have?
               - Where are the text areas located (top, bottom, left, right, center)?
               - What's the purpose of each text area (setup, punchline, label, etc.)?
               - What's the typical format for this template (setup/punchline, single caption, etc.)?
            
            3. EMOTIONAL CONTEXT: What's the mood and tone?
               - What emotions are being conveyed?
               - Is it funny, serious, ironic, dramatic, etc.?
               - What's the overall vibe of the template?
            
            4. CAPTION OPPORTUNITIES: How can captions about "{prompt}" enhance this template?
               - What would make this template funny with text about "{prompt}"?
               - Where should text be placed for maximum impact?
               - What type of humor about "{prompt}" would work best with this visual?
               - How can we make "{prompt}" the focus while using this template's visual elements?
            
            5. USER INTEGRATION: How can "{prompt}" be integrated into this template?
               - How does "{prompt}" relate to what's happening in the template?
               - What captions about "{prompt}" would work perfectly with this template's visual?
               - How can we adapt "{prompt}" to work with this template's layout and style?
               - What specific captions about "{prompt}" would be hilarious with this template?
            
            Return a comprehensive analysis that includes specific information about text areas and placement, with emphasis on how to make "{prompt}" the central focus of the meme.
            NO EMOJIS.
            """
            
            template_analysis_content = types.Content(parts=[
                types.Part(text=template_analysis_prompt),
                template_image_part
            ])
            
            try:
                template_analysis_response = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=[template_analysis_content]
                )
                template_analysis = template_analysis_response.text.strip()
                logger.info(f"Template analysis: {template_analysis}")
                logger.info("Template analysis completed successfully")
            except Exception as e:
                logger.error(f"Template analysis failed: {e}")
                template_analysis = f"Template: {selected_template['name']} - User caption context: {user_caption_context}"
        else:
            logger.warning("Could not download template image, using basic analysis")
            template_analysis = f"Template: {selected_template['name']} - User caption context: {user_caption_context}"
        
        # Generate captions using the template analysis and user's message
        context_info = f"\nUser Context: {user_context}" if user_context else ""
        
        caption_prompt = f"""
        Create a HILARIOUS meme caption for this specific template.
        
        TEMPLATE ANALYSIS:
        {template_analysis}
        
        USER'S REQUEST:
        - Meme concept: "{prompt}"
        - User's caption context: "{user_caption_context}"
        {context_info}
        
        CRITICAL REQUIREMENTS:
        1. The user specifically requested a meme about "{prompt}" - this MUST be the primary focus
        2. Your captions MUST directly reference "{prompt}" in a funny, relevant way
        3. Do NOT create generic captions that ignore the user's request
        4. The meme should be ABOUT "{prompt}" - not just use the template's default text
        """
        
        # Add Hinglish support if enabled
        if hinglish_support:
            caption_prompt += f"""
        
        HINGLISH LANGUAGE SUPPORT ENABLED:
        - Generate captions in Hinglish (Hindi written in English script)
        - Use natural Hinglish expressions and humor
        - Incorporate Hinglish slang and cultural references
        - Make it relatable to Hinglish-speaking audience
        - Use appropriate Hinglish grammar and sentence structure
        - Consider Hinglish internet culture and social media trends
        - Use Hinglish meme culture and popular expressions
        
        EXAMPLES FOR HINGLISH:
        - If user says "modi" and template shows a dog in fire: "MODI KA KARYAKAL" / "Sab theek hai"
        - If user says "exam" and template shows Drake: "Exam ki tayari" / "Exam se bachna"
        - If user says "office" and template shows surprised Pikachu: "Monday ka din" / "Pikachu ka chehra"
        """
        else:
            caption_prompt += f"""
        
        EXAMPLES:
        - If user says "trump" and template shows a dog in fire: "TRUMP'S CAMPAIGN" / "This is fine"
        - If user says "trump" and template shows Drake: "Voting for Trump" / "Voting for anyone else"
        - If user says "trump" and template shows surprised Pikachu: "Trump wins again" / "Surprised Pikachu face"
        """
        
        caption_prompt += f"""
        
        IMPORTANT: Use the template analysis above to understand what's happening in the template image. Your captions should:
        - DIRECTLY reference "{prompt}" as the main subject
        - Work with the visual elements described in the template analysis
        - Use the specific characters, actions, and expressions described in the template
        - Match the emotional tone and mood identified in the template analysis
        - Work with the template's specific format and layout as described
        - Incorporate the user's caption context "{user_caption_context}" naturally into the template's context
        - Be funny and relatable to the template's visual context
        - Work with the template's specific text placement areas
        - Be short and punchy (max 30 characters per line)
        - Have perfect timing and delivery for this specific template
        - Be appropriate for the target audience and context
        
        CRITICAL: Your captions MUST be about "{prompt}" and work with the template's visual elements. Do not create generic captions - make them specific to "{prompt}" and what's happening in this template image.
        DO NOT USE ANY EMOJIS. NO EMOJIS.
        
        Based on the template analysis, determine the appropriate caption format and return:
        
        For templates with two text areas (top/bottom):
        TOP: [text for top area about "{prompt}"]
        BOTTOM: [text for bottom area about "{prompt}"]
        
        For templates with single text area:
        TOP: [single caption about "{prompt}"]
        
        For templates with left/right text areas:
        LEFT: [text for left area about "{prompt}"]
        RIGHT: [text for right area about "{prompt}"]
        
        For templates with center text:
        CENTER: [single caption about "{prompt}"]
        
        Choose the format that best matches the template's text areas as described in the analysis.
        """
        
        if gen_z_style:
            if hinglish_support:
                caption_prompt += """
                
                Make it Gen Z style with Hinglish:
                - Use Gen Z humor (sarcastic, ironic, self-deprecating, chaotic) in Hinglish
                - Include current Hinglish Gen Z references and slang (yaar, bro, bhai, etc.)
                - Reference Hinglish internet culture and social media trends
                - Be relatable to Hinglish-speaking Gen Z lifestyle and experiences
                - Use current Hinglish TikTok trends and viral content
                - Consider the user context for additional relevance
                - Match the template's visual style with Hinglish Gen Z humor
                - Use popular Hinglish meme expressions and internet slang
                """
            else:
                caption_prompt += """
                
                Make it Gen Z style:
                - Use Gen Z humor (sarcastic, ironic, self-deprecating, chaotic)
                - Include current Gen Z references and slang
                - Reference internet culture and social media
                - Be relatable to Gen Z lifestyle and experiences
                - Use current TikTok trends and viral content
                - Consider the user context for additional relevance
                - Match the template's visual style with Gen Z humor
                """
        
        caption_response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[types.Content(parts=[types.Part(text=caption_prompt)])]
        )
        
        caption_text = caption_response.text.strip()
        logger.info(f"Generated captions: {caption_text}")
        logger.info("Caption generation completed with template context")
        
        # Parse the captions based on template format
        top_text = ""
        bottom_text = ""
        
        logger.info(f"Parsing caption response: {caption_text}")
        
        for line in caption_text.split('\n'):
            line = line.strip()
            if line.startswith('TOP:'):
                top_text = line.replace('TOP:', '').strip()
                logger.info(f"Found top text: '{top_text}'")
            elif line.startswith('BOTTOM:'):
                bottom_text = line.replace('BOTTOM:', '').strip()
                logger.info(f"Found bottom text: '{bottom_text}'")
            elif line.startswith('LEFT:'):
                # For left/right templates, map to top/bottom for Imgflip API
                top_text = line.replace('LEFT:', '').strip()
                logger.info(f"Found left text (mapped to top): '{top_text}'")
            elif line.startswith('RIGHT:'):
                bottom_text = line.replace('RIGHT:', '').strip()
                logger.info(f"Found right text (mapped to bottom): '{bottom_text}'")
            elif line.startswith('CENTER:'):
                # For center-only templates, use as top text
                top_text = line.replace('CENTER:', '').strip()
                logger.info(f"Found center text (mapped to top): '{top_text}'")
        
        # If no specific format found, try to parse as single caption
        if not top_text and not bottom_text:
            lines = caption_text.strip().split('\n')
            if len(lines) == 1:
                top_text = lines[0].strip()
                logger.info(f"Using single line as top text: '{top_text}'")
            elif len(lines) == 2:
                top_text = lines[0].strip()
                bottom_text = lines[1].strip()
                logger.info(f"Using two lines as top/bottom: '{top_text}' / '{bottom_text}'")
        
        logger.info(f"Final parsed captions - Top: '{top_text}', Bottom: '{bottom_text}'")
        
        # Step 4: Create the meme
        logger.info("Creating meme with Imgflip API...")
        meme_url = await imgflip.create_meme(
            template_id=str(selected_template['id']),
            text0=top_text or "",
            text1=bottom_text or ""
        )
        
        if meme_url:
            logger.info("Meme created successfully, downloading image...")
            
            # Download the meme image
            image_bytes = await imgflip.download_meme_image(meme_url)
            if image_bytes:
                # Convert to base64
                import base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                logger.info("Meme image downloaded and converted to base64")
                
                # Return both text info and the image
                text_info = f"Meme Created Successfully!\n\n**Template**: {selected_template['name']}\n**Top Text**: {top_text}\n**Bottom Text**: {bottom_text}"
                
                return [
                    TextContent(type="text", text=text_info),
                    ImageContent(type="image", mimeType="image/png", data=image_base64)
                ]
            else:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to download meme image"))
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to create meme with Imgflip API"))
        
    except Exception as e:
        logger.error(f"Error in create_imgflip_meme: {str(e)}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to create Imgflip meme: {str(e)}"))

@mcp.tool(description=ScreenshotMemeDescription.model_dump_json())
async def screenshot_meme_generator(
    puch_image_data: Annotated[str, Field(description="REQUIRED: Base64-encoded screenshot image data to analyze and create meme from. This is the ONLY required parameter.")],
    user_context: Annotated[str | None, Field(description="OPTIONAL: Additional context about the user, situation, or target audience (e.g., 'for my tech team', 'about remote work', 'for Gen Z audience')")] = None,
    auto_pick_template: Annotated[bool, Field(description="OPTIONAL: Let AI automatically pick the best template (default: True)")] = True,
    template_id: Annotated[str | None, Field(description="OPTIONAL: Specific Imgflip template ID to use (only if auto_pick_template is False)")] = None,
    gen_z_style: Annotated[bool, Field(description="OPTIONAL: Generate captions in Gen Z style (default: True)")] = True,
    hinglish_support: Annotated[bool, Field(description="OPTIONAL: Enable Hinglish (Hindi in English script) support for captions (default: False)")] = False,
) -> list[TextContent | ImageContent]:
    """
    Analyze user's screenshot and create a meme using Imgflip templates with AI-generated captions.
    
    This tool first analyzes the screenshot content using Gemini, then selects an appropriate
    Imgflip template and generates captions that relate to what's in the screenshot.
    """
    logger.info("Tool called: screenshot_meme_generator()")
    logger.info(f"User context: {user_context}")
    logger.info(f"Auto pick template: {auto_pick_template}")
    logger.info(f"Template ID: {template_id}")
    logger.info(f"Gen Z style: {gen_z_style}")
    logger.info(f"Hinglish support: {hinglish_support}")
    
    try:
        # Setup Imgflip API
        imgflip = ImgflipAPI()
        
        # Setup Gemini for screenshot analysis and caption generation
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="GEMINI_API_KEY not found in environment variables"))
        
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_api_key)
        
        # Step 1: Analyze the screenshot using Gemini (optimized for speed)
        logger.info("Analyzing screenshot content...")
        import base64
        import io
        
        # Decode the screenshot
        screenshot_bytes = base64.b64decode(puch_image_data)
        logger.info(f"Screenshot size: {len(screenshot_bytes)} bytes")
        
        # Resize screenshot for faster Gemini processing (much smaller size for speed)
        from PIL import Image
        screenshot_image = Image.open(io.BytesIO(screenshot_bytes))
        max_size = 256  # Much smaller for faster processing
        img_width, img_height = screenshot_image.size
        
        if img_width > max_size or img_height > max_size:
            if img_width > img_height:
                new_width = max_size
                new_height = int((img_height * max_size) / img_width)
            else:
                new_height = max_size
                new_width = int((img_width * max_size) / img_height)
            
            logger.info(f"Resizing screenshot from {img_width}x{img_height} to {new_width}x{new_height}")
            screenshot_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save resized screenshot for Gemini
        temp_img = io.BytesIO()
        screenshot_image.save(temp_img, format="PNG", optimize=True)
        temp_img.seek(0)
        
        # Create optimized screenshot analysis prompt (shorter and faster)
        context_info = f"\nUser Context: {user_context}" if user_context else ""
        
        screenshot_analysis_prompt = f"""
        Analyze this screenshot for meme creation:
        1. What type of content? (app/website/error/social media/notification)
        2. What's the main subject or situation?
        3. What makes this relatable or funny?
        
        Return 2-3 focused sentences. NO EMOJIS.
        """
        
        # Create content for screenshot analysis
        screenshot_image_part = types.Part(
            inline_data=types.Blob(
                mime_type="image/png",
                data=temp_img.getvalue()
            )
        )
        analysis_text_part = types.Part(text=screenshot_analysis_prompt)
        analysis_content = types.Content(parts=[analysis_text_part, screenshot_image_part])
        
        try:
            logger.info("Analyzing screenshot with Gemini...")
            # Add timeout for faster processing
            import asyncio
            
            def analyze_screenshot_sync():
                try:
                    response = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[analysis_content]
                    )
                    return response.text.strip()
                except Exception as e:
                    logger.error(f"Gemini API error in analysis: {e}")
                    return None
            
            # Run with 15-second timeout for faster processing
            screenshot_analysis = await asyncio.wait_for(
                asyncio.to_thread(analyze_screenshot_sync), 
                timeout=15.0
            )
            
            if screenshot_analysis:
                logger.info(f"Screenshot analysis: {screenshot_analysis}")
                logger.info("Screenshot analysis completed successfully")
            else:
                screenshot_analysis = "Screenshot contains digital content that could be turned into a meme."
                
        except asyncio.TimeoutError:
            logger.warning("Screenshot analysis timeout, using fallback")
            screenshot_analysis = "Screenshot contains digital content that could be turned into a meme."
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            screenshot_analysis = "Screenshot contains digital content that could be turned into a meme."
        
        # Step 2: Get available templates
        logger.info("Fetching trending meme templates...")
        templates = await imgflip.get_trending_memes()
        if not templates:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to fetch meme templates from Imgflip"))
        
        logger.info(f"Found {len(templates)} templates")
        
        # Step 3: Select template based on screenshot analysis
        selected_template = None
        if auto_pick_template:
            # Use AI to pick the best template based on screenshot analysis
            logger.info("Using AI to pick the best template based on screenshot analysis...")
            
            # Get diverse templates, avoiding recently used ones
            selection_templates = imgflip.get_diverse_templates(templates, count=20)
            template_list = "\n".join([f"- {t['name']} (ID: {t['id']}): {t['url']}" for t in selection_templates])
            
            template_selection_prompt = f"""
            Screenshot content: {screenshot_analysis}
            Available templates: {template_list}
            
            Select the template that would work BEST for creating a meme about this screenshot content.
            Consider: visual style, text placement, humor potential.
            
            Return ONLY the template ID number.
            """
            
            template_response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[types.Content(parts=[types.Part(text=template_selection_prompt)])]
            )
            
            selected_id = template_response.text.strip()
            logger.info(f"AI selected template ID: {selected_id}")
            
            # Find the selected template
            for template in templates:
                if str(template['id']) == selected_id:
                    selected_template = template
                    break
            
            if not selected_template:
                logger.warning("AI-selected template not found, using first available")
                selected_template = templates[0]
        else:
            # Use provided template ID
            if template_id:
                for template in templates:
                    if str(template['id']) == template_id:
                        selected_template = template
                        break
                
                if not selected_template:
                    raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Template ID {template_id} not found"))
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="Template ID required when auto_pick_template is False"))
        
        logger.info(f"Selected template: {selected_template['name']} (ID: {selected_template['id']})")
        
        # Track template usage to avoid repetition
        imgflip.track_template_usage(str(selected_template['id']))
        
        # Step 4: Generate captions based on screenshot analysis and template
        logger.info("Generating captions based on screenshot analysis and template...")
        
        # Download and analyze template image
        template_image_bytes = await imgflip.download_meme_image(selected_template['url'])
        
        if template_image_bytes:
            logger.info("Template image downloaded, analyzing with Gemini...")
            
            # Create content for template analysis
            template_image_part = types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=template_image_bytes
                )
            )
            
            template_analysis_prompt = f"""
            Template: {selected_template['name']}
            Screenshot content: {screenshot_analysis}
            
            Analyze this template for meme creation:
            1. What visual elements are present? (characters, actions, expressions)
            2. How many text areas and where are they located?
            3. How can we create funny captions about the screenshot using this template?
            
            Return 2-3 focused sentences. NO EMOJIS.
            """
            
            template_analysis_content = types.Content(parts=[
                types.Part(text=template_analysis_prompt),
                template_image_part
            ])
            
            try:
                logger.info("Analyzing template with Gemini...")
                
                def analyze_template_sync():
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.0-flash-exp',
                            contents=[template_analysis_content]
                        )
                        return response.text.strip()
                    except Exception as e:
                        logger.error(f"Gemini API error in template analysis: {e}")
                        return None
                
                # Run with 15-second timeout for faster processing
                template_analysis = await asyncio.wait_for(
                    asyncio.to_thread(analyze_template_sync), 
                    timeout=15.0
                )
                
                if template_analysis:
                    logger.info(f"Template analysis: {template_analysis}")
                    logger.info("Template analysis completed successfully")
                else:
                    template_analysis = f"Template: {selected_template['name']} - Screenshot content: {screenshot_analysis}"
                    
            except asyncio.TimeoutError:
                logger.warning("Template analysis timeout, using fallback")
                template_analysis = f"Template: {selected_template['name']} - Screenshot content: {screenshot_analysis}"
            except Exception as e:
                logger.error(f"Template analysis failed: {e}")
                template_analysis = f"Template: {selected_template['name']} - Screenshot content: {screenshot_analysis}"
        else:
            logger.warning("Could not download template image, using basic analysis")
            template_analysis = f"Template: {selected_template['name']} - Screenshot content: {screenshot_analysis}"
        
        # Generate captions using the template analysis and screenshot content
        context_info = f"\nUser Context: {user_context}" if user_context else ""
        
        caption_prompt = f"""
        Template analysis: {template_analysis}
        Screenshot content: {screenshot_analysis}
        
        Create HILARIOUS captions about the screenshot using this template.
        Make it funny, relatable, and specific to the screenshot content.
        """
        
        # Add language support if enabled
        if hinglish_support:
            caption_prompt += "\nUse Hinglish (Hindi in English script)."
        
        caption_prompt += f"""
        
        Return format:
        TOP: [funny setup about screenshot content]
        BOTTOM: [hilarious punchline about screenshot content]
        
        Make it witty and relatable. Keep captions short (max 25 chars each). NO EMOJIS.
        """
        
        if gen_z_style:
            if hinglish_support:
                caption_prompt += "\nUse Gen Z Hinglish humor: sarcastic, relatable, current trends."
            else:
                caption_prompt += "\nUse Gen Z humor: sarcastic, ironic, relatable to internet culture."
        
        logger.info("Generating captions with Gemini...")
        
        def generate_caption_sync():
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=[types.Content(parts=[types.Part(text=caption_prompt)])]
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error in caption generation: {e}")
                return None
        
        try:
            # Run with 15-second timeout for faster processing
            caption_text = await asyncio.wait_for(
                asyncio.to_thread(generate_caption_sync), 
                timeout=15.0
            )
            
            if caption_text:
                logger.info(f"Generated captions: {caption_text}")
                logger.info("Caption generation completed with screenshot context")
            else:
                logger.warning("Caption generation failed, using fallback")
                caption_text = "TOP: Screenshot content\nBOTTOM: Made into meme"
                
        except asyncio.TimeoutError:
            logger.warning("Caption generation timeout, using fallback")
            caption_text = "TOP: Screenshot content\nBOTTOM: Made into meme"
        
        # Parse the captions based on template format
        top_text = ""
        bottom_text = ""
        
        logger.info(f"Parsing caption response: {caption_text}")
        
        for line in caption_text.split('\n'):
            line = line.strip()
            if line.startswith('TOP:'):
                top_text = line.replace('TOP:', '').strip()
                logger.info(f"Found top text: '{top_text}'")
            elif line.startswith('BOTTOM:'):
                bottom_text = line.replace('BOTTOM:', '').strip()
                logger.info(f"Found bottom text: '{bottom_text}'")
            elif line.startswith('LEFT:'):
                # For left/right templates, map to top/bottom for Imgflip API
                top_text = line.replace('LEFT:', '').strip()
                logger.info(f"Found left text (mapped to top): '{top_text}'")
            elif line.startswith('RIGHT:'):
                bottom_text = line.replace('RIGHT:', '').strip()
                logger.info(f"Found right text (mapped to bottom): '{bottom_text}'")
            elif line.startswith('CENTER:'):
                # For center-only templates, use as top text
                top_text = line.replace('CENTER:', '').strip()
                logger.info(f"Found center text (mapped to top): '{top_text}'")
        
        # If no specific format found, try to parse as single caption
        if not top_text and not bottom_text:
            lines = caption_text.strip().split('\n')
            if len(lines) == 1:
                top_text = lines[0].strip()
                logger.info(f"Using single line as top text: '{top_text}'")
            elif len(lines) == 2:
                top_text = lines[0].strip()
                bottom_text = lines[1].strip()
                logger.info(f"Using two lines as top/bottom: '{top_text}' / '{bottom_text}'")
        
        logger.info(f"Final parsed captions - Top: '{top_text}', Bottom: '{bottom_text}'")
        
        # Step 5: Create the meme
        logger.info("Creating meme with Imgflip API...")
        meme_url = await imgflip.create_meme(
            template_id=str(selected_template['id']),
            text0=top_text or "",
            text1=bottom_text or ""
        )
        
        if meme_url:
            logger.info("Meme created successfully, downloading image...")
            
            # Download the meme image
            image_bytes = await imgflip.download_meme_image(meme_url)
            if image_bytes:
                # Convert to base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                logger.info("Meme image downloaded and converted to base64")
                
                # Return both text info and the image (no emojis to prevent charmap issues)
                text_info = f"Screenshot Meme Created Successfully!\n\nTemplate: {selected_template['name']}\nScreenshot Analysis: {screenshot_analysis[:100]}...\nTop Text: {top_text}\nBottom Text: {bottom_text}"
                
                return [
                    TextContent(type="text", text=text_info),
                    ImageContent(type="image", mimeType="image/png", data=image_base64)
                ]
            else:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to download meme image"))
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to create meme with Imgflip API"))
        
    except Exception as e:
        logger.error(f"Error in screenshot_meme_generator: {str(e)}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to create screenshot meme: {str(e)}"))

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    logger.info("Tool called: job_finder()")
    logger.info(f"User goal: {user_goal}")
    logger.info(f"Job description provided: {job_description is not None}")
    logger.info(f"Job URL provided: {job_url}")
    logger.info(f"Raw mode: {raw}")
    if job_description:
        logger.info("Processing job description analysis...")
        result = (
            f"Job Description Analysis\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )
        logger.info("Job description analysis completed")
        return result

    if job_url:
        logger.info(f"Fetching job posting from URL: {job_url}")
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        logger.info("Job posting fetched successfully")
        return (
            f"Fetched Job Posting from URL: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        logger.info(f"Searching for jobs with query: {user_goal}")
        links = await Fetch.google_search_links(user_goal)
        logger.info(f"Found {len(links)} job links")
        return (
            f"Search Results for: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    logger.error("Invalid parameters provided to job_finder")
    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")],
) -> list[TextContent | ImageContent]:
    logger.info("Tool called: make_img_black_and_white()")
    logger.info(f"Image data length: {len(puch_image_data) if puch_image_data else 0} characters")
    
    import base64
    import io

    from PIL import Image

    try:
        logger.info("Decoding base64 image data...")
        image_bytes = base64.b64decode(puch_image_data)
        logger.info(f"Decoded image size: {len(image_bytes)} bytes")
        
        logger.info("Opening image with PIL...")
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Image dimensions: {image.size}")

        logger.info("Converting image to black and white...")
        bw_image = image.convert("L")
        logger.info("Black and white conversion completed")

        logger.info("Saving converted image...")
        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")
        logger.info(f"Final image size: {len(bw_base64)} characters")

        logger.info("Black and white image processing completed successfully")
        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        logger.error(f"Error in make_img_black_and_white: {str(e)}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

ADD_CAPTION_TO_IMAGE_DESCRIPTION = RichToolDescription(
    description="Add captions to user-provided images. REQUIRED: User must provide an image (puch_image_data parameter). Use this for custom image captioning.",
    use_when="ONLY use this tool when user provides an image and wants to add captions to it. Do NOT use for creating memes from templates - use create_imgflip_meme instead.",
    side_effects="Returns a new image with captions overlaid. Text is automatically centered and styled. Accessibility mode provides auto-scaling fonts and high contrast.",
)

@mcp.tool(description=ADD_CAPTION_TO_IMAGE_DESCRIPTION.model_dump_json())
async def add_caption_to_image(
    puch_image_data: Annotated[str, Field(description="REQUIRED: Base64-encoded image data to add captions to. This is the ONLY required parameter.")],
    top_caption: Annotated[str | None, Field(description="OPTIONAL: Top meme text (e.g., 'When you finally fix that bug', 'jab tumhe bug mil jata hai'). If not provided, will auto-generate using Gemini.")] = None,
    bottom_caption: Annotated[str | None, Field(description="OPTIONAL: Bottom meme text (e.g., 'But then you find 10 more', 'phir tumhe 10 aur milte hain'). If not provided, will auto-generate using Gemini.")] = None,
    font_size: Annotated[int, Field(description="OPTIONAL: Text size - use 30-50 for most memes, 60+ for impact text (default: 40)")] = 40,
    text_color: Annotated[str, Field(description="OPTIONAL: Text color - 'white' for most backgrounds, 'black' for light backgrounds, 'high_contrast' for accessibility (default: 'white')")] = "white",
    stroke_color: Annotated[str, Field(description="OPTIONAL: Text outline color - 'black' for white text, 'white' for dark text, 'auto' for accessibility (default: 'black')")] = "black",
    stroke_width: Annotated[int, Field(description="OPTIONAL: Text outline thickness - 2-4 for normal, 5+ for bold impact, 6+ for accessibility (default: 3)")] = 3,
    accessibility_mode: Annotated[bool, Field(description="OPTIONAL: Enable accessibility features: auto-scaling fonts, high contrast, and better readability (default: False)")] = False,
    hinglish_support: Annotated[bool, Field(description="OPTIONAL: Enable Hinglish (Hindi in English script) support for captions (default: False)")] = False,
) -> list[TextContent | ImageContent]:
    logger.info("Tool called: add_caption_to_image()")
    logger.info(f"Image data length: {len(puch_image_data) if puch_image_data else 0} characters")
    logger.info(f"Top caption: {top_caption}")
    logger.info(f"Bottom caption: {bottom_caption}")
    logger.info(f"Font size: {font_size}")
    logger.info(f"Text color: {text_color}")
    logger.info(f"Stroke color: {stroke_color}")
    logger.info(f"Stroke width: {stroke_width}")
    logger.info(f"Hinglish support: {hinglish_support}")
    import base64
    import io
    import math
    import os

    from PIL import Image, ImageDraw, ImageFont
    from google import genai
    from google.genai import types

    try:
        logger.info("Setting up Gemini API...")
        # Setup Gemini
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="GEMINI_API_KEY not found in environment variables"))
        
        logger.info("Gemini API key found")
        # Create client using new Google GenAI SDK
        client = genai.Client(api_key=gemini_api_key)
        logger.info("Gemini client created successfully")
        
        logger.info("Decoding base64 image data...")
        # Decode the image
        image_bytes = base64.b64decode(puch_image_data)
        logger.info(f"Decoded image size: {len(image_bytes)} bytes")
        
        logger.info("Opening image with PIL...")
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Image dimensions: {image.size}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB...")
            image = image.convert('RGB')
            logger.info("Image converted to RGB")
        else:
            logger.info("Image already in RGB format")
        
        # Auto-generate captions if not provided
        if not top_caption and not bottom_caption:
            logger.info("No captions provided, will auto-generate using Gemini...")
            # Check if Gemini is available
            gemini_available = gemini_api_key and gemini_api_key.strip() != ""
            
            if gemini_available:
                logger.info("Gemini available for caption generation")
                # Resize image for faster Gemini processing
                logger.info("Resizing image for faster Gemini processing...")
                max_size = 1024  # Maximum dimension for Gemini processing
                img_width, img_height = image.size
                
                if img_width > max_size or img_height > max_size:
                    # Calculate new dimensions maintaining aspect ratio
                    if img_width > img_height:
                        new_width = max_size
                        new_height = int((img_height * max_size) / img_width)
                    else:
                        new_height = max_size
                        new_width = int((img_width * max_size) / img_height)
                    
                    logger.info(f"Resizing from {img_width}x{img_height} to {new_width}x{new_height}")
                    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    logger.info("Image is already small enough, no resizing needed")
                    resized_image = image
                
                # Save resized image temporarily for Gemini
                logger.info("Preparing resized image for Gemini API...")
                temp_img = io.BytesIO()
                resized_image.save(temp_img, format="PNG", optimize=True)
                temp_img.seek(0)
                logger.info(f"Prepared image size: {len(temp_img.getvalue())} bytes")
            
                # Search for trending topics and meme references using Gemini
                logger.info("Searching for trending topics and meme references...")
                
                # Step 1: Analyze image and get context
                analysis_prompt = f"""
                Analyze this image and provide context for meme generation:
                
                What do you see in this image? (people, expressions, actions, objects, situations)
                What emotions or reactions are being shown?
                What type of situation or scenario is this?
                What would make this image relatable or funny?
                
                Return a brief analysis in 2-3 sentences.
                """
                
                try:
                    # Create content for image analysis
                    analysis_image_part = types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=temp_img.getvalue()
                        )
                    )
                    analysis_text_part = types.Part(text=analysis_prompt)
                    analysis_content = types.Content(parts=[analysis_text_part, analysis_image_part])
                    
                    logger.info("Analyzing image context...")
                    analysis_response = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[analysis_content]
                    )
                    image_context = analysis_response.text.strip()
                    logger.info(f"Image context: {image_context}")
                    
                    # Step 2: Search for Gen Z trending topics and meme references
                    if hinglish_support:
                        search_prompt = f"""
                         Based on this image context: "{image_context}"
                         
                         Search for Hinglish Gen Z specific content:
                         1. Current Hinglish Gen Z viral content, TikTok trends, and popular memes
                         2. Hinglish Gen Z humor styles (sarcastic, ironic, self-deprecating, chaotic)
                         3. Popular Hinglish Gen Z references (internet culture, social media, gaming, streaming)
                         4. Hinglish Gen Z relatable situations (school, work, relationships, social media)
                         5. Current Hinglish Gen Z slang, phrases, and expressions
                         
                         Focus on:
                         - Hinglish TikTok trends and viral content (last 3 months)
                         - Hinglish meme formats and expressions
                         - Hinglish internet culture references (Reddit, Twitter, Discord)
                         - Hinglish Gen Z humor patterns (chaotic good, relatable chaos, ironic humor)
                         - Current Hinglish Gen Z slang and expressions (yaar, bro, bhai, etc.)
                         - Hinglish gaming, streaming, and social media culture
                         - Hinglish Gen Z lifestyle and relatable situations
                         
                         Return 3-5 specific Hinglish Gen Z references, trends, or humor patterns that would make this image hilarious to Hinglish Gen Z.
                         Format as a simple list in Hinglish.
                         """
                    else:
                        search_prompt = f"""
                         Based on this image context: "{image_context}"
                         
                         Search for Gen Z specific content:
                         1. Current Gen Z viral content, TikTok trends, and popular memes
                         2. Gen Z humor styles (sarcastic, ironic, self-deprecating, chaotic)
                         3. Popular Gen Z references (internet culture, social media, gaming, streaming)
                         4. Gen Z relatable situations (school, work, relationships, social media)
                         5. Current Gen Z slang, phrases, and expressions
                         
                         Focus on:
                         - TikTok trends and viral content (last 3 months)
                         - Gen Z meme formats (Drake, Two Buttons, Wojak, etc.)
                         - Internet culture references (Reddit, Twitter, Discord)
                         - Gen Z humor patterns (chaotic good, relatable chaos, ironic humor)
                         - Current Gen Z slang and expressions
                         - Gaming, streaming, and social media culture
                         - Gen Z lifestyle and relatable situations
                         
                         Return 3-5 specific Gen Z references, trends, or humor patterns that would make this image hilarious to Gen Z.
                         Format as a simple list.
                         """
                    
                    logger.info("Searching for trending topics and references...")
                    search_response = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[types.Content(parts=[types.Part(text=search_prompt)])]
                    )
                    trending_references = search_response.text.strip()
                    logger.info(f"Trending references: {trending_references}")
                    
                    # Step 3: Generate Gen Z funny caption with context
                    if hinglish_support:
                        caption_prompt = f"""
                        Create a HILARIOUS Hinglish Gen Z meme caption for this image.
                        
                        IMAGE CONTEXT: {image_context}
                        HINGLISH GEN Z REFERENCES: {trending_references}
                        
                        Create a caption in Hinglish that:
                        - Uses Hinglish Gen Z humor (sarcastic, ironic, self-deprecating, chaotic)
                        - References current Hinglish TikTok trends, viral content, and internet culture
                        - Uses Hinglish Gen Z slang and expressions naturally (yaar, bro, bhai, etc.)
                        - Is relatable to Hinglish Gen Z lifestyle and experiences
                        - Has perfect timing and delivery for Hinglish Gen Z audience
                        - Uses popular Hinglish meme formats and references
                        - Feels authentic to Hinglish Gen Z voice and humor style
                        
                        Hinglish Gen Z humor characteristics:
                        - Self-deprecating and relatable in Hinglish
                        - Ironic and sarcastic in Hinglish
                        - References Hinglish internet culture and social media
                        - Uses current Hinglish slang and expressions
                        - Chaotic and unpredictable
                        - Relates to school, work, relationships, social media in Hinglish context
                        
                        Make it FUNNY for Hinglish Gen Z, not just descriptive. Use current Hinglish Gen Z humor, trends, and references.
                        
                        Return ONLY the caption text in Hinglish, nothing else.
                        """
                    else:
                        caption_prompt = f"""
                        Create a HILARIOUS Gen Z meme caption for this image.
                        
                        IMAGE CONTEXT: {image_context}
                        GEN Z REFERENCES: {trending_references}
                        
                        Create a caption that:
                        - Uses Gen Z humor (sarcastic, ironic, self-deprecating, chaotic)
                        - References current TikTok trends, viral content, and internet culture
                        - Uses Gen Z slang and expressions naturally
                        - Is relatable to Gen Z lifestyle and experiences
                        - Has perfect timing and delivery for Gen Z audience
                        - Uses popular Gen Z meme formats and references
                        - Feels authentic to Gen Z voice and humor style
                        
                        Gen Z humor characteristics:
                        - Self-deprecating and relatable
                        - Ironic and sarcastic
                        - References internet culture and social media
                        - Uses current slang and expressions
                        - Chaotic and unpredictable
                        - Relates to school, work, relationships, social media
                        
                        Make it FUNNY for Gen Z, not just descriptive. Use current Gen Z humor, trends, and references.
                        
                        Return ONLY the caption text, nothing else.
                        """
                    
                    logger.info("Generating funny caption with context...")
                    caption_response = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[types.Content(parts=[types.Part(text=caption_prompt)])]
                    )
                    generated_caption = caption_response.text.strip()
                    
                except Exception as e:
                    logger.error(f"Error in enhanced caption generation: {e}")
                    # Fallback to original prompt
                    logger.info("Using fallback caption generation...")
                    fallback_prompt = """
                Analyze this image and create a funny, viral-style meme caption. 
                    
                    Make it:
                - Humorous and relatable
                - Short and punchy (max 50 characters per line)
                    - Something that would make people laugh and share
                    - Use current humor and trends if possible
                    
                    Return ONLY the caption text, nothing else.
                    """
                    
                    fallback_content = types.Content(
                        parts=[
                            types.Part(text=fallback_prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=temp_img.getvalue()
                                )
                            )
                        ]
                    )
                    
                    fallback_response = client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[fallback_content]
                    )
                    generated_caption = fallback_response.text.strip()
                # Caption generation completed, now process the result
                if generated_caption and len(generated_caption.strip()) > 0:
                    logger.info(f"Generated caption: '{generated_caption}'")
                    # Split into top and bottom if it's long enough
                    if len(generated_caption) > 30 and '\n' not in generated_caption:
                        logger.info("Splitting long caption into top and bottom...")
                        # Try to split at a natural break point
                        words = generated_caption.split()
                        if len(words) > 3:
                            mid = len(words) // 2
                            top_caption = ' '.join(words[:mid])
                            bottom_caption = ' '.join(words[mid:])
                            logger.info(f"Top caption: '{top_caption}'")
                            logger.info(f"Bottom caption: '{bottom_caption}'")
                        else:
                            top_caption = generated_caption
                            logger.info(f"Using single caption: '{top_caption}'")
                    else:
                        top_caption = generated_caption
                        logger.info(f"Using single caption: '{top_caption}'")
                else:
                    logger.warning("No caption generated, using fallback")
                    # Fallback to generic captions
                    top_caption = "When you see this image"
                    bottom_caption = "But you can't look away"
            else:
                logger.info("No Gemini available, using generic captions")
                # No Gemini available, use generic captions
                top_caption = "When you see this image"
                bottom_caption = "But you can't look away"
        else:
            logger.info("User provided captions, using them directly")
            # User provided captions, use them as-is
        
        logger.info("Normalizing captions to ALL CAPS and setting up drawing...")
        # Force ALL CAPS for stronger meme styling
        if top_caption:
            top_caption = top_caption.upper()
        if bottom_caption:
            bottom_caption = bottom_caption.upper()

        logger.info("Setting up image drawing...")
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        # Ensure image dimensions are available early
        img_width, img_height = image.size
        logger.info(f"Image dimensions: {img_width}x{img_height}")
        
        # Accessibility features
        if accessibility_mode:
            logger.info("Accessibility mode enabled - applying accessibility features")
            
            # Auto-scale font size based on image dimensions
            min_dimension = min(img_width, img_height)
            if min_dimension < 400:
                font_size = max(30, min_dimension // 10)
            elif min_dimension < 800:
                font_size = max(40, min_dimension // 15)
            else:
                font_size = max(50, min_dimension // 20)
            
            logger.info(f"Auto-scaled font size to {font_size} for accessibility")
            
            # Auto-detect background brightness for better contrast
            # Sample pixels from corners and center to determine background
            sample_points = [
                (img_width//4, img_height//4),
                (3*img_width//4, img_height//4),
                (img_width//4, 3*img_height//4),
                (3*img_width//4, 3*img_height//4),
                (img_width//2, img_height//2)
            ]
            
            total_brightness = 0
            for x, y in sample_points:
                if 0 <= x < img_width and 0 <= y < img_height:
                    pixel = image.getpixel((x, y))
                    if isinstance(pixel, tuple):
                        # RGB values
                        brightness = sum(pixel[:3]) / 3
                    else:
                        # Grayscale
                        brightness = pixel
                    total_brightness += brightness
            
            avg_brightness = total_brightness / len(sample_points)
            logger.info(f"Average background brightness: {avg_brightness}")
            
            # Set high contrast colors based on background brightness
            if avg_brightness > 128:  # Light background
                text_color = "black"
                stroke_color = "white"
                stroke_width = max(6, stroke_width)  # Thicker outline for better contrast
            else:  # Dark background
                text_color = "white"
                stroke_color = "black"
                stroke_width = max(6, stroke_width)  # Thicker outline for better contrast
            
            logger.info(f"Accessibility colors: text={text_color}, stroke={stroke_color}, width={stroke_width}")
        
        # Handle high contrast mode if explicitly requested
        if text_color == "high_contrast":
            logger.info("High contrast mode enabled")
            text_color = "white"
            stroke_color = "black"
            stroke_width = max(6, stroke_width)
        
        # Handle auto stroke color
        if stroke_color == "auto":
            logger.info("Auto stroke color mode enabled")
            if text_color == "white":
                stroke_color = "black"
            else:
                stroke_color = "white"
            stroke_width = max(5, stroke_width)
        
        # Helpers: load bold font and fit text to width
        logger.info("Preparing bold font loader and auto-fit helpers...")

        def load_bold_font(desired_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            font_candidates = [
                # Windows common
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/ARIALBD.TTF",
                # Linux common
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                # macOS common
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
                # Generic names for PIL to resolve if available in cwd
                "arialbd.ttf",
                "DejaVuSans-Bold.ttf",
            ]
            for path in font_candidates:
                try:
                    return ImageFont.truetype(path, desired_size)
                except (OSError, IOError):
                    continue
            # Fallback to Arial regular, then PIL default
            try:
                return ImageFont.truetype("arial.ttf", desired_size)
            except (OSError, IOError):
                return ImageFont.load_default()

        def fit_font_to_width(text: str, max_width: int, start_size: int) -> ImageFont.ImageFont:
            size = max(10, start_size)
            font_local = load_bold_font(size)
            # If too wide, decrease until fits or minimum size
            bbox = draw.textbbox((0, 0), text, font=font_local)
            text_width = bbox[2] - bbox[0]
            while text_width > max_width and size > 10:
                size = max(10, int(size * 0.9))
                font_local = load_bold_font(size)
                bbox = draw.textbbox((0, 0), text, font=font_local)
                text_width = bbox[2] - bbox[0]
            return font_local
        
        # Image dimensions already computed above
        
        # Helper function to draw text with stroke and accessibility features
        def draw_text_with_stroke(text, position, font, fill_color, stroke_color, stroke_width):
            x, y = position
            current_draw = draw  # Use the outer draw object
            
            # Accessibility: Add extra padding for better readability
            if accessibility_mode:
                # Add a semi-transparent background for better contrast
                bbox = current_draw.textbbox((x, y), text, font=font)
                padding = 10
                bg_rect = [
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding
                ]
                
                # Create a semi-transparent background
                from PIL import ImageColor
                bg_color = ImageColor.getrgb('black') + (128,)  # Semi-transparent black
                bg_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg_image)
                bg_draw.rectangle(bg_rect, fill=bg_color)
                
                # Composite the background onto the main image
                image.paste(bg_image, (0, 0), bg_image)
                current_draw = ImageDraw.Draw(image)  # Refresh draw object
            
            # Draw stroke (outline) with enhanced accessibility
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:  # Skip the center pixel
                        current_draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
            
            # Draw the main text
            current_draw.text((x, y), text, font=font, fill=fill_color)
        
        # Add top caption if provided
        if top_caption:
            logger.info(f"Adding top caption: '{top_caption}'")
            # Auto-fit bold font to image width
            max_text_width = int(img_width * 0.9)
            font_top = fit_font_to_width(top_caption, max_text_width, font_size)
            # Get text size
            bbox = draw.textbbox((0, 0), top_caption, font=font_top)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text horizontally
            x = (img_width - text_width) // 2
            y = 20  # 20 pixels from top
            logger.info(f"Top caption position: ({x}, {y})")
            
            draw_text_with_stroke(top_caption, (x, y), font_top, text_color, stroke_color, stroke_width)
            logger.info("Top caption added")
        
        # Add bottom caption if provided
        if bottom_caption:
            logger.info(f"Adding bottom caption: '{bottom_caption}'")
            # Auto-fit bold font to image width
            max_text_width = int(img_width * 0.9)
            font_bottom = fit_font_to_width(bottom_caption, max_text_width, font_size)
            # Get text size
            bbox = draw.textbbox((0, 0), bottom_caption, font=font_bottom)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text horizontally
            x = (img_width - text_width) // 2
            y = img_height - text_height - 20  # 20 pixels from bottom
            logger.info(f"Bottom caption position: ({x}, {y})")
            
            draw_text_with_stroke(bottom_caption, (x, y), font_bottom, text_color, stroke_color, stroke_width)
            logger.info("Bottom caption added")
        
        # Convert back to base64
        logger.info("Saving final image...")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"Final image size: {len(img_base64)} characters")
        
        logger.info("Image captioning completed successfully")
        return [ImageContent(type="image", mimeType="image/png", data=img_base64)]
        
    except Exception as e:
        logger.error(f"Error in add_caption_to_image: {str(e)}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to add caption to image: {str(e)}"))

# --- Run MCP Server ---
async def main():
    logger.info("Starting MCP server on http://0.0.0.0:8086")
    logger.info("Available tools:")
    logger.info("  - validate() - Authentication tool")
    logger.info("  - job_finder() - Job analysis and search")
    logger.info("  - create_imgflip_meme() - Create memes using Imgflip templates")
    logger.info("  - screenshot_meme_generator() - Analyze screenshots and create memes")
    logger.info("  - make_img_black_and_white() - Image conversion")
    logger.info("  - add_caption_to_image() - Enhanced meme creation with AI")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
